"""Bounded append-only compressed session recorder and verified prefix reader."""
from __future__ import annotations
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import re
import shutil
import threading
import time
import uuid
from time import perf_counter_ns
from daytrading.reliability_metrics import StageMetrics, timed_lock
import zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from .codec import pack, SECRET

SCHEMA=1
IDENTIFIER=re.compile(r'^[A-Za-z0-9_-]{1,100}$')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
def session_root():return Path(os.environ.get('DAYTRADING_DATA_DIR','data'))/'replay-sessions'
def safe_session(root,session_id):
    if not IDENTIFIER.fullmatch(str(session_id or '')):raise ValueError('Invalid recorded session ID')
    path=Path(root)/session_id
    if path.is_symlink() or path.resolve().parent!=Path(root).resolve():raise ValueError('Invalid session path')
    return path

def freeze_source(destination):
    src=Path(__file__).resolve().parents[1];files={str(p.relative_to(src.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src.rglob('*') if p.is_file() and p.suffix in ('.py','.html','.js','.css','.json')}
    build=digest(files);archive=destination/(build+'.zip');destination.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        temp=archive.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):z.write(src.parent/name,name)
        temp.replace(archive)
    return dict(build_id=build,source_files=files,artifact=archive.name,artifact_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())

def dependencies():
    import sys,platform
    return dict(python=sys.version,implementation=platform.python_implementation(),packages={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions() if d.metadata['Name']})

class SessionRecorder:
    def __init__(self, root, *, metadata, max_queue=20000, chunk_size=20000, flush_interval=30., min_free_bytes=256*1024*1024, max_queue_bytes=64*1024*1024, max_event_bytes=16*1024*1024):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.session_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:10]
        self.path=self.root/self.session_id;self.path.mkdir();self._lock=threading.RLock();self._timings=StageMetrics();self._queue=queue.Queue(max_queue);self._sequence=0;self._active_ops=set();self._closed=False;self._failed=False;self._queued_bytes=0;self._max_queue_bytes=max_queue_bytes;self._max_event_bytes=max_event_bytes;self._last_commit_monotonic=time.monotonic();self._queue_high_water=0;self._byte_high_water=0;self._chunk_size=chunk_size;self._flush_interval=flush_interval;self._min_free=min_free_bytes
        self.manifest=dict(schema_version=SCHEMA,session_id=self.session_id,state='recording',created_at=datetime.now(timezone.utc).isoformat(),session_date=datetime.now(ZoneInfo('America/New_York')).date().isoformat(),metadata=metadata,segments=[],committed_sequence=0,failures=[],source=freeze_source(self.root/'artifacts'),dependencies=dependencies())
        atomic(self.path/'manifest.json',self.manifest)
        self._thread=threading.Thread(target=self._writer,name='paper-session-recorder',daemon=True);self._thread.start()
    def fail(self,reason):
        with self._lock:
            self._failed=True
            if reason not in self.manifest['failures']:self.manifest['failures'].append(str(reason)[:400])
            self.manifest['state']='incomplete'
            self._failure_manifest_dirty=True
    def record(self,kind,data,*,exchange_ts=None,operation_id=None,received_at=None,monotonic_ns=None):
        with timed_lock(self._lock,self._timings,'recorder_lock'):
            if self._closed or self._failed:return None
            self._sequence+=1
            try:
                row=dict(sequence=self._sequence,kind=kind,received_at=(received_at or datetime.now(timezone.utc)).isoformat(),monotonic_ns=monotonic_ns if monotonic_ns is not None else time.monotonic_ns(),exchange_ts=exchange_ts.isoformat() if exchange_ts else None,operation_id=operation_id,data=data)
                # Snapshot in producer thread: later mutation cannot rewrite history.
                serialization_started=perf_counter_ns()
                line=json.dumps(row,separators=(',',':'),allow_nan=False)
                self._timings.add('serialization',perf_counter_ns()-serialization_started)
                size=len(line.encode())
                if size>self._max_event_bytes or self._queued_bytes+size>self._max_queue_bytes:
                    self.fail('recorder byte capacity exceeded');return None
                # Carry tiny immutable indexing fields beside the snapshotted
                # line. The writer need not parse every large graph a second
                # time just to recover its sequence/operation identity.
                self._queue.put_nowait((line,size,kind,operation_id,self._sequence,row['received_at']));self._queued_bytes+=size
                self._queue_high_water=max(self._queue_high_water,self._queue.qsize());self._byte_high_water=max(self._byte_high_water,self._queued_bytes)
                if kind=='operation_start':self._active_ops.add(operation_id)
                elif kind=='operation_end':self._active_ops.discard(operation_id)
                return self._sequence
            except queue.Full:self.fail('recorder queue overflow')
            except (TypeError,ValueError) as exc:self.fail('event serialization: '+str(exc))
            return None
    def _writer(self):
        batch=[];batch_bytes=0;active=set();completed_operation=0;index=0;last_flush=time.monotonic()
        try:
            while True:
                if getattr(self,'_failure_manifest_dirty',False):
                    with self._lock:
                        atomic(self.path/'manifest.json',self.manifest)
                        self._failure_manifest_dirty=False
                try:item=self._queue.get(timeout=.5)
                except queue.Empty:item='FLUSH'
                if item not in ('FLUSH',None):
                    with self._lock:self._queued_bytes-=item[1]
                    batch.append(item);batch_bytes+=item[1]
                if batch and (item is None or len(batch)>=self._chunk_size or batch_bytes>=self._max_queue_bytes or time.monotonic()-last_flush>=self._flush_interval):
                    if shutil.disk_usage(self.path).free<self._min_free:raise OSError('insufficient recorder disk reserve')
                    flush_started=perf_counter_ns()
                    index+=1;name=f'events-{index:08d}.jsonl.gz';path=self.path/name
                    # Lossless fast compression: same complete JSONL rows and
                    # reader/schema, with more disk bytes in exchange for less
                    # competition with live market delivery. Disk guard stays.
                    with gzip.open(path.with_suffix('.tmp'),'wt',encoding='utf-8',compresslevel=1) as f:f.write('\n'.join(row[0] for row in batch)+'\n')
                    path.with_suffix('.tmp').replace(path)
                    for line,size,kind,op,sequence,received in batch:
                        if kind=='operation_start':active.add(op)
                        elif kind=='operation_end':
                            active.discard(op)
                            if not active:completed_operation=sequence
                    with self._lock:
                        self.manifest['committed_operation_sequence']=completed_operation
                        self.manifest['segments'].append(dict(file=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first=batch[0][4],last=batch[-1][4]))
                        self.manifest['committed_sequence']=batch[-1][4];self.manifest.setdefault('coverage_start',batch[0][5]);self.manifest['coverage_end']=batch[-1][5];atomic(self.path/'manifest.json',self.manifest);self._last_commit_monotonic=time.monotonic()
                    self._timings.add('writer_flush',perf_counter_ns()-flush_started)
                    batch=[];batch_bytes=0;last_flush=time.monotonic()
                if item is None:return
        except Exception as exc:self.fail('recorder writer: '+str(exc))
    def close(self,timeout=5):
        with self._lock:
            if self._closed:return
            self._closed=True
        try:self._queue.put(None,timeout=timeout)
        except queue.Full:self.fail('recorder close queue overflow')
        self._thread.join(timeout)
        with self._lock:
            if self._thread.is_alive():self.fail('recorder flush timeout')
            if self._active_ops:self.fail('native operations incomplete at recording close')
            if self.manifest['committed_sequence']!=self._sequence:self.fail('uncommitted recorder events')
            self.manifest['state']='incomplete' if self._failed else 'sealed';self.manifest['closed_at']=datetime.now(timezone.utc).isoformat()
            try:atomic(self.path/'manifest.json',self.manifest)
            except OSError:self._failed=True
    def status(self):
        with self._lock:
            return dict(session_id=self.session_id,state=self.manifest['state'],events=self._sequence,
                committed=self.manifest['committed_sequence'],failures=list(self.manifest['failures']),
                stage_timings=self._timings.status(),queue_depth=self._queue.qsize(),queue_capacity=self._queue.maxsize,
                queued_bytes=self._queued_bytes,byte_capacity=self._max_queue_bytes,
                queue_high_water=self._queue_high_water,byte_high_water=self._byte_high_water,
                writer_alive=self._thread.is_alive(),commit_age_seconds=max(0.,time.monotonic()-self._last_commit_monotonic))

def read_session(root,session_id,manifest=None):
    path=safe_session(root,session_id);manifest=manifest or json.loads((path/'manifest.json').read_text())
    if manifest.get('schema_version')!=SCHEMA:raise ValueError('Unsupported recorded session schema')
    events=[];expected=1
    for segment in manifest['segments']:
        name=segment['file']
        if not re.fullmatch(r'events-\d{8}\.jsonl\.gz',name):raise ValueError('Invalid segment name')
        p=path/name
        if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=segment['sha256']:raise ValueError('Recorded segment hash mismatch')
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['sequence']!=expected:raise ValueError('Recorded event sequence gap')
                expected+=1;events.append(row)
    if expected-1!=manifest['committed_sequence']:raise ValueError('Recorded prefix is incomplete')
    return manifest,events

def list_sessions(root=None):
    root=Path(root or session_root());result=[]
    if root.exists():
        for p in sorted(root.glob('*/manifest.json'),reverse=True):
            try:
                m=json.loads(p.read_text());row={k:m.get(k) for k in ('session_id','session_date','state','created_at','committed_sequence','failures')};row['build_id']=m.get('source',{}).get('build_id');result.append(row)
            except (OSError,ValueError):continue
    return result
