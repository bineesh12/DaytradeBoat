"""JSON-only state snapshots. No pickle, executable payloads or client traversal."""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from uuid import UUID
import importlib
import math
import re
import queue
import threading
from functools import lru_cache
from types import SimpleNamespace
from zoneinfo import ZoneInfo
from daytrading.models import Bar, Quote, Tick, CompletedBar

# A protocol-1 encoding shortcut for exact immutable native values. No cache,
# omitted fields, changed node order, checkpoint or alternate recording format.
_NATIVE_VALUES = frozenset((Bar, Quote, Tick, CompletedBar))

SECRET = re.compile(r'(api.?key|secret|password|authorization|access.?token|refresh.?token|(^|_)token($|_)|private.?key|session.?cookie)', re.I)
RUNTIME_MODULES = ('_thread', 'threading', 'queue', 'logging', 'sqlite3', 'requests', 'urllib3', 'concurrent', 'socket', 'alpaca')
_LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))

@lru_cache(maxsize=4096)
def _plain_secret_key(key):
    return bool(SECRET.search(key))

def _secret_key(key):
    return _plain_secret_key(key) if type(key) is str else bool(SECRET.search(key))

class StateError(ValueError): pass


def resolve(name):
    module, _, qual = name.partition(':')
    if not module.startswith('daytrading.') or '<' in qual or not qual:
        raise StateError('Unapproved state type '+name)
    value = importlib.import_module(module)
    for part in qual.split('.'):
        if part.startswith('__'): raise StateError('Invalid state type')
        value = getattr(value, part)
    if not isinstance(value, type): raise StateError('State type is not a class')
    return value

class StateCodec:
    """Preserve graph identity; runtime callbacks/locks are supplied by the factory.

    Restoring merges nodes into the freshly constructed native runner, preserving
    its bound callbacks, locks and injected I/O. New model value objects are safe
    daytrading classes only. Unsupported state is a capture error, never dropped.
    """
    def __init__(self, externals=None):
        self.externals = externals or {}
        self.ids = {id(v):k for k,v in self.externals.items() if v is not None}
        for key, value in self.externals.items():
            target=vars(value).get('_target') if hasattr(value,'__dict__') else None
            if target is not None:self.ids[id(target)]=key
        # Discovery symbol lists are large and mostly unchanged between ticks.
        # Fast paths still encode their current contents on every operation;
        # no state cache, skipped field, event or mutation-invalidation assumption.
        self._primitive_externals = any(type(v) in (str,bool,int,float) for v in self.externals.values())
        self.memo = {}; self.nodes = []
    def encode(self, value):
        ids, memo, nodes = self.ids, self.memo, self.nodes
        def enc(v):
            identity, cls = id(v), type(v)
            if identity in ids:return {'external':ids[identity]}
            # Exact built-ins only: str/int Enums and subclasses retain the
            # original typed encoding below, including secret redaction rules.
            if v is None or cls is str or cls is bool or cls is int:return v
            if cls is float:return v if math.isfinite(v) else {'float':repr(v)}
            if cls is datetime:return {'datetime':v.isoformat()}
            if cls in _NATIVE_VALUES:
                if identity in memo:return {'ref':memo[identity]}
                idx=len(nodes);memo[identity]=idx;node={};nodes.append(node)
                node.update(kind='object',type=cls.__module__+':'+cls.__qualname__,
                    fields={k:enc(x) for k,x in vars(v).items() if not _secret_key(k)})
                return {'ref':idx}
            # Exact built-in containers cannot be enums, SDK values, locks or
            # models. Keep the same graph/memo order without those type checks.
            if cls is dict or cls is list or cls is tuple:
                if identity in memo:return {'ref':memo[identity]}
                idx=len(nodes);memo[identity]=idx;node={};nodes.append(node)
                if cls is dict:
                    node.update(kind='dict',counter=False,default=None,
                        items=[[enc(k),enc(x)] for k,x in v.items()
                               if not (isinstance(k,str) and _secret_key(k))])
                else:
                    items=list(v)
                    if (self._primitive_externals or (items and type(items[0]) is not str)
                            or not set(map(type,items)) <= {str}):
                        items=[enc(x) for x in items]
                    node.update(kind=cls.__name__,items=items,maxlen=None)
                return {'ref':idx}
            if type(v).__module__=='alpaca.data.timeframe' and type(v).__name__=='TimeFrame':return {'alpaca_timeframe':[v.amount,v.unit.value]}
            if isinstance(v, Enum) and type(v).__module__.startswith('alpaca'):return enc(v.value)
            if isinstance(v, Enum):return {'enum':type(v).__module__+':'+type(v).__qualname__,'value':enc(v.value)}
            if v is None or isinstance(v,(str,bool,int)):return v
            if isinstance(v,float):return v if math.isfinite(v) else {'float':repr(v)}
            if isinstance(v,datetime):return {'datetime':v.isoformat()}
            if isinstance(v,date):return {'date':v.isoformat()}
            if isinstance(v,time):return {'time':v.isoformat()}
            if isinstance(v,timedelta):return {'timedelta':v.total_seconds()}
            if isinstance(v,Decimal):return {'decimal':str(v)}
            if isinstance(v,Path):return {'path':str(v)}
            if isinstance(v,UUID):return {'uuid':str(v)}
            if isinstance(v,ZoneInfo):return {'zone':v.key}
            if isinstance(v,timezone):return {'offset':v.utcoffset(None).total_seconds()}
            if isinstance(v,threading.Event):return {'thread_event':v.is_set()}
            if type(v) in _LOCK_TYPES:return {'thread_lock':type(v).__name__}
            if isinstance(v,queue.Queue):
                with v.mutex:return {'queue_type':type(v).__name__,'maxsize':v.maxsize,'items':[enc(x) for x in v.queue]}
            if callable(v) or (type(v).__module__.startswith(RUNTIME_MODULES) and not hasattr(v,'model_dump')):return {'runtime':True}
            if id(v) in self.memo:return {'ref':self.memo[id(v)]}
            idx=len(self.nodes);self.memo[id(v)]=idx;node={};self.nodes.append(node)
            literal_items = list(v) if type(v) in (list,tuple) and not self._primitive_externals else None
            if literal_items is not None and set(map(type, literal_items)) <= {str}:
                # Primitive strings have no graph nodes. Preserve the collection
                # node, item order, aliases and exact legacy JSON representation.
                node.update(kind=type(v).__name__,items=literal_items,maxlen=None)
            elif isinstance(v,(dict,Counter,defaultdict)):
                node.update(kind='dict',counter=isinstance(v,Counter),default=(getattr(v,'default_factory',None).__name__ if getattr(v,'default_factory',None) in (int,float,list,dict,set) else None),items=[[enc(k),enc(x)] for k,x in v.items() if not (isinstance(k,str) and _secret_key(k))])
            elif isinstance(v,(list,tuple,set,frozenset,deque)):
                kind='deque' if isinstance(v,deque) else type(v).__name__
                if hasattr(v,'_fields'):node.update(kind='object',type=type(v).__module__+':'+type(v).__qualname__,namedtuple=True,fields={k:enc(getattr(v,k)) for k in v._fields})
                else:node.update(kind=kind,items=[enc(x) for x in (sorted(v,key=repr) if isinstance(v,(set,frozenset)) else v)],maxlen=v.maxlen if isinstance(v,deque) else None)
            elif type(v).__module__.startswith('daytrading.') and hasattr(v,'__dict__'):
                node.update(kind='object',type=type(v).__module__+':'+type(v).__qualname__,fields={k:enc(x) for k,x in vars(v).items() if not _secret_key(k)})
            elif isinstance(v,SimpleNamespace):node.update(kind='namespace',fields={k:enc(x) for k,x in vars(v).items() if not _secret_key(k)})
            elif hasattr(v,'model_dump'):
                # SDK output values replay as data-only namespaces, not clients.
                node.update(kind='namespace',fields={k:enc(getattr(v,k)) for k in getattr(type(v),'model_fields',{}) if not _secret_key(k)} if getattr(type(v),'model_fields',None) else {k:enc(x) for k,x in v.model_dump().items() if not _secret_key(k)})
            else:raise StateError('Unsupported state: '+type(v).__module__+'.'+type(v).__qualname__)
            return {'ref':idx}
        root=enc(value)
        return {'root':root,'nodes':self.nodes}
    def decode(self, payload, existing=None):
        nodes=payload['nodes'];memo={}
        def dec(v, old=None):
            if not isinstance(v,dict):return v
            if 'external' in v:
                if v['external'] not in self.externals:raise StateError('Missing dependency '+v['external'])
                return self.externals[v['external']]
            if 'thread_event' in v:
                target=old if isinstance(old,threading.Event) else threading.Event()
                target.set() if v['thread_event'] else target.clear()
                return target
            if 'thread_lock' in v:return old if old is not None else (threading.RLock() if v['thread_lock']=='RLock' else threading.Lock())
            if 'queue_type' in v:
                # The recording FIFO adds receipt capture, not different queue
                # ordering. The offline audit installs its own receipt-aware
                # queue; graph restoration only needs these stored FIFO items.
                target={'Queue':queue.Queue,'RecordedEventQueue':queue.Queue,'PriorityQueue':queue.PriorityQueue,'LifoQueue':queue.LifoQueue}[v['queue_type']](v['maxsize'])
                for item in v['items']:target.put_nowait(dec(item))
                return target
            if 'alpaca_timeframe' in v:
                from alpaca.data.timeframe import TimeFrame,TimeFrameUnit
                return TimeFrame(v['alpaca_timeframe'][0],TimeFrameUnit(v['alpaca_timeframe'][1]))
            if 'runtime' in v:return old
            if 'enum' in v:return resolve(v['enum'])(dec(v['value']))
            if 'datetime' in v:return datetime.fromisoformat(v['datetime'])
            if 'date' in v:return date.fromisoformat(v['date'])
            if 'time' in v:return time.fromisoformat(v['time'])
            if 'timedelta' in v:return timedelta(seconds=v['timedelta'])
            if 'decimal' in v:return Decimal(v['decimal'])
            if 'path' in v:return Path(v['path'])
            if 'uuid' in v:return UUID(v['uuid'])
            if 'zone' in v:return ZoneInfo(v['zone'])
            if 'offset' in v:return timezone(timedelta(seconds=v['offset']))
            if 'float' in v:return float(v['float'])
            i=v['ref']
            if i in memo:return memo[i]
            node=nodes[i];kind=node['kind']
            if kind=='dict':
                original=dict(old) if isinstance(old,dict) else {}
                target=old if isinstance(old,dict) else (Counter() if node.get('counter') else defaultdict({'int':int,'float':float,'dict':dict,'list':list,'set':set}[node['default']]) if node.get('default') else {})
                memo[i]=target;target.clear()
                for k,x in node['items']:
                    key=dec(k);target[key]=dec(x,original.get(key))
            elif kind in ('object','namespace'):
                cls=resolve(node['type']) if kind=='object' else SimpleNamespace
                if node.get('namedtuple'):
                    target=cls(**{k:dec(x) for k,x in node['fields'].items()});memo[i]=target
                else:
                    target=old if isinstance(old,cls) else cls.__new__(cls);memo[i]=target
                    # A newly allocated delegating object may not yet have its
                    # delegate. getattr would execute __getattr__ recursively
                    # (for example the native checkpoint event adapter). Only
                    # reuse actually stored fields; never run user lookup code
                    # while restoring the captured object graph.
                    prior_fields = vars(target)
                    for k,x in node['fields'].items():object.__setattr__(target,k,dec(x,prior_fields.get(k)))
            else:
                values=[];memo[i]=values
                for j,x in enumerate(node['items']):values.append(dec(x,old[j] if isinstance(old,(list,tuple,deque)) and j<len(old) else None))
                target={'list':list,'tuple':tuple,'set':set,'frozenset':frozenset,'deque':lambda xs:deque(xs,maxlen=node.get('maxlen'))}[kind](values);memo[i]=target
            return target
        return dec(payload['root'],existing)


def pack(value, externals=None):return StateCodec(externals).encode(value)
def unpack(value, externals=None, existing=None):return StateCodec(externals).decode(value,existing)
