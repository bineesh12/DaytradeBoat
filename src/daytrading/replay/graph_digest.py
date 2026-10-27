"""Exact canonical state hashing with bounded, value-checked string-list reuse.

Every operation still packs its complete current state. Only canonical JSON
encoding of large plain string collections is reused, keyed by their complete
immutable current contents. In-place edits, ordering and alias changes therefore
remain visible to the original SHA-256 state check. Never caches trade decisions.
"""
from collections import OrderedDict
import hashlib,json,threading


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _update_owned_digest(result, data):
    """Keep tiny operation checks from handing the GIL to recorder workers.

    CPython/OpenSSL releases the GIL for updates above 2047 bytes. A repeated
    state check contains several large cached discovery nodes, so each update
    can wait another thread's interpreter quantum despite negligible SHA work.
    Bounded updates consume the exact same bytes, without that hand-off. This
    does not cache mutable state, omit operations or weaken the SHA-256 check.
    """
    if len(data) <= 2047:
        result.update(data)
        return
    view = memoryview(data)
    for start in range(0, len(view), 2047):
        result.update(view[start:start+2047])


class CanonicalGraphDigest:
    def __init__(self,capacity=8,minimum_items=64):
        self.capacity=capacity;self.minimum_items=minimum_items
        self._cache=OrderedDict();self._lock=threading.Lock()

    def _cached_node(self,node):
        if type(node) is not dict:return None
        items=node.get('items')
        if (set(node)=={'kind','items','maxlen'} and node['kind'] in ('list','tuple')
                and node['maxlen'] is None and type(items) is list and len(items)>=self.minimum_items):
            # Snapshot once before classification; do not race mutable contents.
            snapshot=tuple(items)
            if set(map(type, snapshot)) <= {str}:
                key=(node['kind'],snapshot)
                with self._lock:
                    result=self._cache.pop(key,None)
                    if result is None:result=_json(dict(node,items=list(snapshot)))
                    self._cache[key]=result
                    while len(self._cache)>self.capacity:self._cache.popitem(last=False)
                return result
        return None

    def digest(self,graph):
        if type(graph) is not dict or set(graph)!={'root','nodes'} or type(graph['nodes']) is not list:
            return hashlib.sha256(_json(graph)).hexdigest()
        # Same bytes as json.dumps(graph, sort_keys=True, separators=(',', ':'),
        # allow_nan=False): nodes sorts before root; node serialization is unchanged.
        result=hashlib.sha256(b'{"nodes":[')
        pending=[];written=False
        def flush():
            nonlocal written
            if not pending:return
            if written:result.update(b',')
            _update_owned_digest(result, _json(pending)[1:-1]);pending.clear();written=True
        for node in graph['nodes']:
            cached=self._cached_node(node)
            if cached is None:pending.append(node);continue
            flush()
            if written:result.update(b',')
            _update_owned_digest(result, cached);written=True
        flush()
        result.update(b'],"root":');result.update(_json(graph['root']));result.update(b'}')
        return result.hexdigest()
