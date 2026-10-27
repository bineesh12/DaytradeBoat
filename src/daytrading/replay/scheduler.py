"""Deterministic virtual actors: synchronous native code yields only at I/O/waits.

No broker outcomes, seed selection or wall-clock sleeps enter scheduling. A
single native segment runs at a time; network waits release it to other actors.
"""
from __future__ import annotations
from datetime import timedelta
import threading,time

class CaptureBoundary(BaseException):
    """Right censor at the final recording timestamp, never invent future data."""

class VirtualScheduler:
    def __init__(self,clock,end):
        self.clock=clock;self.end=end;self.condition=threading.Condition();self.actors={};self.active=None;self.closed=False;self.error=None;self.local=threading.local();clock.scheduler=self
    def spawn(self,name,at,fn):
        ready=threading.Event();state=dict(name=name,at=at,done=False,turn=False,ready=ready)
        with self.condition:self.actors[name]=state
        def run():
            self.local.state=state
            try:
                self.wait(at)
                fn()
            except CaptureBoundary:pass
            except BaseException as exc:
                with self.condition:self.error=exc
            finally:
                with self.condition:
                    state['done']=True
                    if self.active is state:self.active=None
                    self.condition.notify_all()
        state['thread']=threading.Thread(target=run,name='predict-'+name,daemon=True);state['thread'].start();ready.wait(5)
    def wait(self,at):
        state=getattr(self.local,'state',None)
        if state is None:self.clock.now=max(self.clock.now,at);return
        with self.condition:
            state['at']=max(self.clock.now,at);state['turn']=False
            if self.active is state:self.active=None
            state['ready'].set();self.condition.notify_all()
            while not state['turn'] and not self.closed:self.condition.wait()
            if self.closed:raise CaptureBoundary()
    def run(self):
        with self.condition:
            while not self.error:
                if self.active is not None:
                    self.condition.wait(1)
                    continue
                waiting=[s for s in self.actors.values() if not s['done']]
                if not waiting:break
                state=min(waiting,key=lambda s:(s['at'],s['name']))
                if state['at']>self.end:break
                self.clock.now=max(self.clock.now,state['at']);self.active=state;state['turn']=True;self.condition.notify_all()
                self.condition.wait(.01)
            self.closed=True;self.condition.notify_all()
        for state in self.actors.values():state['thread'].join(2)
        if self.error:raise self.error


class VirtualEvent:
    """threading.Event interface with virtual-time waits and immediate wakeups."""
    def __init__(self,scheduler,initial=False):self.scheduler=scheduler;self.flag=initial;self.waiters=[]
    def is_set(self):return self.flag
    def set(self):
        with self.scheduler.condition:
            self.flag=True
            for state in self.waiters:state['at']=min(state['at'],self.scheduler.clock.now)
            self.scheduler.condition.notify_all()
    def clear(self):self.flag=False
    def wait(self,timeout=None):
        if self.flag:return True
        state=getattr(self.scheduler.local,'state',None)
        if state is None:raise RuntimeError('Virtual event wait outside a scheduled native actor')
        self.waiters.append(state)
        try:self.scheduler.wait(self.scheduler.clock.now+timedelta(seconds=timeout if timeout is not None else 86400))
        finally:self.waiters.remove(state)
        return self.flag
