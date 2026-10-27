"""Worker-only write barrier. Replay may write solely into its disposable job."""
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
import builtins,io,os,sqlite3

@contextmanager
def worker_write_barrier(root):
 root=Path(root).resolve()
 def check(path):
  if isinstance(path,int):return
  candidate=Path(path).resolve()
  if candidate!=root and root not in candidate.parents:raise PermissionError('Replay write outside its isolated job is prohibited')
 real_open=builtins.open;real_io_open=io.open;real_connect=sqlite3.connect
 def opener(real):
  def run(path,mode='r',*a,**kw):
   if any(c in mode for c in 'wax+'):check(path)
   return real(path,mode,*a,**kw)
  return run
 def connection(path,*a,**kw):
  if str(path)!=':memory:':check(path)
  return real_connect(path,*a,**kw)
 def mutation(real,two=False):
  def run(path,*a,**kw):
   check(path)
   if two and a:check(a[0])
   return real(path,*a,**kw)
  return run
 with ExitStack() as stack:
  stack.enter_context(patch('builtins.open',opener(real_open)));stack.enter_context(patch('io.open',opener(real_io_open)));stack.enter_context(patch('sqlite3.connect',connection))
  for name in ('mkdir','remove','unlink','rmdir','rename','replace'):
   stack.enter_context(patch('os.'+name,mutation(getattr(os,name),name in ('rename','replace'))))
  yield
