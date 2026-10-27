"""Fail-closed network/credential guard for disposable benchmark processes."""
import errno,os,socket,sys
from pathlib import Path
_installed=False
_DENIED={'socket.connect','socket.getaddrinfo','socket.sendto','socket.sendmsg','socket.gethostbyname','socket.gethostbyaddr','socket.getnameinfo','socket.bind'}

def install():
 global _installed
 if _installed:return
 for key in tuple(os.environ):
  upper=key.upper()
  if upper.startswith(('ALPACA_','APCA_','AWS_')) or upper in {'GOOGLE_APPLICATION_CREDENTIALS','GOOGLE_API_KEY'} or (upper.startswith('DAYTRADING_') and any(x in upper for x in ('API_KEY','SECRET','TOKEN','PASSWORD'))):os.environ.pop(key,None)
 def deny(event,args):
  if event in _DENIED:raise OSError(errno.EPERM,'All network forbidden in offline DaytradeBoat benchmark')
  if event=='socket.__new__' and len(args)>1 and args[1] in (socket.AF_INET,socket.AF_INET6):raise OSError(errno.EPERM,'Internet sockets forbidden in offline benchmark')
 sys.addaudithook(deny);_installed=True
 os.environ['DAYTRADEBOAT_OFFLINE_GUARD_ACTIVE']='1'

def assert_guard_active():
 if not _installed or os.environ.get('DAYTRADEBOAT_OFFLINE_GUARD_ACTIVE')!='1':raise RuntimeError('Offline audit guard must load before application imports')

def assert_network_namespace(host_netns,*,proc_root='/proc',interfaces=None,readlink=os.readlink):
 if sys.platform!='linux':raise RuntimeError('Remote workload requires Linux network namespace')
 if not host_netns or not host_netns.startswith('net:['):raise RuntimeError('Missing operator-recorded host namespace identity')
 own=readlink(str(Path(proc_root)/'self/ns/net'))
 if own==host_netns:raise RuntimeError('Workload is still in host network namespace')
 names=[name for _,name in (socket.if_nameindex() if interfaces is None else interfaces)]
 if set(names)!={'lo'}:raise RuntimeError('Private network namespace must expose loopback only')
 routes=(Path(proc_root)/'net/route').read_text().splitlines()[1:]
 if any(line.split() and line.split()[0]!='lo' for line in routes):raise RuntimeError('External IPv4 route present')
 v6=Path(proc_root)/'net/ipv6_route'
 if v6.exists() and any(line.split() and line.split()[-1]!='lo' for line in v6.read_text().splitlines()):raise RuntimeError('External IPv6 route present')
 return dict(host_namespace=host_netns,workload_namespace=own,interfaces=names,private_network_verified=True)
