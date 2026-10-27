"""Offline SDK port underneath the real AlpacaBroker lifecycle.

No credentials, requests library, broker outcomes or live TradingClient are
constructed. Only the exchange matcher is simulated; request pricing, polling,
cancel reconciliation, partial callbacks and protection use native code.
"""
from concurrent.futures import Future
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
from daytrading.execution.alpaca_broker import AlpacaBroker
from daytrading.replay.predict import SimulatedBroker

class InlinePositionPool:
    """Virtual I/O yields on the same deterministic actor, no real worker thread."""
    def submit(self,fn,*args,**kwargs):
        result=Future()
        try:result.set_result(fn(*args,**kwargs))
        except Exception as exc:result.set_exception(exc)
        return result
    def shutdown(self,*args,**kwargs):pass

class OfflineTradingSDK:
    def __init__(self,venue):self.venue=venue;self.counter=0
    def _advance(self,method,one_way=True):
        self.venue._advance_venue(self.venue.clock.now+timedelta(seconds=self.venue.view.latency(method,one_way)))
    def _snapshot(self,row):
        values=deepcopy(row)
        exchange_submitted=values.pop('_exchange_submitted_at',None)
        if exchange_submitted is not None:values['submitted_at']=exchange_submitted
        for key in ('stop_price','limit_price','filled_at','canceled_at','replaced_by'):values.setdefault(key,None)
        values['client_order_id']=values.get('client_order_id',values['id']);values['order_type']=values['type']
        return SimpleNamespace(**values)
    def _id(self):self.counter+=1;return 'predict-native-'+str(self.counter)
    def submit_order(self,*,order_data):
        request=order_data;side=getattr(request.side,'value',request.side);kind=getattr(request.type,'value',request.type)
        if kind not in ('limit','stop'):raise ValueError('Unsupported predictive SDK order type: '+str(kind))
        oid=self._id();venue=self.venue
        from daytrading.execution.origin import EXECUTION_ORIGIN
        origin=(EXECUTION_ORIGIN.get() if side=='buy' and not venue.positions.get(request.symbol)
                else venue.order_origins.get(request.symbol) or EXECUTION_ORIGIN.get())
        venue.events.append(dict(kind='protection_request' if kind=='stop' else 'order_request',id=oid,time=venue.clock.now.isoformat(),symbol=request.symbol,side=side,type=kind,qty=float(request.qty),limit=getattr(request,'limit_price',None),stop=getattr(request,'stop_price',None),opportunity_id=origin,operation_id=venue.view.operation_id))
        self._advance('submit_order')
        row=dict(id=oid,client_order_id=request.client_order_id,symbol=request.symbol,side=side,type=kind,qty=float(request.qty),limit_price=getattr(request,'limit_price',None),stop_price=getattr(request,'stop_price',None),filled_qty=0,filled_avg_price=None,status='new',submitted_at=venue.clock.now,_exchange_submitted_at=venue._venue_now(),filled_at=None,opportunity_id=origin,operation_id=venue.view.operation_id)
        venue.orders[oid]=row;venue._trigger_stops();venue._match_limits()
        changed=getattr(venue,'changed',None)
        if changed is not None:changed.set()
        observed=self._snapshot(row);server_at=venue.clock.now
        self._advance('submit_order')
        venue.events.append(dict(kind='protection' if kind=='stop' else 'order_ack',id=oid,time=venue.clock.now.isoformat(),server_at=server_at.isoformat(),status=observed.status,filled_qty=observed.filled_qty,symbol=request.symbol,qty=float(request.qty),stop=getattr(request,'stop_price',None),opportunity_id=origin,operation_id=venue.view.operation_id))
        return observed
    def get_order_by_id(self,order_id):return self._snapshot(self.venue._observe_order(str(order_id)))
    def get_order_by_client_id(self,client_order_id):
        row=next(o for o in self.venue.orders.values() if o['client_order_id']==client_order_id)
        return self.get_order_by_id(row['id'])
    def get_orders(self,filter=None,**kwargs):
        self._advance('get_orders');rows=self.venue.get_orders(filter=filter)
        symbols=getattr(filter,'symbols',None)
        result=[self._snapshot(vars(row)) for row in rows if not symbols or row.symbol in symbols]
        self._advance('get_orders');return result
    def cancel_order_by_id(self,order_id):return self.venue.cancel_order_by_id(str(order_id))
    def replace_order_by_id(self,order_id,*,order_data):
        self._advance('replace_order_by_id');old=self.venue.orders[str(order_id)]
        if old['status'] not in ('new','partially_filled'):raise ValueError('Cannot replace settled order')
        old['status']='canceled';oid=self._id();row=deepcopy(old)
        row.update(id=oid,qty=float(order_data.qty),stop_price=float(order_data.stop_price),status='new',submitted_at=self.venue.clock.now,_exchange_submitted_at=self.venue._venue_now(),filled_qty=0,filled_avg_price=None)
        old['replaced_by']=oid;self.venue.orders[oid]=row;result=self._snapshot(row)
        self._advance('replace_order_by_id');return result
    def get_all_positions(self):
        self._advance('get_all_positions');result=deepcopy(self.venue.get_all_positions());self._advance('get_all_positions');return result
    def get_account(self):
        self._advance('get_account');account=deepcopy(self.venue.get_account());result=SimpleNamespace(**account,portfolio_value=account['equity'],pattern_day_trader=False)
        self._advance('get_account');return result
    def get_clock(self):
        self._advance('get_clock');result=deepcopy(self.venue.get_clock());self._advance('get_clock');return result
    def cancel_orders(self):
        ids=[oid for oid,row in self.venue.orders.items() if row['status'] in ('new','partially_filled')]
        for oid in ids:self.cancel_order_by_id(oid)
        return [SimpleNamespace(id=oid) for oid in ids]
    def close_all_positions(self,*,cancel_orders=True):
        if cancel_orders:self.cancel_orders()
        return self.venue.close_all_positions()

class NativePredictiveBroker(AlpacaBroker):
    def __init__(self,view,mode,seed,initial_cash,matching_profile=None):
        # Deliberately do not call AlpacaBroker.__init__: that constructs a real
        # TradingClient. Native defaults are read from its constructor signature.
        import inspect
        signature=inspect.signature(AlpacaBroker.__init__).parameters
        self.venue=SimulatedBroker(view,mode,seed,initial_cash,matching_profile=matching_profile);self.view=view;self.clock=view.clock
        self._client=OfflineTradingSDK(self.venue);self._paper=True
        self._max_wait=signature['max_wait_seconds'].default;self._poll_interval=signature['poll_interval'].default
        self._limit_buffer_pct=signature['limit_buffer_pct'].default;self._cancel_grace_seconds=2.
        self._slippage_guard=None;self._position_pool=InlinePositionPool();self._position_future=None
        self._now=lambda:self.clock.now;self._safety_state()
    @property
    def events(self):return self.venue.events
    @property
    def positions(self):return self.venue.positions
    @property
    def orders(self):return self.venue.orders
    @property
    def order_origins(self):return self.venue.order_origins
    @property
    def cash(self):return self.venue.cash
    @cash.setter
    def cash(self,value):self.venue.cash=value
    def _trigger_stops(self):return self.venue._trigger_stops()
    def _match_limits(self):return self.venue._match_limits()
