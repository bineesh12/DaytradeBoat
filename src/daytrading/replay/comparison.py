"""Post-run matching. This module is never a predictive input provider."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from .codec import unpack


def capture_reference_start(events):
    if not events:return None
    first=datetime.fromisoformat(events[0]['received_at'])
    for event in events:
        if event['kind']!='clock_calibration':continue
        calibration=unpack(event['data']['value'])
        if calibration.get('synchronized') and calibration.get('offset_seconds') is not None:
            return first+timedelta(seconds=calibration['offset_seconds'])
    return first


def observed_intents(events):
    """Final cumulative SDK quantities override an earlier synchronous return.

    Lost acknowledgements and late fills remain outcomes for comparison only.
    Protective-stop / close-all fills are included without double counting.
    """
    requests=defaultdict(list);sdk_pending=defaultdict(list);clients={};ids={};latest={};seen_at={};rows=[]
    first=capture_reference_start(events)
    for event in events:
        d=event['data'];kind=event['kind'];op=event.get('operation_id')
        if kind=='external_request' and d['role']=='_broker' and d['method'] in ('submit','submit_urgent_exit'):
            order=unpack(d['args'])[0]
            row=dict(operation_id=op,opportunity_id=d.get('opportunity_id'),symbol=order.symbol,side=order.side.value,qty=int(order.quantity),limit=None,time=event['received_at'],filled_qty=0,price=None,cancelled=False)
            requests[op].append(row);rows.append(row)
        elif kind=='external_request' and d['role']=='broker_client' and d['method']=='submit_order':
            kwargs=unpack(d['kwargs']);args=unpack(d['args']);request=kwargs.get('order_data') or (args[0] if args else None)
            parent=requests[op][-1] if requests[op] else None
            if request is not None:
                reqtype=getattr(getattr(request,'type',None),'value',getattr(request,'type',None))
                if parent is not None and reqtype!='stop' and getattr(request,'symbol',None)==parent['symbol']:
                    parent['limit']=getattr(request,'limit_price',None)
                    if getattr(request,'client_order_id',None):clients[str(request.client_order_id)]=parent
                else:parent=None
                sdk_pending[op].append(parent)
        elif kind=='external_response' and d['role']=='broker_client' and 'value' in d:
            value=unpack(d['value']);values=value if isinstance(value,list) else [value]
            if d.get('method')=='submit_order' and sdk_pending[op]:
                parent=sdk_pending[op].pop()
                if parent is not None and getattr(value,'id',None):ids[str(value.id)]=parent
            for order in values:
                if not all(hasattr(order,k) for k in ('id','symbol','side','filled_qty')):continue
                oid=str(order.id);parent=ids.get(oid) or clients.get(str(getattr(order,'client_order_id','')))
                if parent is not None:ids[oid]=parent
                old=latest.get(oid);old_qty=float(getattr(old,'filled_qty',0) or 0);new_qty=float(order.filled_qty or 0)
                terminal=('filled','canceled','cancelled','expired','rejected')
                status=getattr(getattr(order,'status',None),'value',getattr(order,'status',None));old_status=getattr(getattr(old,'status',None),'value',getattr(old,'status',None))
                if new_qty>old_qty or old is None or (new_qty==old_qty and (old_status not in terminal or status in terminal) and (new_qty==0 or getattr(order,'filled_avg_price',None) is not None)):
                    latest[oid]=order;seen_at[oid]=event['received_at']
        elif kind=='external_response' and d['role']=='_broker' and d['method'] in ('submit','submit_urgent_exit') and requests[op]:
            row=requests[op].pop()
            if 'value' in d:
                fill,status=unpack(d['value']);row['cancelled']=getattr(status,'value',status) in ('cancelled','canceled')
                if fill is not None:row.update(filled_qty=fill.quantity,price=fill.price,fill_time=fill.ts.isoformat())
    for oid,order in latest.items():
        parent=ids.get(oid);quantity=float(order.filled_qty or 0);status=getattr(getattr(order,'status',None),'value',getattr(order,'status',None))
        if parent is not None:
            parent['cancelled']=parent['cancelled'] or status in ('canceled','cancelled','expired')
            if quantity>0 and quantity>=parent['filled_qty']:
                parent.update(filled_qty=quantity,price=float(order.filled_avg_price),fill_time=getattr(order,'filled_at',None).isoformat() if getattr(order,'filled_at',None) else seen_at[oid],broker_id=oid)
    buys=sorted((r for r in rows if r['side']=='buy' and r['filled_qty']>0),key=lambda r:r['time'])
    for oid,order in latest.items():
        if oid in ids:continue
        kind=getattr(getattr(order,'type',None),'value',getattr(order,'type',None));side=getattr(order.side,'value',order.side);submitted=getattr(order,'submitted_at',None)
        if kind not in ('stop','market') or side!='sell' or not submitted or submitted<first or float(order.filled_qty or 0)<=0:continue
        origins=[r for r in buys if r['symbol']==order.symbol and r['time']<=submitted.isoformat()]
        rows.append(dict(operation_id=None,opportunity_id=origins[-1].get('opportunity_id') if origins else None,id=oid,symbol=order.symbol,side='sell',qty=int(order.qty),limit=None,time=submitted.isoformat(),filled_qty=float(order.filled_qty),price=float(order.filled_avg_price),fill_time=getattr(order,'filled_at',None).isoformat() if getattr(order,'filled_at',None) else seen_at[oid],cancelled=False,source='broker '+str(kind)))
    return sorted(rows,key=lambda r:r['time'])


def compare_execution(events,prediction,*,risk_budget):
    paper=observed_intents(events);simulated=[dict(x) for x in prediction.get('events',[]) if x['kind']=='order_request'];fills=defaultdict(list)
    ids={r['id'] for r in simulated}
    for event in prediction.get('events',[]):
        if event['kind']=='fill' and event['id'] not in ids:
            ids.add(event['id']);simulated.append(dict(kind='order_request',id=event['id'],opportunity_id=event.get('opportunity_id'),symbol=event['symbol'],side=event['side'],qty=event['qty'],limit=None,time=event['time']))
    for event in prediction.get('events',[]):
        if event['kind']=='fill':fills[event['id']].append(event)
    for row in simulated:
        executed=fills[row['id']];quantity=sum(x['qty'] for x in executed);row.update(filled_qty=quantity,price=sum(x['qty']*x['price'] for x in executed)/quantity if quantity else None)
    keyed=defaultdict(list)
    for row in simulated:keyed[(row.get('opportunity_id'),row['symbol'],row['side'])].append(row)
    comparisons=[];unmatched=[];paired=[]
    for actual in paper:
        candidates=keyed[(actual.get('opportunity_id'),actual['symbol'],actual['side'])];estimated=candidates.pop(0) if candidates else None
        if estimated is None:unmatched.append(dict(paper=actual,replay=None));continue
        intent_match=actual['qty']==estimated['qty'] and (actual['limit'] is None or (estimated['limit'] is not None and abs(actual['limit']-estimated['limit'])<1e-8))
        if not intent_match:unmatched.append(dict(paper=actual,replay=estimated,reason='request quantity or limit differs'))
        comparison=dict(symbol=actual['symbol'],side=actual['side'],paper_filled_qty=actual['filled_qty'],filled_qty=estimated['filled_qty'],cancelled=actual['cancelled'],paper_price=actual['price'],price=estimated['price'],intent_match=intent_match,paper=actual,replay=estimated)
        comparisons.append(comparison);paired.append((actual,estimated))
    for candidates in keyed.values():
        for row in candidates:unmatched.append(dict(paper=None,replay=row))
    # Aggregate scale-ins/partial exits by their originating setup. A closed
    # trade exists only when both compared position ledgers finish flat.
    ledgers={};trades=[]
    for actual,estimated in paired:
        key=(actual['symbol'],actual.get('opportunity_id'))
        ledger=ledgers.setdefault(key,dict(buy_qty=0.,sell_qty=0.,buy_cost=0.,sell_value=0.,p_buy_qty=0.,p_sell_qty=0.,p_buy_cost=0.,p_sell_value=0.))
        for row,prefix in ((actual,'p_'),(estimated,'')):
            qty=row['filled_qty'];price=row['price'] or 0.
            if row['side']=='buy':ledger[prefix+'buy_qty']+=qty;ledger[prefix+'buy_cost']+=qty*price
            else:ledger[prefix+'sell_qty']+=qty;ledger[prefix+'sell_value']+=qty*price
        l=ledger
        if l['buy_qty']>0 and l['p_buy_qty']>0 and l['buy_qty']==l['sell_qty'] and l['p_buy_qty']==l['p_sell_qty']:
            trades.append(dict(symbol=actual['symbol'],entry=l['buy_cost']/l['buy_qty'],exit=l['sell_value']/l['sell_qty'],paper_entry=l['p_buy_cost']/l['p_buy_qty'],paper_exit=l['p_sell_value']/l['p_sell_qty'],pnl=l['sell_value']-l['buy_cost'],paper_pnl=l['p_sell_value']-l['p_buy_cost'],risk_budget=risk_budget));ledgers.pop(key)
    return dict(matched_intents=comparisons,matched_trades=trades,unmatched_intents=unmatched)


def complete_session(manifest,events):
    from daytrading.market_calendar import is_us_trading_day
    if manifest['state']!='sealed' or not events:return False
    first=datetime.fromisoformat(events[0]['received_at']).astimezone(ZoneInfo('America/New_York'));last=datetime.fromisoformat(events[-1]['received_at']).astimezone(ZoneInfo('America/New_York'))
    return is_us_trading_day(first.date()) and first.date().isoformat()==manifest['session_date'] and first<=first.replace(hour=4,minute=0,second=5,microsecond=0) and last>=first.replace(hour=16,minute=0,second=0,microsecond=0)


def protection_evidence(events,*,maximum_unprotected_seconds=2.):
    """Reconcile cumulative SDK observations once; do not count aggregate fills twice.

    This reports broker-held coverage, not a guarantee that a stop executes at
    its trigger. Premarket software protection remains a separate native audit.
    """
    cumulative={};held=defaultdict(float);stops={};since={};intervals=[];failures=[]
    capture_start=capture_reference_start(events)
    for event in events:
        data=event['data'];now=datetime.fromisoformat(event['received_at'])
        if event['kind']=='external_response' and data['role']=='broker_client' and 'value' in data:
            value=unpack(data['value']);values=value if isinstance(value,list) else [value]
            for order in values:
                if not all(hasattr(order,k) for k in ('id','symbol','status','filled_qty')):continue
                submitted=getattr(order,'submitted_at',None)
                if submitted is not None and capture_start is not None and submitted<capture_start:continue
                oid=str(order.id);symbol=order.symbol;side=getattr(order.side,'value',order.side);status=getattr(order.status,'value',order.status)
                qty=float(order.filled_qty or 0);delta=qty-cumulative.get(oid,0.);cumulative[oid]=max(qty,cumulative.get(oid,0.))
                if delta>0:held[symbol]+=delta if side=='buy' else -delta
                if getattr(order,'stop_price',None) is not None:
                    replaces=getattr(order,'replaces',None)
                    if replaces:stops.pop(str(replaces),None)
                    if status not in ('filled','canceled','cancelled','expired','rejected','replaced'):stops[oid]=(symbol,max(0.,float(order.qty)-qty))
                    else:stops.pop(oid,None)
        coverage=defaultdict(float)
        for symbol,qty in stops.values():coverage[symbol]+=qty
        for symbol,qty in held.items():
            if qty<-.00001:failures.append(dict(symbol=symbol,time=now.isoformat(),reason='observed unintended short exposure'))
            uncovered=max(0.,qty-coverage[symbol])
            if uncovered>1e-7:since.setdefault(symbol,now)
            elif symbol in since:
                start=since.pop(symbol);seconds=(now-start).total_seconds();row=dict(symbol=symbol,start=start.isoformat(),end=now.isoformat(),seconds=seconds)
                intervals.append(row)
                if seconds>maximum_unprotected_seconds:failures.append(dict(**row,reason='broker-held coverage delayed beyond 2 seconds'))
            if coverage[symbol]>max(0.,qty)+1e-7:failures.append(dict(symbol=symbol,time=now.isoformat(),reason='observed protective stop quantity exceeds holdings'))
    for symbol,start in since.items():failures.append(dict(symbol=symbol,start=start.isoformat(),reason='broker-held protection remains unconfirmed at capture end'))
    return dict(status='failed' if failures else 'observed' if cumulative else 'unverified',failures=failures,unprotected_intervals=intervals)


def decision_comparison_rows(events,prediction,symbol,audit_matched):
    """Report filtering only: both native lanes already ran the entire universe."""
    rows=[];simulated=defaultdict(list);protections=[e for e in prediction.get('events',[]) if e['kind']=='protection' and e['symbol']==symbol]
    for event in prediction.get('decisions',[]):
        payload=unpack(event['payload'])
        if payload.get('symbol')==symbol:simulated[event['type']].append((event,payload))
    stages={'scan_hit':'setup','signal':'confirmation','entry_decision':'rejection / entry gate','trade_fill':'entry','trade_exit':'exit / P&L','execution_timer_bar':'10-second confirmation','hot_watch':'watchlist'}
    for event in events:
        d=event['data']
        if event['kind']=='decision':
            payload=unpack(d['payload'])
            if payload.get('symbol')!=symbol:continue
            candidates=simulated[d['type']];estimated,other=candidates.pop(0) if candidates else (None,None)
            rows.append(dict(time=event['received_at'],stage=stages.get(d['type'],d['type']),paper=payload,replay=other,match=payload==other,audit='exact' if audit_matched else 'see first divergence',reason='different or missing predictive decision' if payload!=other else None))
        elif event['kind']=='external_request' and d['role']=='_broker':
            args=unpack(d['args']);kwargs=unpack(d['kwargs']);method=d['method']
            if method in ('place_protective_stop','replace_protective_stop') and args and args[0]==symbol:
                other=protections.pop(0) if protections else None
                rows.append(dict(time=event['received_at'],stage='protection',paper=dict(method=method,args=args,kwargs=kwargs),replay=other,match=bool(other and args[-2]==other['qty'] and round(args[-1],2)==other['stop']),audit='exact' if audit_matched else 'unverified'))
            elif method in ('submit','submit_urgent_exit') and args and getattr(args[0],'symbol',None)==symbol:
                order=args[0];rows.append(dict(time=event['received_at'],stage='order request',paper=dict(side=order.side.value,qty=order.quantity,limit=order.limit_price),replay='See independent intent comparison',match=None,audit='exact' if audit_matched else 'unverified'))
        elif event['kind']=='operation_start' and d['method'] in ('_scan_discovery','_process_candidate_hydration_batch','_process_hod_seed_batch'):
            args=unpack(d['args'])
            if symbol in repr(args):rows.append(dict(time=event['received_at'],stage='discovery / hydration',paper=dict(method=d['method'],inputs=args),replay='Native full-universe discovery; see coverage status',match=None,audit='exact' if audit_matched else 'unverified'))
    return sorted(rows,key=lambda row:row['time'])
