"""Frozen opt-in micro qualification. Not a lower shared score or risk rule."""
from dataclasses import dataclass
from datetime import timedelta
from daytrading.data.completed_bars import Rejection
from daytrading.strategy.scalping.micro_pullback import evaluate_micro_pullback

@dataclass(frozen=True)
class MicroQualification:
    symbol:str
    observed_at:object
    kind:str
    trigger_start:object
    trigger_end:object
    deadline:object
    close_location:float
    trigger_volume:float
    prior_five_volume:float
    volume_ratio:float
    anchor:float
    technical_stop:float
    def valid_for(self,symbol,now):
        return (self.symbol==symbol and self.observed_at==now and self.kind in ('completed_10s','trade_quote_event')
            and self.trigger_start<self.trigger_end<=now<self.deadline
            and self.close_location>=.65-1e-12 and self.prior_five_volume>0 and self.volume_ratio>=.8-1e-12
            and 0<self.technical_stop<self.anchor)
    def metadata(self):
        return dict(profile='unified-micro-profile-v1',kind=self.kind,observed_at=self.observed_at.isoformat(),
            trigger_start=self.trigger_start.isoformat(),trigger_end=self.trigger_end.isoformat(),deadline=self.deadline.isoformat(),
            close_location=self.close_location,trigger_volume=self.trigger_volume,prior_five_volume=self.prior_five_volume,
            volume_ratio=self.volume_ratio,anchor=self.anchor,technical_stop=self.technical_stop,
            shared_score='diagnostic only for this experimental typed micro qualification; other hard guards unchanged')

def qualify(hit,evidence,now):
    if evidence.symbol!=hit.symbol or evidence.observed_at!=now:
        return Rejection('micro_profile','evidence identity/time mismatch')
    event=hit.criteria.get('_event_evidence')
    if event is not None:
        from daytrading.strategy.scalping.event_reclaim import candidate_from_event
        candidate=candidate_from_event(event,now);bars=[w.bar for w in event.windows]
        kind='trade_quote_event';end=event.observed_at;deadline=event.deadline
    else:
        w=hit.confirmation_window
        if w is None:return Rejection('micro_profile','original confirmation missing')
        candidate=evaluate_micro_pullback(hit.symbol,w.bars,now=now,minute_bars=evidence.session_minutes)
        bars=[x.bar for x in w.bars[-6:]];kind='completed_10s';end=w.bars[-1].closed_at;deadline=end+timedelta(seconds=15)
    if isinstance(candidate,Rejection):return Rejection('micro_profile',candidate.code+': '+candidate.reason)
    if len(bars)!=6:return Rejection('micro_profile','six original micro windows required')
    z=bars[-1];prior=sum(x.volume for x in bars[:-1])/5
    from daytrading.strategy.entry_checks import record,diagnostics
    record(diagnostics(hit),'micro_trigger_volume',now=now,measured=z.volume,threshold=prior*.8,
        units='shares',comparison='>=',passed=prior>0 and z.volume>=prior*.8-1e-12,
        window={'source':kind,'trigger_start':z.ts.isoformat(),'trigger_end':end.isoformat(),
                'reference':'mean of five preceding original 10s windows','reference_shares':prior},
        quote=evidence.quote_cost.last_quote)
    loc=(z.close-z.low)/(z.high-z.low) if z.high>z.low else 0
    record(diagnostics(hit),'micro_close_location',now=now,measured=loc,threshold=.65,
        units='fraction of trigger range',comparison='>=',passed=z.close>z.open and loc>=.65-1e-12,
        window={'source':kind,'start':z.ts.isoformat(),'end':end.isoformat()},quote=evidence.quote_cost.last_quote)
    if z.close<=z.open or loc<.65-1e-12:return Rejection('micro_profile','green trigger in top 35% required')
    if prior<=0 or z.volume<prior*.8-1e-12:
        return Rejection('micro_profile','trigger volume {:.4f}x < existing 0.8x confirmation requirement'.format(z.volume/prior if prior else 0))
    record(diagnostics(hit),'micro_original_reclaim',now=now,measured=evidence.last_trade.price,
        threshold=candidate.base_high,units='USD/share',comparison='>=',passed=evidence.last_trade.price>=candidate.base_high,
        window={'source':'fresh trade vs original trigger reclaim','trade_at':evidence.last_trade.ts.isoformat()},quote=evidence.quote_cost.last_quote)
    if evidence.last_trade.price<candidate.base_high:return Rejection('micro_profile','fresh trade lost original reclaim level')
    result=MicroQualification(hit.symbol,now,kind,z.ts,end,deadline,loc,z.volume,prior,z.volume/prior,candidate.anchor,candidate.technical_stop)
    return result if result.valid_for(hit.symbol,now) else Rejection('micro_profile','invalid or expired typed qualification')
