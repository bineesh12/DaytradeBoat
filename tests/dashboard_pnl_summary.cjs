const {readFileSync}=require('node:fs');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const path=require('node:path');
const source=readFileSync(path.join(__dirname,'../src/daytrading/dashboard/assets/dashboard.js'),'utf8');
const sandbox={state:{recent_trades:[]},Intl,Date,Set,Number,escapeHtml:s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;'),fmtPnl:n=>(n>=0?'+$':'-$')+Math.abs(n).toFixed(2),shortEtTime:s=>s};
vm.createContext(sandbox);
vm.runInContext(source.slice(source.indexOf('function backtestTimestamp('),source.indexOf('function saveBacktestPreferences('))+source.slice(source.indexOf('function backtestPaperHistory('),source.indexOf('function renderRecordedValidation(')),sandbox);
const a={symbol:'NCPL',trade_type:'exit',entry_time:'2026-10-16T12:20:07Z',exit_time:'2026-10-16T12:21:45Z',entry_price:2.48,exit_price:2.52,quantity:384,pnl:15.36};
const b={...a,entry_time:'2026-10-16T12:47:07Z',exit_time:'2026-10-16T12:51:25Z',entry_price:2.5847,exit_price:2.59,quantity:500,pnl:2.65};
sandbox.state.recent_trades=[a,b,{...a,symbol:'BENF',pnl:900},{...a,exit_time:'2026-10-15T12:21:45Z',pnl:-25}];
let historical={symbol:'NCPL',date:'2026-10-16',mode:'historical_estimate',bars:509,round_trips:[],scorecard:{total_pnl:0,trades_taken:0}};
let s=sandbox.backtestSummary(historical);
assert.equal(s.paper.rows.length,2);assert.equal(s.paper.pnl.toFixed(2),'18.01');assert.equal(s.pnl,null);assert.equal(s.status,'Not reproduced');
let html=sandbox.backtestResultOverview(historical);
assert.ok(html.includes('+$18.01'));assert.ok(!html.includes('+$0.00'));assert.ok(html.includes('No closed simulated trades'));
// A genuine closed breakeven is numeric zero, never treated as absent.
s=sandbox.backtestSummary({...historical,round_trips:[{pnl:0}]});assert.equal(s.pnl,0);assert.equal(s.closed.length,1);
assert.ok(sandbox.backtestResultOverview({...historical,round_trips:[{pnl:0}]}).includes('+$0.00'));
const recorded={...historical,mode:'paper_validation',manifest:{coverage_start:'2026-10-16T22:31:18Z',coverage_end:'2026-10-16T23:11:29Z'},execution_comparison:{paper:{matched_trades:[]}},predictions:{paper:{realized_pnl:0}},decision_parity:'matched'};
s=sandbox.backtestSummary(recorded);assert.equal(s.status,'Missing recording');assert.equal(s.pnl,null);assert.equal(s.outside,true);
assert.ok(!sandbox.backtestResultOverview(recorded).includes('+$0.00'));
// Another day/stock cannot inherit NCPL's winners.
s=sandbox.backtestSummary({...historical,symbol:'SPY'});assert.equal(s.paper.pnl,null);
s=sandbox.backtestSummary({...historical,date:'2026-10-17'});assert.equal(s.paper.pnl,null);
// New York session boundaries, not the browser's local day.
assert.equal(sandbox.backtestDateKey('2026-10-17T00:30:00Z'),'2026-10-16');assert.equal(sandbox.backtestDateKey('bad'),'');
// Missing P&L never turns into zero or NaN.
s=sandbox.backtestSummary({...recorded,execution_comparison:{paper:{matched_trades:[{symbol:'NCPL',pnl:null}]}}});assert.equal(s.pnl,null);
assert.equal(sandbox.backtestPrice(null),'—');assert.equal(sandbox.backtestPrice(''),'—');
assert.ok(!sandbox.backtestResultOverview({...recorded,first_divergence:{reason:'<script>bad</script>'},manifest:{coverage_start:'2026-10-16T08:00:00Z',coverage_end:'2026-10-17T00:00:00Z'}}).includes('<script>'));
sandbox.state.recent_trades=Array.from({length:50},()=>a);assert.equal(sandbox.backtestPaperHistory('NCPL','2026-10-16').limited,true);
console.log('Profit semantics passed: missing versus zero, symbol/day isolation, capture coverage, history limits and escaping.');

// Real SDK history separates entries from exits and has microsecond timestamps.
sandbox.state.recent_trades=[{...a,trade_type:'entry',exit_time:null,pnl:null,entry_time:'2026-10-16 12:20:07.417496+00:00'},{...a,entry_time:'',exit_time:'2026-10-16 12:21:45.509554+00:00'}];
s=sandbox.backtestSummary(recorded);assert.equal(s.paper.rows[0].entry_time,'2026-10-16T12:20:07.417Z');assert.equal(s.status,'Missing recording');assert.equal(s.paper.pnl,15.36);
assert.equal(sandbox.backtestDateKey('2026-10-17 00:30:00.123456+00:00'),'2026-10-16');assert.equal(sandbox.backtestDateKey('2026-10-17 00:30:00'),'');

vm.runInContext(source.slice(source.indexOf('function backtestChartRange('),source.indexOf('function setBacktestChartRange(')),sandbox);
sandbox.state.backtest={chart:{start:null,end:null}};let range=sandbox.backtestChartRange(899);assert.equal(range.start,0);assert.equal(range.end,898);
sandbox.state.backtest={chart:{start:10,end:40}};range=sandbox.backtestChartRange(899);assert.equal(range.start,10);assert.equal(range.end,40);

// A data failure and a rule rejection are different outcomes.
sandbox.state.recent_trades=[];
s=sandbox.backtestSummary({...historical,bars:0,funnel:{scan_hits:0,signals:0,entries:0}});assert.equal(s.status,'Price data unavailable');assert.equal(s.pnl,null);
s=sandbox.backtestSummary({...historical,symbol:'MEDS',funnel:{scan_hits:38,signals:0,entries:0}});assert.equal(s.status,'No setups passed');assert.ok(s.message.includes('38 pattern hits'));
s=sandbox.backtestSummary({...historical,symbol:'MEDS',funnel:{scan_hits:38,signals:3,entries:0}});assert.equal(s.status,'No entries filled');
html=sandbox.renderStrategyBreakdown({strategy_breakdown:[{pattern:'vwap_pullback',hits:29,blocked_by_stage:{verifier:29},top_reasons:[{reason:'<script>bad</script>'}]}],micro_diagnostics:{status:'detector_only',candidates:0,first_block_counts:{sparse_bar:1608,reclaim:18,history:5},missing_execution_inputs:['quotes'],reason:'Detector only'}});
assert.ok(html.includes('Bar completion took over 15 seconds: 1608 windows'));
assert.ok(html.includes('not included as verified signals, orders or profit'));
assert.ok(!html.includes('<script>bad</script>'));
