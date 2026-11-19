import json,sys,statistics as st,random
from collections import defaultdict
def load(d):
    rows=[]
    for l in open(d+'/latency/traces.jsonl'):
        t=json.loads(l)
        if t.get('warmup') or t.get('error') or not t.get('segments'): continue
        end=t['t0_ns']+t['segments'][0]['end_s']*1e9
        ev=defaultdict(list)
        for e in t['events']: ev[e['kind']].append(e)
        if not all(ev[k] for k in('endpoint','asr_final')): continue
        ep,af=ev['endpoint'][0],ev['asr_final'][0]
        rows.append(((ep['recv_ns']-end)/1e6,(af['recv_ns']-ep['recv_ns'])/1e6,(af['recv_ns']-end)/1e6))
    return rows
def p(v,q):
    v=sorted(v);return v[min(len(v)-1,int(round(q*(len(v)-1))))]
def ci(v):
    random.seed(1);m=sorted(st.median(random.choices(v,k=len(v))) for _ in range(2000));return m[50],m[1949]
base=sys.argv[1]
print("brazo | n | espera p50 | asr_fin p50 [IC] p95 | endpoint->asr_final... | fin habla->asr_final p50 p95")
for a in "off adaptsimul adaptsimulfin stream4adapt64 stream4adapt64fin off2".split():
    r=load(f"{base}/run/{a}")
    e=[x[0] for x in r];f=[x[1] for x in r];t=[x[2] for x in r]
    lo,hi=ci(f);lo2,hi2=ci(t)
    print(f"{a} | {len(r)} | {p(e,.5):.0f} | {p(f,.5):.0f} [{lo:.0f},{hi:.0f}] p95 {p(f,.95):.0f} | fin->asr_final {p(t,.5):.0f} [{lo2:.0f},{hi2:.0f}] p95 {p(t,.95):.0f}")
