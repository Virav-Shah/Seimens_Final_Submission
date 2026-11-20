"""Cross-solver and limiting-case validation of generated prototype candidates."""
import json
from pathlib import Path
from model import Parameters,simulate
ROOT=Path(__file__).resolve().parent.parent
r=json.loads((ROOT/'output/prototype/results.json').read_text());checks=[]
for name,v in r.items():
    p=Parameters(**v['selected']['parameters'])
    a=simulate(p,rtol=1e-9,atol=1e-12)
    b=simulate(p,rtol=1e-9,atol=1e-12,method='Radau')
    delta=max(abs(a['tumor'][-1]-b['tumor'][-1]),abs(a['healthy'][-1]-b['healthy'][-1]))
    mass=max(abs(a['mass_error']))/p.dose
    assert delta<1e-6,(name,delta)
    assert mass<1e-6,(name,mass)
    assert b['healthy'][-1]<.0095,(name,b['healthy'][-1])
    checks.append(dict(scenario=name,max_endpoint_difference_fraction=float(delta),relative_mass_error=float(mass),healthy_kill_pct=float(b['healthy'][-1]*100),passed=True))
(ROOT/'output/prototype/validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
