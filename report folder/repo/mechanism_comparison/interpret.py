from pathlib import Path
import sys
_SUBMISSION_ROOT = Path(__file__).resolve().parents[3]
__file__ = str(_SUBMISSION_ROOT / 'repo/mechanism_comparison/interpret.py')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_SUBMISSION_ROOT / "repo"))
from mechanism_comparison.study import *

def write_observations():
 rows=json.loads((OUT/'mechanism_results.json').read_text());pol=json.loads((OUT/'policy_results.json').read_text());ref=rows[0]['fixed'];lookup={r['case']:r for r in rows}
 lines=['# What these comparisons show','',f"The reference design gives **{ref['tumour']:.3f}% tumour killing**, **{ref['healthy']:.3f}% overall healthy-cell killing**, and **{ref['subgroup']:.3f}% maximum healthy-subgroup killing** at 24 hours for the nominal inputs. This is a different summary from V5’s lowest tumour killing across three input variants.",'','## Effects of each process','','| Condition | Transferred rates: tumour kill (%) | Re-optimised: tumour kill (%) | Gain from re-optimisation (percentage points) | Transferred design meets both healthy-cell limits? |','|---|---:|---:|---:|---|']
 for row in rows[1:]:
  a,b=row['fixed'],row['reoptimised'];ok=a['healthy']<=.95 and a['subgroup']<=10
  lines.append(f"| {row['label']} | {a['tumour']:.3f} | {b['tumour']:.3f} | {b['tumour']-a['tumour']:.3f} | {'Yes' if ok else 'No'} |")
 lines += ['','The gain column compares the same biological condition before and after changing the binding rates. It is not the benefit of adding a biological mechanism to the model. If the transferred design violates a healthy-cell limit, a fall in tumour killing after re-optimisation can reflect the cost of meeting that limit.','']
 a,b=lookup['dual_slow'],lookup['dual_fast']
 lines.append(f"**Internalisation:** increasing the dually bound rate from 0.2 to 4 per hour changes tumour killing from {a['fixed']['tumour']:.3f}% to {b['fixed']['tumour']:.3f}% at fixed binding rates, and from {a['reoptimised']['tumour']:.3f}% to {b['reoptimised']['tumour']:.3f}% after re-optimisation. The singly bound internalisation rates and every other biological input remain fixed. These outcomes apply to this assumed model and do not establish a universal advantage of faster internalisation.")
 active=sum(abs(r['reoptimised']['subgroup']-10)<.01 for r in rows)
 lines += ['',f"**Which limit determines the selected designs?** In {active} of {len(rows)} conditions, the selected design is within 0.01 percentage points of the 10% healthy-subgroup limit. Similar re-optimised tumour killing therefore need not mean the mechanisms are unimportant: changing a process can improve delivery to both tumour cells and target-expressing healthy cells, requiring the binding rates to be readjusted."]
 stable,short=lookup['linker_stable']['fixed'],lookup['linker_short']['fixed']
 lines += ['',f"**Premature release at fixed binding rates:** moving from no premature release to a 48-hour linker half-life changes tumour killing from {stable['tumour']:.3f}% to {short['tumour']:.3f}%, while overall healthy killing changes from {stable['healthy']:.3f}% to {short['healthy']:.3f}%. This comparison includes free-payload movement at the unchanged reference rate."]
 lines += ['','**Linker cleavage and free payload:** the premature-release panels vary loss of payload outside cells; the processing panels vary release after internalisation. The membrane-movement panels change bidirectional free-payload exchange. These are distinct model terms. The no-movement condition prevents free payload entering cells, so the consequences of premature release depend on whether movement is present. This one-process-at-a-time experiment does not quantify all interactions.','', '**Binding rates:** the rate plot and CSV show the actual changes in all four rates. Target-2 association remains approximately 100 M⁻¹ s⁻¹, the lower search limit, in every selected mechanism condition. Thus these results remain conditional on the assumed search domain. Nearly identical tumour killing does not establish that the selected rates are unique. Weakly determined rates and finite search effort can also produce differences between returned combinations.','', '## What changes when healthy subgroups are included?','']
 a,b,c=pol
 lines.append(f"With only the overall healthy-cell limit, the highest-tumour candidate in the pooled searches gives **{a['tumour']:.3f}% tumour killing**, **{a['healthy']:.3f}% overall healthy killing**, but **{a['subgroup']:.3f}% killing in the most affected healthy subgroup**. Adding the subgroup objective alone leaves this highest-tumour endpoint available. It exposes the trade-off but does not force a safer selection.")
 lines += ['',f"Adding the illustrative 10% subgroup limit changes the selected endpoint to **{c['tumour']:.3f}% tumour killing**, **{c['healthy']:.3f}% overall healthy killing**, and **{c['subgroup']:.3f}% maximum subgroup killing**. The tumour-kill change is **{a['tumour']-c['tumour']:.3f} percentage points**. The protection comes from the explicit constraint and selection rule; it must not be attributed merely to running a multi-objective algorithm.",'','The policy comparison pools tightly integrated nominal candidates before applying each policy. This allows the first two policies, which have identical feasible sets, to be compared without confusing differences in search luck with an effect of an extra objective. Each optimisation’s separate sampled results are retained.','', '## Limits of the conclusions','']
 runs=[json.loads(p.read_text()) for seed in SEEDS for p in OUT.glob(f'*_{seed}.json')]
 unsuccessful=[r for r in runs if not r['local_success']]
 failures=sum('error' in h for r in runs for h in r['history'])
 lines.append(f"There were {len(runs)} searches using two seeds per condition/formulation; {len(unsuccessful)} local refinements did not report successful termination. The search histories contain {failures} numerical evaluation errors. Only tightly integrated feasible candidates supply the orange bars. These are best-found endpoints under the stated rate limits, not proven global optima.")
 if unsuccessful:lines.append('Local termination details are retained in `search_runs.csv`: '+ '; '.join(f"{r['case']} ({r['policy']}), seed {r['seed']}: {r['message']}" for r in unsuccessful)+'.')
 spreads=[]
 for row in rows:
  rr=[r['best']['tumour'] for r in runs if r['case']==row['case'] and r['policy']=='subgroup_limit' and r['best']]
  spreads.append(max(rr)-min(rr))
 lines += ['',f"The largest difference between the two searches’ best tumour-kill values among the mechanism conditions is {max(spreads):.3f} percentage points. Small differences should be read alongside this finite-search variation; it is not a statistical confidence interval.",'','The biological values are synthetic, and two alternatives per process do not describe a complete sensitivity curve. The model has one well-mixed extracellular payload pool, expression-defined healthy groups rather than organs, and no spatial tissue model. It cannot demonstrate clinical safety or actual manufacturable affinities. A small effect in these comparisons is not a reason to remove a mechanism when measured inputs could place it in a different regime.']
 (OUT/'observations.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines))
if __name__=='__main__':write_observations()
