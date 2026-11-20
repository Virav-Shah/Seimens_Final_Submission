from pathlib import Path
import json,nbformat,numpy as np,pandas as pd

def render_conclusions():
    s=json.loads((OUT/'summary.json').read_text());runs=json.loads((OUT/'runs_summary.json').read_text())
    old=s['v5']['best'];a=s['original']['best'];b=s['expanded']['best']
    ra=10**np.array(a['theta']);rb=10**np.array(b['theta'])
    text=f'''## 6. What the results support

**The tumour-kill endpoint is reproducible within the original search domain, but the preferred binding rates remain dependent on the domain.**

| Search | Lowest tumour cell kill across inputs | Maximum overall healthy cell kill | Maximum healthy subgroup cell kill |
|---|---:|---:|---:|
| V5 | {old['tumour']:.4f}% | {old['healthy']:.4f}% | {old['subgroup']:.4f}% |
| Longer searches, original limits | {a['tumour']:.4f}% | {a['healthy']:.4f}% | {a['subgroup']:.4f}% |
| Longer searches, expanded limits | {b['tumour']:.4f}% | {b['healthy']:.4f}% | {b['subgroup']:.4f}% |

These extrema can occur in different biological input variants; a row does not describe one simulated individual.

Widening the limits changes the best-found tumour endpoint by **{b['tumour']-a['tumour']:.4f} percentage points**. The best expanded-domain candidate has a target-1 association rate of **{rb[0]:.3g} M⁻¹ s⁻¹** and a target-2 dissociation rate of **{rb[3]:.3g} s⁻¹**, reaching the new lower and upper limits respectively. Therefore simply enlarging the domain does not establish an interior optimum or justify selecting the new extremes as a better engineered antibody.

| Rate or affinity | Original-limit endpoint | Expanded-limit endpoint |
|---|---:|---:|
| Target 1 association (M⁻¹ s⁻¹) | {ra[0]:.5g} | {rb[0]:.5g} |
| Target 1 dissociation (s⁻¹) | {ra[1]:.5g} | {rb[1]:.5g} |
| Target 2 association (M⁻¹ s⁻¹) | {ra[2]:.5g} | {rb[2]:.5g} |
| Target 2 dissociation (s⁻¹) | {ra[3]:.5g} | {rb[3]:.5g} |
| Target 1 dissociation constant (µM) | {ra[1]/ra[0]*1e6:.5g} | {rb[1]/rb[0]*1e6:.5g} |
| Target 2 dissociation constant (µM) | {ra[3]/ra[2]*1e6:.5g} | {rb[3]/rb[2]*1e6:.5g} |

The association/dissociation rates are the optimisation variables; each dissociation constant is their ratio. The four entries in a column belong together.

**The original boundary percentage overstated how broadly the numerical solutions were sampled.** The 1,634 V5 entries reduce to {s['v5']['representative_boundary']['total']} outcome representatives at the declared reporting resolution; {s['v5']['representative_boundary']['count']} ({s['v5']['representative_boundary']['percent']:.1f}%) of these are near a boundary. This depends on the reporting resolution and is not a probability or a fraction of all possible antibodies. The original raw count, 1,540/1,634, remains correct for the stored entries.

'''
    for d,label in [('original','Original limits'),('expanded','Expanded limits')]:
        vals=[r['best_tumour'] for r in runs if r['domain']==d and not r['warm_start'] and r['best_tumour'] is not None]
        text+=f"**{label}: independent starts.** After local refinement, cold-start endpoints range from {min(vals):.6f}% to {max(vals):.6f}% tumour killing. Agreement supports this endpoint within the tested model; it does not establish global optimality or convergence of the full trade-off surface.\n\n"
    alternatives=[r['endpoint'] for r in runs if r['domain']=='expanded' and r['refinement'] and r['refinement']['success'] and r['endpoint']]
    if alternatives:
        alt=max(alternatives,key=lambda r:r['tumour']);ar=10**np.array(alt['theta'])
        text+=f"**Different rate combinations give almost the same tumour killing.** The successfully terminated expanded-domain searches find an alternative with target-2 association near {ar[2]:.3g} M⁻¹ s⁻¹ and target-1 association near {ar[0]:.4g} M⁻¹ s⁻¹, giving {alt['tumour']:.6f}% tumour killing. Its complete rates (target-1 on, target-1 off, target-2 on, target-2 off) are ({ar[0]:.6g}, {ar[1]:.6g}, {ar[2]:.6g}, {ar[3]:.6g}), in M⁻¹ s⁻¹ for on-rates and s⁻¹ for off-rates. Thus even the preferred association-rate asymmetry is not reliably selected by maximising this nearly flat endpoint. The small performance difference does not justify choosing one branch for wet-lab construction.\n\n"
    unfinished=[r for r in runs if r['refinement'] and not r['refinement']['success']]
    if unfinished:
        text+='**Local-search limits:** '+ '; '.join(f"{r['domain']}, seed {r['seed']}: {r['refinement']['message']}" for r in unfinished)+'. Feasible candidates from these runs are retained after tight integration, but these local searches did not report successful termination.\n\n'
    if s['experimental_reference']['best'] is None:
        text+='''**Experimental-reference limits:** none of the three searches found a combination meeting the unchanged healthy-cell constraints. This means the generic V5 setup and this transferred kinetic envelope did not produce a feasible candidate in the search. It does not establish that the continuous domain is infeasible, that these are universal antibody limits, or that the published antibodies lack selectivity. The experiment concerns different targets and biology.

'''
    text+='''**Implication for the method:** retain the complete trade-off combinations and declare the assumed rate limits. Report sensitivity to those limits alongside the selected rates. For generic targets, use the expanded domain as a diagnostic only. When actual target-specific kinetics become available, replace the domain with supported rate pairs or supported joint ranges before selecting candidates. Do not resolve the boundary issue by calling a newly imposed limit the optimal affinity.

A boundary optimum is not inherently wrong: it may be appropriate when the limit is an actual engineering constraint. These generic bounds have no such target-specific support. The original paper's optimised rates also fall outside the transferred experimental envelope, so that comparison cannot establish universal attainability limits.

The full sampled trade-off surface remains less firmly established than its highest-tumour endpoint. Checkpoint curves show whether objective-space coverage is still improving; longer searches and local endpoint agreement alone do not certify the whole surface. The biological assumptions, including the second-binding dissociation relation, remain unchanged.

Files: `binding_combinations.csv` contains tightly integrated representative combinations; `run_comparison.csv` retains independent-run results; `convergence.csv` retains intermediate search progress. The new work improves the assessment and reporting of affinity selection; it does not demonstrate a biologically superior ADC.
'''
    return text


interpretation=render_conclusions()
(OUT/'conclusions.md').write_text(interpretation)
display(Markdown(interpretation))