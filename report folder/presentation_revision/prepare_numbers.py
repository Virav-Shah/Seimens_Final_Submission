from pathlib import Path
import json,csv,shutil,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/presentation_revision';DATA=OUT/'data'
r=json.loads((ROOT/'repo/results_mechanism_comparison/mechanism_results.json').read_text())
choices=json.loads((DATA/'three_complete_designs.json').read_text())
lines=['# Numerical examples for the replacement slides','','All percentages below are calculated at 24 hours. Reference-input mechanism comparisons and three-input candidate selection are kept separate.','','## One process changed at a time','','| Condition | Tumour, same rates (%) | Tumour, re-optimised (%) | Overall healthy, same rates (%) | Overall healthy, re-optimised (%) | Maximum healthy subgroup, same rates (%) | Maximum healthy subgroup, re-optimised (%) |','|---|---:|---:|---:|---:|---:|---:|']
for x in r:
 a,b=x['fixed'],x['reoptimised'];v=[a['tumour'],b['tumour'],a['healthy'],b['healthy'],a['subgroup'],b['subgroup']]
 lines.append('| '+x['label']+' | '+' | '.join(f'{z:.3f}' for z in v)+' |')
lines += ['','## Internalisation and selected binding properties','','| Dually bound internalisation (per hour) | T1 association (M⁻¹ s⁻¹) | T1 dissociation (s⁻¹) | T2 association (M⁻¹ s⁻¹) | T2 dissociation (s⁻¹) | T1 Kᴅ (µM) | T2 Kᴅ (µM) |','|---|---:|---:|---:|---:|---:|---:|']
for case,rate in [('dual_slow',.2),('reference',2),('dual_fast',4)]:
 x=next(x for x in r if x['case']==case)['reoptimised'];v=10.**np.array(x['theta']);kd=[1e6*v[1]/v[0],1e6*v[3]/v[2]]
 lines.append('| '+str(rate)+' | '+' | '.join(f'{z:.6g}' for z in [*v,*kd])+' |')
lines += ['','These selected designs all meet the 10% subgroup requirement. Target-2 association lies at the assumed lower rate limit, so do not present these rates as uniquely determined or experimentally established.','','## Three complete designs assessed under all three input conditions','','| Design | Lowest tumour kill (%) | Highest overall healthy kill (%) | Highest subgroup kill (%) |','|---|---:|---:|---:|']
for x in choices:lines.append(f"| {x['label']} | {x['tumour']:.3f} | {x['healthy']:.3f} | {x['subgroup']:.3f} |")
lines += ['','| Design | T1 association (M⁻¹ s⁻¹) | T1 dissociation (s⁻¹) | T2 association (M⁻¹ s⁻¹) | T2 dissociation (s⁻¹) | T1 Kᴅ (µM) | T2 Kᴅ (µM) |','|---|---:|---:|---:|---:|---:|---:|']
for x in choices:lines.append('| '+x['label']+' | '+' | '.join(f'{z:.6g}' for z in x['rates']+x['KD_uM'])+' |')
lines += ['','These examples select the highest-tumour candidate among the stored representative designs satisfying subgroup requirements of 1%, 5% and 10%, respectively, with overall healthy killing at most 0.95%. They are examples of selection from saved calculations, not three newly solved global optima. The extrema in a row can occur in different biological conditions.','','## Search and interpretation limits','','The mechanism comparison used two searches per condition/formulation. Twelve of the 30 local refinements reached their iteration limits; their feasible candidates were retained, and the maximum difference between paired mechanism-search tumour endpoints was about 0.074 percentage points. Very small differences between re-optimised endpoints should not be presented as a reliable ranking of biological mechanisms. All values shown are best-found results within assumed binding-rate limits.','', 'The internalisation sweep adds 324 unique tightly integrated model evaluations. It changes dually bound internalisation and, for the two-dimensional grid, both dissociation rates by a shared factor. It does not add optimisation runs.']
(OUT/'Results_numbers.md').write_text('\n'.join(lines)+'\n')
with (DATA/'three_complete_designs.csv').open('w') as f:
 fields=['design','lowest_tumour_kill_percent','highest_overall_healthy_kill_percent','highest_subgroup_kill_percent','T1_association_M_inverse_s_inverse','T1_dissociation_s_inverse','T2_association_M_inverse_s_inverse','T2_dissociation_s_inverse','T1_KD_uM','T2_KD_uM'];w=csv.writer(f);w.writerow(fields)
 for x in choices:w.writerow([x['label'],x['tumour'],x['healthy'],x['subgroup'],*x['rates'],*x['KD_uM']])
for name in ['mechanism_results.json','policy_results.json','comparisons_and_binding_rates.csv','search_runs.csv','study_configuration.json']:
 shutil.copy2(ROOT/'repo/results_mechanism_comparison'/name,DATA/name)
provenance={'new_sweep_evaluations':324,'new_optimisation_runs':0,'all_outputs_are_simulated':True,'paper_title':'Model-optimized bispecific antibodies improve the selectivity of antibody–drug conjugates for tumors','paper_doi':'10.20935/AcadDrug8407','source_presentation_sha256':hashlib.sha256((ROOT/'references/final_presentation.pdf').read_bytes()).hexdigest(),'source_model_sha256':hashlib.sha256((ROOT/'repo/mechanism_comparison/model.py').read_bytes()).hexdigest(),'source_files':['repo/results_mechanism_comparison/mechanism_results.json','repo/results_mechanism_comparison/policy_results.json','repo/results_boundary_review/original_representative_front.json'],'maximum_new_sweep_mass_error':max(x['mass'] for x in json.loads((DATA/'internalisation_sweep.json').read_text()))}
(DATA/'provenance.json').write_text(json.dumps(provenance,indent=2))
print('Numbers and provenance complete')
