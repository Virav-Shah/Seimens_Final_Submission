"""Charge model-selection labels to ML and give direct search the same total budget."""
from run_trial import *
def main():
 path=OUT/'results.json';results=json.loads(path.read_text())
 if results['configuration'].get('comparison_budget_including_selection')==224:
  print('Fair-budget comparison already complete');return
 results['initial_search_results']=results['runs'].copy();results['runs']=json.loads(json.dumps(results['runs']))
 val=json.loads((OUT/'validation.json').read_text());validation_wall=float(results['validation_wall_seconds']); results['validation_wall_seconds']=validation_wall
 for record in results['runs']:
  seed=record['seed'];initial=json.loads((OUT/f'initial_{seed}.json').read_text());old=json.loads((OUT/f'direct_{seed}.json').read_text());cache={tuple(r['x']):r for r in old};history=list(initial);seen={tuple(r['x']) for r in history};budget=224;random.seed(seed)
  p=Problem(4,3,2);p.types[:]=[Real(*b) for b in BOUNDS];p.constraints[:]='<=0'
  def fn(z):
   row=cache[tuple(z)];o=convert([row['y']])[0];return [-o[0],o[1],o[2]],[o[1]-.95,o[2]-10]
  p.function=fn
  class Batch(Evaluator):
   def evaluate_all(self,jobs,**kw):
    jobs=list(jobs);xs=[]
    for job in jobs:
     key=tuple(job.solution.variables)
     if key not in seen and len(seen)<budget:
      seen.add(key);xs.append(key)
    new=[z for z in xs if z not in cache]
    for rr in Parallel(n_jobs=4)(delayed(ev)(z) for z in new):
     if rr['error']:raise RuntimeError(rr['error'])
     cache[tuple(rr['x'])]=rr
    for key in xs:history.append(cache[key])
    for j in jobs:
     if tuple(j.solution.variables) not in seen:raise StopIteration()
     j.run()
    return jobs
  starts=[]
  for row in initial:sol=Solution(p);sol.variables[:]=row['x'];starts.append(sol)
  alg=NSGAII(p,population_size=N0,generator=InjectedPopulation(starts),evaluator=Batch());t=time.perf_counter()
  try:
   while len(history)<budget:alg.step()
  except StopIteration:pass
  added=time.perf_counter()-t
  # Replay must recover exactly the original 192 unique points.
  assert {tuple(r['x']) for r in history[:192]}=={tuple(r['x']) for r in old}
  vd,verifytime=verified(history,f'direct_fair_{seed}')
  timing=record['timing'];timing['direct_total_wall']=timing['initial_wall']+timing['direct_new_ode_wall']+added+verifytime;timing['direct_extra_32_wall']=added;timing['assisted_total_wall']+=validation_wall;timing['assisted_serial_equivalent_seconds']+=sum(a['seconds'] for a in val);timing['direct_serial_equivalent_seconds']=sum(a['seconds'] for a in history)+sum(a['seconds'] for a in vd)
  record['direct']=stats(history);record['direct_tight_subset']=stats([a for a in vd if not a['error']]);record['direct_budget']=224;record['assisted_budget_including_validation']=224
  (OUT/f'direct_fair_{seed}.json').write_text(json.dumps(history));print('FAIR',seed,record['assisted'],record['direct'],flush=True)
 results['configuration']['comparison_budget_including_selection']=224
 path.write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
