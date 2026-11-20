"""One fixed, single-condition PINN fit and an endpoint GP on identical designs.

No training restart, test-driven tuning, or search-performance inference.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import copy, hashlib, json, time, warnings
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import joblib
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from data import TIMES, coordinates, trajectory, INPUT_SET
from train import tensor, endpoints, aggregate, accuracy
from balanced_model import BalancedNet, KNOTS
from balanced_trial import GROUPS, norms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/pinn_single_condition'
OUT.mkdir(exist_ok=True)
SEED = 20261121
TRAIN_SECONDS = 1200
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2))

def solve(x):
    start = time.perf_counter()
    try:
        return {'y': trajectory(x, 0), 'seconds': time.perf_counter()-start, 'error': None}
    except Exception as e:
        return {'x': list(x), 'error': repr(e), 'seconds': time.perf_counter()-start}

def dataset(name, n, seed):
    path = OUT/f'{name}.npz'
    if path.exists():
        d = np.load(path)
        return d['x'], d['y']
    x = coordinates(n, seed)
    start = time.perf_counter()
    rows = Parallel(n_jobs=4)(delayed(solve)(a) for a in x)
    failed = [r for r in rows if r['error']]
    write(f'{name}_generation.json', {'seconds': time.perf_counter()-start,
          'individual_solve_seconds': [r['seconds'] for r in rows], 'errors': failed})
    if failed:
        raise RuntimeError('ODE failures retained; trial stopped')
    y = np.stack([r['y'] for r in rows])[:, None]
    np.savez_compressed(path, x=x, y=y, t=TIMES)
    print('DATA', name, n, round(time.perf_counter()-start, 2), flush=True)
    return x, y

def prediction(m, x, times=np.array([24.])):
    outputs = []
    with torch.no_grad():
        # Condition is always reference. A/B are never evaluated.
        for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/16)))):
            t = tensor(np.tile(times, len(chunk)))
            z = tensor(np.repeat(chunk, len(times), axis=0))
            outputs.append(m(t, z, torch.zeros(len(t), dtype=torch.long)).numpy().reshape(len(chunk), 1, len(times), 97))
    return np.concatenate(outputs)

def fit_pinn(x, y, xv, yv):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    m = BalancedNet(x, y)
    # Retain the same physical architecture, but the condition indicator is fixed.
    params = tuple(m.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    xt, yt, tt = tensor(x), tensor(y[:, 0]), tensor(TIMES)
    # Compress the wide range of positive state magnitudes in the supervised loss.
    # This is an asinh loss, not a log transform of concentrations or of the ODE.
    state_floor = .01*m.scale
    best, state, selected, hist = float('inf'), None, None, []
    logweights = None
    start = last = time.perf_counter()
    step = 0
    while time.perf_counter()-start < TRAIN_SECONDS-2:
        elapsed = time.perf_counter()-start
        progress = elapsed/TRAIN_SECONDS
        ramp = float(np.clip((progress-.1)/.2, 0, 1))
        lr = 1e-4 + .5*9e-4*(1+np.cos(np.pi*progress))
        for group in opt.param_groups:
            group['lr'] = lr
        ix = torch.as_tensor(rng.integers(0, len(x), 256))
        ti = torch.as_tensor(rng.integers(0, len(TIMES), 256))
        yp = m(tt[ti], xt[ix], torch.zeros(256, dtype=torch.long))
        trajectory_loss = ((torch.asinh(yp/state_floor)-torch.asinh(yt[ix, ti]/state_floor))/5).square().mean()
        ie = torch.as_tensor(rng.integers(0, len(x), 32))
        pe = m(torch.full((32,), 24.), xt[ie], torch.zeros(32, dtype=torch.long))
        pk = -100*torch.expm1(-pe[:, 87:92])
        ak = -100*torch.expm1(-yt[ie, -1, 87:92])
        err = pk-ak
        endpoint_loss = (err/tensor([10, 1, 2, 2, 5])).square().mean() + ((err[:, 1:]@tensor([.79, .1, .1, .01]))/.25).square().mean()
        supervised = trajectory_loss+endpoint_loss
        loss = supervised
        ploss = {}
        if ramp:
            seg = rng.integers(0, len(KNOTS)-1, 64)
            ct = tensor(KNOTS[seg]+rng.uniform(.001, .999, 64)*(KNOTS[seg+1]-KNOTS[seg]))
            bounds = np.array([[2, 7], [-5, 2], [2, 7], [-5, 2]])
            cx = tensor(rng.uniform(bounds[:, 0], bounds[:, 1], (64, 4)))
            _, residual = m.residual(ct, cx, torch.zeros(64, dtype=torch.long))
            ploss = {name: residual[:, ids].square().mean() for name, ids in GROUPS.items()}
            if logweights is None or step % 25 == 0:
                ref = max(norms(supervised, params), 1e-30)
                new = {name: float(np.log(np.clip(ref/max(norms(l, params), 1e-30), 1e-30, 1e6))) for name, l in ploss.items()}
                logweights = new if logweights is None else {n: .9*logweights[n]+.1*new[n] for n in new}
            # A predeclared, modest physics contribution avoids overwhelming data.
            loss = supervised+.1*ramp*sum(np.exp(logweights[n])*l for n, l in ploss.items())/len(ploss)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; no restart permitted')
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.)
        opt.step()
        step += 1
        if time.perf_counter()-last > 30 or step == 1:
            met = accuracy(endpoints(yv[:, :, -1]), endpoints(prediction(m, xv)[:, :, 0]))
            score = float(np.mean([met[n]['mae_pp']/s for n, s in zip(['tumour', 'overall_healthy', 'worst_subgroup'], [1, .05, .5])]))
            rec = {'step': step, 'seconds': time.perf_counter()-start, 'physics_ramp': ramp, 'learning_rate': lr,
                   'trajectory_loss': float(trajectory_loss.detach()), 'endpoint_loss': float(endpoint_loss.detach()),
                   'validation_score': score, 'validation': met, 'physics_weights': {n: float(np.exp(w)) for n, w in (logweights or {}).items()},
                   'physics_losses': {n: float(l.detach()) for n, l in ploss.items()}}
            hist.append(rec)
            if ramp == 1 and score < best:
                best, state, selected = score, copy.deepcopy(m.state_dict()), rec
                torch.save({'state_dict': state, 'selected': selected}, OUT/'PINN_checkpoint.pt')
            write('training_history.json', hist)
            last = time.perf_counter()
            print('PINN', step, round(rec['seconds']), {k: round(met[k]['mae_pp'], 4) for k in ['tumour', 'overall_healthy', 'worst_subgroup']}, flush=True)
    if state is None:
        raise RuntimeError('No full-physics checkpoint')
    duration = time.perf_counter()-start
    m.load_state_dict(state)
    np.savez(OUT/'PINN_scaling.npz', input_mean=m.mean.numpy(), input_std=m.std.numpy(), state_scale=m.scale.numpy())
    record = {'training_seconds': duration, 'updates': step, 'selected': selected,
              'training_limit_seconds': TRAIN_SECONDS, 'restarts': 0}
    write('PINN_training.json', record)
    return m, record

class GP:
    def fit(self, x, y):
        self.xs = StandardScaler().fit(x)
        self.ys = StandardScaler().fit(y)
        self.optimizer_record = {}
        def optimizer(obj, theta, bounds):
            start = time.perf_counter()
            # Bound a single hyperparameter fit; no restarts or test selection.
            def callback(z):
                if time.perf_counter()-start > 120:
                    raise StopIteration
            r = minimize(obj, theta, jac=True, bounds=bounds, method='L-BFGS-B', callback=callback,
                         options={'maxiter': 100, 'maxfun': 150})
            self.optimizer_record = {'seconds': time.perf_counter()-start, 'iterations': int(r.nit),
                                     'success': bool(r.success), 'message': str(r.message)}
            return r.x, r.fun
        kernel = ConstantKernel(1., (.01, 100))*Matern(np.ones(4), (.05, 20), nu=2.5)+WhiteKernel(.001, (1e-6, .1))
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=False,
                                             n_restarts_optimizer=0, optimizer=optimizer, random_state=SEED)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.model.fit(self.xs.transform(x), self.ys.transform(y))
            self.warnings = [str(a.message) for a in w]
        self.seconds = time.perf_counter()-start
        return self
    def predict(self, x):
        return self.ys.inverse_transform(self.model.predict(self.xs.transform(x)))

def main():
    start = time.perf_counter()
    if (OUT/'results.json').exists() or (OUT/'PINN_checkpoint.pt').exists():
        raise RuntimeError('Existing trial found: refusing an unrecorded training restart')
    ref = ROOT/'repo/mechanism_comparison/model.py'
    cfg = {'seed': SEED, 'condition_index': 0, 'condition_name': INPUT_SET[0][0],
           'biology': asdict(INPUT_SET[0][1]), 'killing': asdict(INPUT_SET[0][2]),
           'training_designs': 512, 'validation_designs': 64, 'test_designs': 128,
           'partition_seeds': [SEED, SEED+1, SEED+2], 'training_limit_seconds': TRAIN_SECONDS,
           'bounds_log10': [[2, 7], [-5, 2], [2, 7], [-5, 2]], 'knots_h': KNOTS.tolist(),
           'source_sha256': hashlib.sha256(ref.read_bytes()).hexdigest(), 'data_source': 'reference-condition ODE simulations only',
           'loss': 'asinh-scaled trajectory error + endpoint error + 0.1 times gradient-balanced grouped ODE loss',
           'checkpoint_selection': 'validation mean of MAE/1, MAE/0.05, MAE/0.5; full physics only',
           'training': 'one fresh initialisation, Adam cosine 1e-3 to 1e-4; warmup 10%, ramp 20%; CPU float64',
           'comparison': 'same 512 designs; PINN gets trajectories, GP gets five 24-hour killing outputs; no test selection',
           'GP': 'standardised log rates and standardised outputs; Matern 5/2; same kernel family as earlier GP; no output clipping',
           'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    write('configuration.json', cfg)
    x, y = dataset('training', 512, SEED)
    xv, yv = dataset('validation', 64, SEED+1)
    assert not set(map(tuple, x)) & set(map(tuple, xv))
    # Check only the chosen condition, including exact initial state/accounting.
    torch.manual_seed(SEED)
    m0 = BalancedNet(x, y)
    xx = tensor(x[:8]); cc = torch.zeros(8, dtype=torch.long)
    torch.testing.assert_close(m0(torch.zeros(8), xx, cc), m0.physics.initial(cc), atol=0, rtol=0)
    ts = tensor(np.geomspace(1e-6, 24, 8)); states = m0(ts, xx, cc)
    assert states.min() >= 0 and m0.physics.balance(states, ts, cc).abs().max() < 1e-12
    write('preflight.json', {'initial_conditions': True, 'nonnegative': True, 'accounting': True,
          'equation_copy': 'same previously checked Physics implementation; reference source checksum recorded'})
    m, trained = fit_pinn(x, y, xv, yv)
    gp = GP().fit(x, endpoints(y[:, :, -1])[:, 0])
    gp.model.optimizer = None  # Fitted hyperparameters retained; make checkpoint portable.
    joblib.dump({'model': gp.model, 'input_scaler': gp.xs, 'output_scaler': gp.ys}, OUT/'GP_checkpoint.joblib')
    gp_record = {'training_seconds': gp.seconds, 'kernel': str(gp.model.kernel_),
                 'optimizer': gp.optimizer_record, 'warnings': gp.warnings}
    write('GP_training.json', gp_record)
    # The final test trajectories are generated only after both fits are frozen.
    tx, ty = dataset('test', 128, SEED+2)
    assert not set(map(tuple, tx)) & (set(map(tuple, x)) | set(map(tuple, xv)))
    tp = prediction(m, tx, TIMES)
    pk = endpoints(tp[:, :, -1]); gk = gp.predict(tx)[:, None]; actual = endpoints(ty[:, :, -1])
    np.savez_compressed(OUT/'predictions.npz', x=tx, actual=actual, PINN=pk, GP=gk, trajectories=tp, t=TIMES)
    metrics = {'PINN': accuracy(actual, pk), 'GP': accuracy(actual, gk)}
    balance = 0.
    with torch.no_grad():
        for row in tp:
            balance = max(balance, float(m.physics.balance(tensor(row[0]), tensor(TIMES), torch.zeros(len(TIMES), dtype=torch.long)).abs().max()))
    metrics['PINN'].update(negative_states=int((tp < 0).sum()), max_inventory_error_fraction=balance,
                           scaled_trajectory_mae=float(np.mean(abs(tp-ty)/m.scale.numpy())))
    cap = -100*np.expm1(-24*np.array(INPUT_SET[0][2].maximum_rate_h))
    for name, pred in [('PINN', pk), ('GP', gk)]:
        metrics[name]['negative_killing_predictions'] = int((pred < 0).sum())
        metrics[name]['above_ODE_killing_bound'] = int((pred > cap+1e-8).sum())
    # Same warmed batch and repetitions, endpoint prediction only.
    speed = {}
    for name, fn in [('PINN', lambda: prediction(m, tx)), ('GP', lambda: gp.predict(tx))]:
        fn(); durations = []
        for _ in range(10):
            t0 = time.perf_counter(); fn(); durations.append(time.perf_counter()-t0)
        speed[name] = {'median_seconds_128_designs': float(np.median(durations)),
                       'milliseconds_per_design': float(np.median(durations)*1000/len(tx))}
    result = {'status': 'completed', 'metrics': metrics, 'PINN_training': trained, 'GP_training': gp_record,
              'prediction_timing': speed, 'total_wall_seconds': time.perf_counter()-start,
              'source_unchanged': cfg['source_sha256'] == hashlib.sha256(ref.read_bytes()).hexdigest(),
              'optimisation_run': False, 'interpretation': 'one completed single-condition prediction benchmark; no search-quality or clinical claim'}
    write('results.json', result)
    print('COMPLETE', json.dumps(metrics), flush=True)

if __name__ == '__main__':
    main()
