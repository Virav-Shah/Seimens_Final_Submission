"""Use the saved single-condition models without retraining.

Example:
python repo/pinn_small_trial/single_condition_predict.py --log-rates 4 -2 4 -2
The association rates are in M^-1 s^-1 and dissociation rates in s^-1.
"""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import torch
from balanced_model import BalancedNet
from single_condition_trial import prediction
from train import endpoints, aggregate

def predict(log_rates):
    root = Path(__file__).resolve().parents[2]
    out = root/'output/pinn_single_condition'
    x = np.asarray(log_rates, dtype=float).reshape(-1, 4)
    lo, hi = np.array([2, -5, 2, -5]), np.array([7, 2, 7, 2])
    if not np.isfinite(x).all() or np.any(x < lo) or np.any(x > hi):
        raise ValueError('Rates outside the trained domain; extrapolation is not supported')
    d = np.load(out/'training.npz')
    model = BalancedNet(d['x'], d['y'])
    checkpoint = torch.load(out/'PINN_checkpoint.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    pk = endpoints(prediction(model, x)[:, :, 0])
    gp = joblib.load(out/'GP_checkpoint.joblib')
    gk = gp['output_scaler'].inverse_transform(gp['model'].predict(gp['input_scaler'].transform(x)))[:, None]
    rates = 10.**x
    result = {'condition': 'reference only', 'log10_rates': x.tolist(), 'rates': rates.tolist(),
              'KD1_M': (rates[:, 1]/rates[:, 0]).tolist(), 'KD2_M': (rates[:, 3]/rates[:, 2]).tolist(),
              'population_order': ['tumour', 'healthy_neither', 'healthy_T1', 'healthy_T2', 'healthy_both'],
              'aggregate_order': ['tumour', 'overall_healthy', 'worst_healthy_subgroup'],
              'PINN_killing_pct': pk[:, 0].tolist(), 'GP_killing_pct': gk[:, 0].tolist(),
              'PINN_aggregate_pct': aggregate(pk).tolist(), 'GP_aggregate_pct': aggregate(gk).tolist(),
              'note': 'Predictions, not ODE-confirmed acceptance. Both models are restricted to the saved reference biology.'}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-rates', type=float, nargs=4, required=True,
                        metavar=('LOG_KON1', 'LOG_KOFF1', 'LOG_KON2', 'LOG_KOFF2'))
    args = parser.parse_args()
    print(json.dumps(predict(args.log_rates), indent=2))
