"""Verify saved checkpoints, partitions, notebook and physical accounting."""
import contextlib, hashlib, io, json
from pathlib import Path
import numpy as np
import nbformat
from single_condition_predict import predict

def main():
    root = Path(__file__).resolve().parents[2]
    out = root/'output/pinn_single_condition'
    d = np.load(out/'predictions.npz')
    answer = predict(d['x'][:3])
    np.testing.assert_allclose(answer['PINN_killing_pct'], d['PINN'][:3, 0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(answer['GP_killing_pct'], d['GP'][:3, 0], rtol=1e-10, atol=1e-10)
    parts = [set(map(tuple, np.load(out/f'{name}.npz')['x'])) for name in ['training', 'validation', 'test']]
    assert all(not parts[i]&parts[j] for i in range(3) for j in range(i))
    cfg = json.loads((out/'configuration.json').read_text())
    result = json.loads((out/'results.json').read_text())
    assert result['source_unchanged']
    assert hashlib.sha256((root/'repo/mechanism_comparison/model.py').read_bytes()).hexdigest() == cfg['source_sha256']
    assert result['PINN_training']['training_seconds'] + result['GP_training']['training_seconds'] < 1800
    assert all(np.isfinite(d[n]).all() for n in ['PINN', 'GP', 'actual', 'trajectories'])
    nb = nbformat.read(root/'repo/Bispecific_ADC_PINN_Single_Condition.ipynb', as_version=4)
    nbformat.validate(nb)
    ns = {}
    # In-process execution of display-only notebook cells; no new training.
    for cell in nb.cells:
        if cell.cell_type == 'code':
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(cell.source, '<saved-results-cell>', 'exec'), ns)
    checks = {'checkpoint_predictions_match_saved_results': True, 'partitions_disjoint': True,
              'reference_model_unchanged': True, 'total_model_training_under_30_minutes': True,
              'finite_saved_predictions': True, 'notebook_schema_valid': True,
              'notebook_cells_checked_in_process': True, 'notebook_kernel_execution': False,
              'test_designs': len(d['x']), 'training_condition_count': 1,
              'clinical_validation': False}
    (out/'verification.json').write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    main()
