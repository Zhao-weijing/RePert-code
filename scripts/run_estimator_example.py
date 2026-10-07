"""Exercise the copied estimators on supplied, explicitly synthetic inputs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis/baselines'))
sys.path.insert(0, str(ROOT / 'analysis/repeat_calibration'))
from ablation_common import Rows
from run_rceb_screen import fit_rceb, rceb_predict, save_model, load_model
from run_validation_calibrated_ensemble import simplex_mse_weights, shrunken_weight


def main():
    with np.load(ROOT / 'examples/synthetic_repeat_profiles.npz', allow_pickle=False) as data:
        fitting = Rows('synthetic', 'CP', 'train', data['compound'], data['dose'], data['plate'], data['delta'])
        fit_ids = set(data['compound'].tolist())
        calibration_ids = set(data['calibration_compound'].tolist())
        query_ids = set(data['query_compound'].tolist())
        assert not (fit_ids & calibration_ids or fit_ids & query_ids or calibration_ids & query_ids)
        model, fit_record = fit_rceb(fitting, ridge_relative=0.01)
        fit_weight = simplex_mse_weights(data['calibration_candidates'], data['calibration_reference'])
        weight, gamma = shrunken_weight(fit_weight, len(calibration_ids))
        prediction = rceb_predict(model, data['query_support'], budget=1)
        assert prediction.shape == data['query_support'].shape
        assert np.isfinite(prediction).all()
        assert np.all(weight >= 0) and np.isclose(weight.sum(), 1)
        out = ROOT / 'outputs/example'
        out.mkdir(parents=True, exist_ok=True)
        save_model(out / 'synthetic_eb_model.npz', model)
        np.testing.assert_allclose(prediction, rceb_predict(load_model(out / 'synthetic_eb_model.npz'), data['query_support'], budget=1))
        np.savetxt(out / 'predictions.csv', prediction, delimiter=',', header=','.join(f'feature_{i}' for i in range(prediction.shape[1])), comments='')
        result = {'status': 'PASS', 'input_kind': 'synthetic_software_example',
                  'fit_compounds': len(fit_ids), 'calibration_compounds': len(calibration_ids),
                  'query_compounds': len(query_ids), 'feature_dimension': prediction.shape[1],
                  'role_overlap': 0, 'prediction_finite': True,
                  'fitted_weights': fit_weight.tolist(), 'shrunken_weights': weight.tolist(),
                  'shrinkage_gamma': gamma,
                  'scientific_validation': False}
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        expected = json.loads((ROOT / 'examples/expected_structure.json').read_text(encoding='utf-8'))
        for key, value in expected.items():
            assert result[key] == value, (key, result[key], value)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
