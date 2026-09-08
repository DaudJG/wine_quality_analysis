"""Run with uv run --locked python verify.py [--notebook] [--write-notebook]."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
import numpy as np
import pandas as pd
import app
import wine_quality_utils as utils


class ProjectChecks(unittest.TestCase):
    def test_source_and_training_preprocessing(self):
        data = pd.read_csv(ROOT / 'winequality-red.csv')
        self.assertEqual(hashlib.sha256((ROOT / 'winequality-red.csv').read_bytes()).hexdigest(),
                         'd6a0d9bd24806944818795f22500c46cb6424cbff517aacda36595d3ed9b2daa')
        self.assertEqual(data.shape, (1599, 12))
        self.assertEqual(int(data.duplicated().sum()), 240)
        self.assertFalse(data.isna().any().any())
        train, test, ytrain, ytest = utils.preprocess_data(data, 'quality')
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertEqual(len(train) + len(test), len(data))
        np.testing.assert_allclose(train.drop(columns='const').mean(), 0, atol=1e-12)
        # Test values must be transformed with training statistics.
        raw_train = data.loc[train.index].drop(columns='quality')
        expected = (data.loc[test.index].drop(columns='quality') - raw_train.mean()) / raw_train.std(ddof=0)
        np.testing.assert_allclose(test.drop(columns='const'), expected, atol=1e-12)
        for options in ({}, {'robust': True}, {'wls': True}):
            model, _ = utils.fit_and_evaluate_model(train, ytrain, **options)
            pred = model.predict(test)
            self.assertEqual(len(pred), len(ytest))
            self.assertTrue(np.isfinite(pred).all())
            if options.get('wls'):
                self.assertTrue((model.model.weights > 0).all())

    def test_pearson_helper(self):
        data = pd.DataFrame({'a': [1., 2., 3., 4.], 'b': [2., 4., 6., 8.]})
        corr, pvals = utils.calculate_pearson_correlation(data)
        self.assertAlmostEqual(corr.loc['a', 'b'], 1.)
        self.assertEqual(corr.loc['a', 'b'], corr.loc['b', 'a'])
        self.assertLess(pvals.loc['a', 'b'], 0.05)

    def test_dashboard_and_callback_requests(self):
        client = app.app.server.test_client()
        for route in ('/', '/_dash-layout', '/_dash-dependencies'):
            self.assertEqual(client.get(route).status_code, 200)
        pairs = [(a, b) for a in app.df.columns[:-1] for b in app.df.columns[:-1]] + [(None, None), ('invalid', None)]
        for first, second in pairs:
            hist, scatter = app.update_plots(first, second)
            self.assertTrue(len(hist.data))
            self.assertEqual(sum(len(t.x) for t in scatter.data), 1599)
        key = next(k for k in app.app.callback_map if 'histogram-plot' in k)
        result = client.post('/_dash-update-component', json={
            'output': key,
            'outputs': [{'id': 'histogram-plot', 'property': 'figure'}, {'id': 'scatter-plot', 'property': 'figure'}],
            'inputs': [{'id': 'feature-dropdown-1', 'property': 'value', 'value': 'Alcohol'}, {'id': 'feature-dropdown-2', 'property': 'value', 'value': 'Volatile Acidity'}],
            'state': [], 'changedPropIds': ['feature-dropdown-1.value']})
        self.assertEqual(result.status_code, 200)
        self.assertIn('scatter-plot', result.get_json()['response'])
        self.assertEqual(sum(len(t.x) for t in app.generate_additional_plot(None).data), 1599)


def execute_notebook(write_back=False):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook = nbformat.read(ROOT / 'wine_quality_analysis.ipynb', as_version=4)
    code_count = sum(c.cell_type == 'code' for c in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    notebook.cells.append(nbformat.v4.new_code_cell('''
import json
assert len(wine_data) == 1359
sets = [set(X_train.index), set(X_validation.index), set(X_test.index)]
assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i + 1, 3))
assert set.union(*sets) == set(wine_data.index)
np.testing.assert_allclose(classification_scaler.mean_, X_train_raw.mean(), atol=1e-12)
np.testing.assert_allclose(X_train.mean(), 0, atol=1e-12)
assert np.isfinite(y_pred_proba).all() and ((y_pred_proba >= 0) & (y_pred_proba <= 1)).all()
assert np.isfinite(model_wls.model.weights).all() and (model_wls.model.weights > 0).all()
assert set(box_train_index) == set(X_train_box.index)
assert np.isfinite(df_metrics[['MSE', 'R_squared']].to_numpy()).all()
assert best_threshold == thresholds[np.argmax(f1_scores)]
assert np.isclose(f1_score(y_validation, validation_proba >= best_threshold), max(f1_scores))
assert len(y_pred_best_threshold) == len(y_test)
print(json.dumps({'deduplicated_rows': len(wine_data), 'classification_splits': list(map(len, sets)),
                  'threshold': float(best_threshold), 'test_auc': float(roc_auc),
                  'test_f1': float(f1_score(y_test, y_pred_best_threshold)),
                  'regression_mse': df_metrics.set_index('Model_Name')['MSE'].to_dict()}))
'''))
    km = KernelManager(kernel_name='python3')
    # Always use the invoking locked environment, not a globally installed kernel.
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(notebook, km=km, timeout=600, allow_errors=False, resources={'metadata': {'path': str(ROOT)}})
    print(f'Executing {code_count} notebook code cells in a fresh kernel...', flush=True)
    client.execute(cleanup_kc=True)
    checks = notebook.cells.pop()
    print(''.join(o.get('text', '') for o in checks.outputs), flush=True)
    figures = sum('image/png' in output.get('data', {}) for cell in notebook.cells for output in cell.get('outputs', []))
    assert figures >= 30, f'Expected rendered notebook charts, found {figures}'
    # Keep warnings, but do not publish machine-specific home or environment paths.
    for cell in notebook.cells:
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                for prefix, label in ((str(ROOT), '[project]'), (sys.prefix, '[environment]'), (str(Path.home()), '[home]')):
                    output.text = output.text.replace(prefix, label)
    out = ROOT / 'verification-output'
    out.mkdir(exist_ok=True)
    nbformat.write(notebook, out / 'wine_quality_analysis.executed.ipynb')
    if write_back:
        nbformat.write(notebook, ROOT / 'wine_quality_analysis.ipynb')
    print(f'Notebook passed: {code_count} code cells, {figures} charts and numerical/split checks.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook', action='store_true')
    parser.add_argument('--write-notebook', action='store_true')
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProjectChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print('Dataset SHA256:', hashlib.sha256((ROOT / 'winequality-red.csv').read_bytes()).hexdigest())
    if args.notebook or args.write_notebook:
        execute_notebook(args.write_notebook)
