"""Render S3 using the original panel function and frozen contrast CSV."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / 'support'))
import render_figure4_cfra_mixed as original


def main():
    data = {'d_contrasts': pd.read_csv(ROOT / 'data/source_data/figure4_supp_downstream_contrasts.csv')}
    assert data['d_contrasts'].compound_count.tolist() == [262, 140, 146, 202]
    original.set_style()
    figure = original.plt.figure(figsize=(146.0 / 25.4, 64.0 / 25.4))
    frame = original.panel_d(figure, data)
    figure.axes[0].set_position([0.48, 0.18, 0.50, 0.76])
    for item in list(figure.texts):
        item.remove()
    for item in figure.findobj(match=original.Text):
        item.set_text(item.get_text().replace('CFRA', 'ReCA'))
    output = ROOT / 'outputs/figures'
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'svg', 'png'):
        figure.savefig(output / f'figure4_supp_downstream.{suffix}', dpi=220)
    assert len(frame) == 4
    print('Supplementary Fig. S3: 4 endpoints from frozen contrasts; no model fitting.')


if __name__ == '__main__':
    main()
