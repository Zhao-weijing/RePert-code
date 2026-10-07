"""Shared journal-scale plotting style. All figure text must be English."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.python_libs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text
import json

COLORS = {
    'raw': '#85898F', 'imr': '#7563A5', 'cp': '#7563A5',
    'lso': '#478D9F', 'imceb': '#3C786F', 'ge': '#3C786F',
    'cfra': '#C47A37', 'update': '#C47A37', 'reference': '#252A31',
    'control': '#85898F', 'shuffle': '#A6ABB1', 'foreign': '#5B7186',
    'text': '#252A31', 'light': '#ECEEF1', 'zero': '#A5A9AE',
}
MARKERS = {'raw':'o','imr':'o','lso':'s','imceb':'^','cfra':'D','update':'D','reference':'s'}

def setup():
    plt.rcParams.update({
        'font.family':'Arial', 'font.size':8, 'axes.labelsize':8,
        'axes.titlesize':8.5, 'xtick.labelsize':8, 'ytick.labelsize':8,
        'legend.fontsize':8, 'figure.titlesize':9,
        'text.color':COLORS['text'], 'axes.labelcolor':COLORS['text'],
        'axes.edgecolor':COLORS['text'], 'xtick.color':COLORS['text'],
        'ytick.color':COLORS['text'], 'axes.linewidth':0.6,
        'lines.linewidth':1.1, 'lines.markersize':4,
        'xtick.major.width':0.6, 'ytick.major.width':0.6,
        'xtick.major.size':3, 'ytick.major.size':3,
        'axes.spines.top':False, 'axes.spines.right':False,
        'axes.grid':False, 'axes.axisbelow':True,
        'legend.frameon':False, 'legend.handlelength':1.4,
        'legend.borderaxespad':0.2, 'legend.labelspacing':0.35,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
        'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white', 'savefig.dpi':600,
        'axes.unicode_minus':True,
    })

def new_figure(height_mm=185, width_mm=180):
    setup()
    return plt.figure(figsize=(width_mm/25.4,height_mm/25.4))

def panel_label(ax, letter, x=-0.18, y=1.08):
    return ax.text(x,y,letter.lower(),transform=ax.transAxes,
                   weight='bold',fontsize=10,va='bottom',ha='left')

def panel_heading(ax, title):
    # Short identifying labels, not claim headlines or full figure titles.
    ax.set_title(title, loc='left', pad=7, weight='normal')

def clean_axis(ax, *, horizontal=False, zero=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if horizontal:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if zero: ax.axvline(0,color=COLORS['zero'],lw=0.7,ls=(0,(3,3)),zorder=0)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if zero: ax.axhline(0,color=COLORS['zero'],lw=0.7,ls=(0,(3,3)),zorder=0)

def errorbar(ax,x,y,lo,hi,color='imr',marker='o',horizontal=True,**kwargs):
    import numpy as np
    x=np.asarray(x); y=np.asarray(y)
    value=x if horizontal else y
    err=np.array([value-np.asarray(lo),np.asarray(hi)-value])
    kw={'xerr':err} if horizontal else {'yerr':err}
    return ax.errorbar(x,y,fmt=marker,color=COLORS.get(color,color),
        elinewidth=1.0,capsize=2,capthick=0.8,ms=4,zorder=3,**kw,**kwargs)

def save_figure(fig, name, *, root=ROOT):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[]; clipped=[]; non_english=[]
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text(): continue
        txt=obj.get_text()
        if any('\u3400' <= c <= '\u9fff' for c in txt): non_english.append(txt)
        box=obj.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            clipped.append(txt)
        texts.append({'text':txt,'font_pt':obj.get_fontsize()})
    if non_english: raise ValueError('Non-English figure text: '+repr(non_english))
    for ext in ['pdf','svg','png']:
        fig.savefig(root/f'{name}.{ext}',dpi=600)
    fig.savefig(root/f'{name}_preview.png',dpi=180)
    qa={'width_mm':round(fig.get_figwidth()*25.4,3),'height_mm':round(fig.get_figheight()*25.4,3),
        'min_font_pt':min(t['font_pt'] for t in texts), 'cjk_text':non_english,
        'text_outside_canvas':clipped, 'texts':texts}
    (root/'notes').mkdir(exist_ok=True)
    (root/'notes'/f'{name}_render_audit.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in qa.items() if k!='texts'}))
    return qa

setup()
