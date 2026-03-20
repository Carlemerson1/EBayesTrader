"""
analysis/plot_style.py

Shared style configuration for all EBayesTrader publication plots.

Usage:
    from analysis.plot_style import apply_style, C, RCPARAMS
    apply_style()   # call once at top of notebook or script

    fig, ax = plt.subplots(figsize=C.FIG_WIDE)
    ax.hist(..., color=C.BLUE_HIST, ...)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter


# ── Colour palette ─────────────────────────────────────────────────────────────

class Colors:
    # Blues
    NAVY          = '#00539B'   # primary brand colour
    NAVY_DARK     = '#003366'   # titles, strong emphasis
    BLUE_HIST     = '#2E6DA4'   # histogram bars
    KDE           = '#1A3F5C'   # KDE / density lines

    # Accents
    ORANGE        = '#E8913A'   # p < 0.05 region
    RED           = '#C0392B'   # p < 0.01 region / drawdown
    GREEN         = '#1A7A30'   # positive / significant

    # Neutrals
    GRAY          = '#7F8C8D'   # subtitles, secondary text
    TEXT_DARK     = '#2C3E50'   # axis labels, body text
    LIGHT_GRAY    = '#F4F6F8'   # axes background
    GRID          = '#FFFFFF'   # grid lines (white on light grey bg)
    SPINE         = '#CCCCCC'   # axis spines

    # Annotation
    BOX_EDGE      = '#00539B'   # annotation box border (= NAVY)
    BOX_FACE      = '#FFFFFF'   # annotation box fill

C = Colors()


# ── rcParams ───────────────────────────────────────────────────────────────────

RCPARAMS = {
    'font.family':       'Georgia',
    'axes.facecolor':    C.LIGHT_GRAY,
    'figure.facecolor':  'white',
    'axes.grid':         True,
    'grid.color':        C.GRID,
    'grid.linewidth':    1.2,
    'axes.axisbelow':    True,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'axes.labelsize':    11,
    'axes.labelcolor':   C.TEXT_DARK,
    'axes.labelpad':     8,
    'xtick.color':       '#AAAAAA',
    'ytick.color':       '#AAAAAA',
}


def apply_style():
    """Apply shared rcParams. Call once at the top of each notebook/script."""
    plt.rcParams.update(RCPARAMS)


# ── Figure sizes ───────────────────────────────────────────────────────────────

class FigSize:
    WIDE      = (12, 5.5)    # full-width single chart (permutation histogram)
    STANDARD  = (10, 5.5)    # standard single chart
    SQUARE    = (7, 7)       # scatter / correlation
    TALL      = (10, 7)      # multi-panel tall
    HALF      = (6, 5)       # side-by-side half-width

FS = FigSize()


# ── Typography helpers ─────────────────────────────────────────────────────────

TITLE_KW = dict(fontsize=16, fontweight='bold', color=C.NAVY_DARK, pad=15, loc='left')
SUBTITLE_Y = 1.0

def add_subtitle(ax, text):
    ax.text(0, SUBTITLE_Y, text,
            transform=ax.transAxes,
            fontsize=9.5, color=C.GRAY, va='bottom')


# ── Annotation box style ───────────────────────────────────────────────────────

ANNOT_BOX = dict(
    boxstyle='round,pad=0.45',
    facecolor=C.BOX_FACE,
    edgecolor=C.BOX_EDGE,
    linewidth=1.3,
    alpha=0.96
)

STATS_BOX = dict(
    boxstyle='round,pad=0.35',
    facecolor='white',
    edgecolor=C.SPINE,
    linewidth=0.8,
    alpha=0.92
)


# ── Spine / tick cleanup ───────────────────────────────────────────────────────

def clean_spines(ax):
    """Remove top/right spines, style left/bottom."""
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(C.SPINE)


# ── Formatters ─────────────────────────────────────────────────────────────────

PCT_FMT  = FuncFormatter(lambda x, _: f'{x:.0%}')
PCT1_FMT = FuncFormatter(lambda x, _: f'{x:.1%}')
PCT2_FMT = FuncFormatter(lambda x, _: f'{x:.2%}')
USD_FMT  = FuncFormatter(lambda x, _: f'${x:,.0f}')
NUM2_FMT = FuncFormatter(lambda x, _: f'{x:.2f}')


# ── Common legend patches ──────────────────────────────────────────────────────

def hist_patch(label='Null distribution'):
    return mpatches.Patch(color=C.BLUE_HIST, alpha=0.82, label=label)

def p05_patch(threshold='54%'):
    return mpatches.Patch(color=C.ORANGE, alpha=0.75, label=f'p < 0.05  (>{threshold})')

def p01_patch(threshold='58%'):
    return mpatches.Patch(color=C.RED, alpha=0.75, label=f'p < 0.01  (>{threshold})')

def kde_line_handle():
    return plt.Line2D([0], [0], color=C.KDE, linewidth=2.2, label='Kernel density')

def mean_line_handle(mean_val):
    return plt.Line2D([0], [0], color=C.NAVY, linewidth=1.8,
                      linestyle='-', label=f'Null mean ({mean_val:.1%})')


# ── Save helper ────────────────────────────────────────────────────────────────

def save(fig, path, dpi=300):
    """Save figure with consistent settings."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")