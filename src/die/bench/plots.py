"""Charts for benchmark results.

Two figures, because there are two different questions:

- **The heatmap** answers "who beats inaction, and where". It is *diverging*,
  not sequential, and centred on the do-nothing baseline for each scenario --
  the meaningful quantity on this env is not how much energy a policy gathered
  but whether it gathered more than a policy that never moved. Agents pay to
  act, so raw magnitude hides the sign of the result: a return of +2 looks
  respectable on a sequential ramp and is actually a loss.
- **The response curves** answer "how does each agent degrade along an axis".
  One panel per axis, seeds as an error band.

The three training seeds of each learned agent are collapsed into one series
with a visible spread rather than drawn as three lines. That is the honest
shape: docs/JOURNAL.md's recurring failure mode on this benchmark is reading a
seed difference as an effect, and a band makes the seed spread impossible to
overlook.

Palette follows the validated categorical order (blue, green, magenta, yellow,
aqua, orange); the six slots pass the lightness, chroma, CVD and normal-vision
gates on a light surface. Three of them sit under 3:1 contrast, which obligates
"relief" -- so every curve is also directly labelled and the underlying CSV is
the table view.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # headless: these are written to files, never shown
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

#: Validated categorical order. Assigned to agents in a fixed order and never
#: cycled, so an agent keeps its colour no matter which subset is plotted.
SERIES_COLORS = ('#2a78d6', '#008300', '#e87ba4', '#eda100', '#1baf7a', '#eb6834')

#: Diverging ramp for "better/worse than doing nothing": two poles that read as
#: opposite, with a neutral gray midpoint so "no difference" reads as nothing.
#: Never a rainbow, and never a hue at the midpoint.
_DIVERGING = LinearSegmentedColormap.from_list('die_div', [
    '#7a1616', '#b32a2a', '#e34948', '#f09a9a', '#fbd5d5',
    '#f0efec',
    '#9ec5f4', '#3987e5', '#1c5cab', '#104281', '#0d366b',
])

TEXT_PRIMARY, TEXT_SECONDARY, GRID = '#0b0b0b', '#52514e', '#dcdbd6'

#: Display order and grouping. Learned checkpoints arrive as one file per
#: training seed; they are folded into a single series here.
AGENT_ORDER = ('null', 'brownian', 'gradient', 'physarum', 'rl_ppo', 'jx_nca')
AGENT_LABELS = {
    'null': 'do nothing', 'brownian': 'brownian', 'gradient': 'gradient',
    'physarum': 'physarum', 'rl_ppo': 'PPO (RL)', 'jx_nca': 'NCA (evolved)',
}


def agent_group(name: str) -> str:
    """Folds ``rl_ppo_seed1`` into ``rl_ppo``; leaves scripted names alone."""
    return name.split('_seed')[0]


def _style(ax) -> None:
    """Recessive axes: the data should be the most prominent thing present."""
    ax.set_facecolor('#fcfcfb')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _declutter(positions: list[float], min_gap: float) -> list[float]:
    """Nudges end-of-line labels apart while preserving their vertical order.

    Six series on one panel routinely end within a few percent of each other,
    and overlapping labels are worse than none -- they misattribute a curve.
    Sorting and pushing upward keeps each label nearest its own line and never
    swaps two, so identity survives the adjustment.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    adjusted = list(positions)
    for rank, index in enumerate(order[1:], start=1):
        previous = adjusted[order[rank - 1]]
        if adjusted[index] - previous < min_gap:
            adjusted[index] = previous + min_gap
    return adjusted


def _grouped(rows: list[dict]) -> dict[tuple[str, str], list[float]]:
    """``(agent_group, scenario) -> returns``, pooling seeds and checkpoints."""
    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        pooled[(agent_group(row['agent']), row['scenario'])].append(row['return'])
    return pooled


def plot_heatmap(rows: list[dict], out_path: Path) -> Path:
    """Advantage over inaction, per agent x scenario.

    Colour is the *difference* from the ``null`` policy on the same scenario,
    so each column is read against its own baseline -- a scarce world scores
    lower for everyone, and that shared shift is not an agent's fault.
    """
    pooled = _grouped(rows)
    scenarios = list(dict.fromkeys(row['scenario'] for row in rows))
    agents = [a for a in AGENT_ORDER if any(key[0] == a for key in pooled)]

    baseline = {s: np.mean(pooled[('null', s)]) for s in scenarios
                if ('null', s) in pooled}
    grid = np.array([[np.mean(pooled[(a, s)]) - baseline.get(s, 0.)
                      if (a, s) in pooled else np.nan
                      for s in scenarios] for a in agents])

    # Scale on a robust quantile, not the maximum. One scenario (wave-flow)
    # produces advantages an order of magnitude larger than the rest, and
    # scaling to it flattens the other eighteen columns into indistinguishable
    # gray. Outliers saturate instead; every cell carries its exact value as a
    # label, so saturation costs no information.
    finite = np.abs(grid[np.isfinite(grid)])
    limit = float(np.quantile(finite, 0.93)) if finite.size else 1.
    limit = limit or float(np.nanmax(finite)) or 1.
    saturated = int((np.abs(grid) > limit).sum())
    figure, ax = plt.subplots(figsize=(1.0 + 0.52 * len(scenarios),
                                       1.2 + 0.42 * len(agents)))
    image = ax.imshow(grid, cmap=_DIVERGING, aspect='auto',
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit))

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=7.5)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([AGENT_LABELS.get(a, a) for a in agents], fontsize=9)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ax.spines.values():
        side.set_visible(False)

    # Direct value labels: the relief the contrast gate requires, and they let
    # the figure be read without referring back to the colour bar.
    for row_index in range(grid.shape[0]):
        for col_index in range(grid.shape[1]):
            value = grid[row_index, col_index]
            if np.isnan(value):
                continue
            strong = abs(value) > 0.62 * limit
            ax.text(col_index, row_index, f'{value:+.1f}', ha='center',
                    va='center', fontsize=6.5,
                    color='#ffffff' if strong else TEXT_PRIMARY)

    bar = figure.colorbar(image, ax=ax, pad=0.015, fraction=0.025, extend='both')
    label = 'return minus do-nothing, same scenario'
    if saturated:
        label += f'\n({saturated} cells beyond ±{limit:.0f}, see labels)'
    bar.set_label(label, fontsize=8, color=TEXT_SECONDARY)
    bar.ax.tick_params(labelsize=7, colors=TEXT_SECONDARY, length=2)
    bar.outline.set_visible(False)

    ax.set_title('Does the agent beat doing nothing?', fontsize=11,
                 color=TEXT_PRIMARY, pad=10, loc='left')
    figure.tight_layout()
    figure.savefig(out_path, dpi=200, facecolor='#fcfcfb')
    plt.close(figure)
    return out_path


def plot_axis_curves(rows: list[dict], out_path: Path) -> Path:
    """One panel per environment axis; band is the spread across world seeds."""
    pooled_raw: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    axes_values: dict[str, set[float]] = defaultdict(set)
    baseline_row = {}
    for row in rows:
        if row['axis'] == 'baseline':
            baseline_row[agent_group(row['agent'])] = row['scenario']
            continue
        pooled_raw[(agent_group(row['agent']), row['axis'], row['value'])].append(
            row['return'])
        axes_values[row['axis']].add(row['value'])

    # The baseline is the shared control every axis passes through, so it is
    # drawn into each panel rather than shown once elsewhere.
    baseline_pool: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row['axis'] == 'baseline':
            baseline_pool[agent_group(row['agent'])].append(row['return'])

    # A response curve needs something to respond along. Single-valued axes
    # (food_flow is on/off, not a magnitude) would plot as isolated dots on an
    # invented x-range; they are categorical and the heatmap already shows them.
    axis_names = [a for a in dict.fromkeys(r['axis'] for r in rows)
                  if a != 'baseline' and len(axes_values[a]) > 1]
    agents = [a for a in AGENT_ORDER
              if any(key[0] == a for key in pooled_raw)]
    colors = {agent: SERIES_COLORS[i % len(SERIES_COLORS)]
              for i, agent in enumerate(AGENT_ORDER)}

    columns = min(3, len(axis_names))
    rows_n = int(np.ceil(len(axis_names) / columns))
    figure, grid_axes = plt.subplots(rows_n, columns,
                                     figsize=(4.4 * columns, 3.4 * rows_n),
                                     squeeze=False)

    for index, axis_name in enumerate(axis_names):
        ax = grid_axes[index // columns][index % columns]
        _style(ax)
        values = sorted(axes_values[axis_name])
        ends: list[tuple[float, float, str]] = []
        for agent in agents:
            means, lows, highs, xs = [], [], [], []
            for value in values:
                pool = pooled_raw.get((agent, axis_name, value))
                if not pool:
                    continue
                xs.append(value)
                means.append(np.mean(pool))
                lows.append(np.mean(pool) - np.std(pool, ddof=1))
                highs.append(np.mean(pool) + np.std(pool, ddof=1))
            if not xs:
                continue
            ax.fill_between(xs, lows, highs, color=colors[agent], alpha=0.13,
                            linewidth=0)
            ax.plot(xs, means, color=colors[agent], linewidth=2.0,
                    marker='o', markersize=4.5,
                    markeredgecolor='#fcfcfb', markeredgewidth=0.8,
                    label=AGENT_LABELS.get(agent, agent))
            ends.append((xs[-1], means[-1], agent))

        if baseline_pool:
            level = float(np.mean(baseline_pool.get('null', [0.])))
            ax.axhline(level, color=TEXT_SECONDARY, linewidth=1.0,
                       linestyle=(0, (4, 3)), alpha=0.8)

        # Direct labels last, once the y-scale is settled, so the gap that
        # separates them is computed against the axis actually drawn.
        low, high = ax.get_ylim()
        ax.set_ylim(low, high)
        placed = _declutter([end[1] for end in ends], 0.052 * (high - low))
        for (x_end, _, agent), y_text in zip(ends, placed):
            ax.annotate(AGENT_LABELS.get(agent, agent), (x_end, y_text),
                        xytext=(5, 0), textcoords='offset points', fontsize=6.5,
                        color=colors[agent], va='center', annotation_clip=False)
        if baseline_pool:
            # Anchored to the left edge, away from the labels on the right.
            ax.annotate('do nothing', (values[0], level), xytext=(0, 3),
                        textcoords='offset points', fontsize=6.5,
                        color=TEXT_SECONDARY, ha='left')

        ax.set_title(axis_name, fontsize=10, color=TEXT_PRIMARY, loc='left')
        ax.set_xlabel(axis_name, fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylabel('episode return', fontsize=8, color=TEXT_SECONDARY)
        ax.margins(x=0.22)

    for empty in range(len(axis_names), rows_n * columns):
        grid_axes[empty // columns][empty % columns].set_visible(False)

    handles, labels = grid_axes[0][0].get_legend_handles_labels()
    legend = figure.legend(handles, labels, loc='lower center',
                           ncol=min(6, len(labels)), frameon=False, fontsize=9,
                           bbox_to_anchor=(0.5, -0.005))
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    figure.suptitle('Agent response to each environment axis '
                    '(band = spread over world seeds)',
                    fontsize=12, color=TEXT_PRIMARY, x=0.01, ha='left')
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(out_path, dpi=200, facecolor='#fcfcfb',
                   bbox_inches='tight')
    plt.close(figure)
    return out_path
