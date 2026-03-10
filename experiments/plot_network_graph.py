#!/usr/bin/env python3
"""
Generate static network graph figures for paper (white background).

Uses network_snapshot.npz and spike dump data to create publication-quality
network visualizations.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from pathlib import Path
import colorsys

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.5,
    'pdf.fonttype': 42,
})

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / 'network_snapshot.npz'
DUMP_DIR = ROOT / 'results' / 'activity_vis' / 'dump'
OUT_DIR = ROOT / 'results' / 'activity_vis'

C_INPUT = '#27ae60'
C_RES_E = '#c0392b'
C_RES_I = '#2874a6'


def load_network():
    snap = np.load(SNAPSHOT)
    n = int(snap['n_neurons'])
    pos = snap['positions'].reshape(n, 3)
    weights = snap['weights'].reshape(n, n)
    is_inh = snap['is_inhibitory'].astype(bool)
    input_idx = set(snap['input_neuron_indices'].tolist())
    input_zone_idx = set(snap['input_zone_indices'].tolist())
    res_idx = set(snap['reservoir_zone_indices'].tolist())
    sphere_radius = float(snap['sphere_radius'])

    neurons_csv = DUMP_DIR / 'neurons.csv'
    mel_bins = {}
    if neurons_csv.exists():
        df = pd.read_csv(neurons_csv)
        for _, row in df.iterrows():
            nid = int(row['neuron_id'])
            bins = [int(b) for b in str(row['freq_bins']).split(';')]
            ws = [float(w) for w in str(row['weights']).split(';')]
            tw = sum(ws)
            mel_bins[nid] = sum(b * w for b, w in zip(bins, ws)) / tw if tw > 0 else bins[0]

    radii = np.linalg.norm(pos, axis=1)

    return dict(n=n, pos=pos, weights=weights, is_inh=is_inh,
                input_idx=input_idx, input_zone_idx=input_zone_idx,
                res_idx=res_idx, mel_bins=mel_bins, sphere_radius=sphere_radius,
                radii=radii)


def neuron_color(i, net):
    if i in net['input_idx']:
        t = net['mel_bins'].get(i, 64) / 127.0
        # Map to green–yellow range (hue 0.15–0.42) with wider lightness contrast
        h = 0.15 + t * 0.27
        l = 0.30 + t * 0.25  # dark green → bright yellow-green
        r, g, b = colorsys.hls_to_rgb(h, l, 0.85)
        return (r, g, b)
    elif net['is_inh'][i]:
        return matplotlib.colors.to_rgb(C_RES_I)
    else:
        return matplotlib.colors.to_rgb(C_RES_E)


def project_2d(pos, elev_deg=25, azim_deg=135):
    elev = np.radians(elev_deg)
    azim = np.radians(azim_deg)
    cam = np.array([np.cos(elev) * np.cos(azim),
                    np.cos(elev) * np.sin(azim),
                    np.sin(elev)])
    cam /= np.linalg.norm(cam)
    up = np.array([0, 0, 1.0])
    right = np.cross(cam, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, cam)
    return pos @ right, pos @ up2, pos @ cam


def get_all_edges(net):
    W = net['weights']
    rows, cols = np.nonzero(W)
    return [(int(rows[k]), int(cols[k]), abs(W[rows[k], cols[k]]),
             W[rows[k], cols[k]] > 0) for k in range(len(rows))]


# =========================================================================
# A: 2D sphere projection — annotated, shell/core readable
# =========================================================================
def fig_A_sphere_projection(net, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor('white')

    pos = net['pos']
    n = net['n']
    R = net['sphere_radius']
    proj_x, proj_y, depth = project_2d(pos)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)

    colors = np.array([neuron_color(i, net) for i in range(n)])

    all_edges = get_all_edges(net)
    weights_arr = np.array([e[2] for e in all_edges])
    w_95 = np.percentile(weights_arr, 95)

    # Draw edges: inhibitory behind, then excitatory
    for draw_exc in [False, True]:
        segs = []
        edge_cols = []
        for src, tgt, w, is_exc in all_edges:
            if is_exc != draw_exc:
                continue
            w_norm = min(w / w_95, 1.0)
            d = min(depth_norm[src], depth_norm[tgt])
            alpha = (w_norm ** 0.7) * (0.008 + 0.05 * d)
            if is_exc:
                edge_cols.append((0.75, 0.15, 0.15, alpha))
            else:
                edge_cols.append((0.15, 0.35, 0.65, alpha * 1.5))
            segs.append([(proj_x[src], proj_y[src]), (proj_x[tgt], proj_y[tgt])])
        if segs:
            lc = LineCollection(segs, colors=edge_cols, linewidths=0.15, zorder=1)
            ax.add_collection(lc)

    # Draw neurons back to front, input shell larger + brighter
    order = np.argsort(depth_norm)
    for i in order:
        d = depth_norm[i]
        is_input = i in net['input_idx']
        if is_input:
            s = 12 + 22 * d
            alpha = 0.5 + 0.5 * d
            ec = '#333'
            ew = 0.2 if d > 0.4 else 0
        else:
            s = 2 + 6 * d
            alpha = 0.25 + 0.75 * d
            ec = 'none'
            ew = 0
        ax.scatter(proj_x[i], proj_y[i], c=[colors[i]], s=s, alpha=alpha,
                   edgecolors=ec, linewidths=ew, zorder=2 + d)

    # Sphere outline
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(R * np.cos(theta), R * np.sin(theta), color='#bbb', alpha=0.25, lw=0.7)

    # Subtle shell halo (no text labels)
    input_radii = np.array([net['radii'][i] for i in net['input_idx']])
    shell_r = np.median(input_radii)
    ax.plot(shell_r * np.cos(theta), shell_r * np.sin(theta),
            color=C_INPUT, alpha=0.08, lw=6, zorder=0)

    # Legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_INPUT,
               markeredgecolor='#333', markeredgewidth=0.3,
               markersize=7, label=f'Input (n={len(net["input_idx"])})', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_RES_E,
               markersize=5.5, label=f'Excitatory (n={sum(1 for i in net["res_idx"] if not net["is_inh"][i])})',
               linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_RES_I,
               markersize=5.5, label=f'Inhibitory (n={sum(1 for i in net["res_idx"] if net["is_inh"][i])})',
               linestyle='None'),
        Line2D([0], [0], color=(0.75, 0.15, 0.15), lw=1.5, alpha=0.6,
               label='Excitatory synapse'),
        Line2D([0], [0], color=(0.15, 0.35, 0.65), lw=1.5, alpha=0.6,
               label='Inhibitory synapse'),
    ]
    leg = ax.legend(handles=handles, loc='lower right', fontsize=8,
                    frameon=True, facecolor='white', edgecolor='#ccc',
                    framealpha=0.95, borderpad=0.6, handletextpad=0.4)
    leg.get_frame().set_linewidth(0.5)

    ax.set_xlim(-R * 1.12, R * 1.12)
    ax.set_ylim(-R * 1.12, R * 1.12)
    ax.set_aspect('equal')
    ax.set_axis_off()

    if standalone:
        fig = ax.get_figure()
        path = OUT_DIR / 'fig_A_sphere.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {path} + .pdf")


# =========================================================================
# B: Connectivity matrix — split E/I, darker dots, no overlapping text
# =========================================================================
def fig_B_connectivity_matrix(net, ax=None, fig_ref=None):
    standalone = ax is None
    n = net['n']
    W = net['weights']

    input_neurons = sorted(net['input_idx'], key=lambda i: net['mel_bins'].get(i, 64))
    res_e = sorted((i for i in net['res_idx'] if not net['is_inh'][i]),
                   key=lambda i: net['radii'][i])
    res_i = sorted((i for i in net['res_idx'] if net['is_inh'][i]),
                   key=lambda i: net['radii'][i])
    order = input_neurons + res_e + res_i
    boundaries = [len(input_neurons), len(input_neurons) + len(res_e)]
    zone_labels = ['Input', 'Res. E', 'Res. I']
    zone_colors = [C_INPUT, C_RES_E, C_RES_I]

    idx = np.array(order)
    W_ord = W[np.ix_(idx, idx)]

    # Single panel with diverging colormap: blue (inhibitory) — white — red (excitatory)
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')
    else:
        fig = fig_ref

    # Colormap with aggressive ramp so sparse dots are clearly visible
    cmap_div = LinearSegmentedColormap.from_list('div',
        ['#082a45', '#0a3d62', '#1a5276', '#f6f6f6', '#922b21', '#641e16', '#3b0f0c'])

    vmax_pos = np.percentile(W_ord[W_ord > 0], 85) if np.any(W_ord > 0) else 1
    vmax_neg = np.percentile(-W_ord[W_ord < 0], 85) if np.any(W_ord < 0) else 1
    vlim = max(vmax_pos, vmax_neg)

    im = ax.imshow(W_ord, cmap=cmap_div, vmin=-vlim, vmax=vlim,
                   aspect='equal', interpolation='nearest')
    if standalone:
        ax.set_title('Synaptic weight matrix', fontsize=9, fontweight='bold', pad=10)

    # Zone boundaries
    for b in boundaries:
        ax.axhline(b - 0.5, color='#888', lw=0.5, ls='--', alpha=0.6)
        ax.axvline(b - 0.5, color='#888', lw=0.5, ls='--', alpha=0.6)

    # Zone labels on LEFT (y-axis) — imshow origin='upper' so row 0 is at top
    prev = 0
    for k, b in enumerate(boundaries):
        mid = (prev + b) / 2
        ax.text(-0.02, 1.0 - mid / len(order), zone_labels[k],
                ha='right', va='center', fontsize=9,
                color=zone_colors[k], fontweight='bold',
                transform=ax.transAxes)
        prev = b
    mid = (boundaries[-1] + len(order)) / 2
    ax.text(-0.02, 1.0 - mid / len(order), zone_labels[-1],
            ha='right', va='center', fontsize=9,
            color=zone_colors[-1], fontweight='bold',
            transform=ax.transAxes)

    # Zone labels on BOTTOM (x-axis)
    prev = 0
    for k, b in enumerate(boundaries):
        mid = (prev + b) / 2
        ax.text(mid / len(order), -0.02, zone_labels[k],
                ha='center', va='top', fontsize=9,
                color=zone_colors[k], fontweight='bold',
                rotation=0, transform=ax.transAxes)
        prev = b
    mid = (boundaries[-1] + len(order)) / 2
    ax.text(mid / len(order), -0.02, zone_labels[-1],
            ha='center', va='top', fontsize=9,
            color=zone_colors[-1], fontweight='bold',
            transform=ax.transAxes)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#aaa')
        spine.set_linewidth(0.4)

    ax.set_ylabel('Pre-synaptic', fontsize=10, fontweight='bold', labelpad=40)
    ax.set_xlabel('Post-synaptic', fontsize=10, fontweight='bold', labelpad=20)

    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=30)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Weight (red = exc, blue = inh)', fontsize=9, fontweight='bold', labelpad=5)

    if standalone:
        fig.tight_layout()
        path = OUT_DIR / 'fig_B_connectivity.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {path} + .pdf")


# =========================================================================
# C: Zone schematic — compact self-loops, separated E↔I labels
# =========================================================================
def fig_C_zone_schematic(net, ax=None):
    n = net['n']
    W = net['weights']

    zone_map = {}
    for i in range(n):
        if i in net['input_idx']:
            zone_map[i] = 0
        elif net['is_inh'][i]:
            zone_map[i] = 2
        else:
            zone_map[i] = 1

    zone_short = ['Input\n(shell)', 'Excitatory\n(core)', 'Inhibitory\n(core)']
    zone_colors = [C_INPUT, C_RES_E, C_RES_I]
    zone_counts = [sum(1 for v in zone_map.values() if v == z) for z in range(3)]

    agg_w = np.zeros((3, 3))
    agg_n = np.zeros((3, 3))
    for i in range(n):
        for j in range(n):
            if W[i, j] != 0:
                zi, zj = zone_map.get(i, -1), zone_map.get(j, -1)
                if zi >= 0 and zj >= 0:
                    agg_w[zi, zj] += abs(W[i, j])
                    agg_n[zi, zj] += 1

    # Triangle layout
    positions = np.array([
        (0, 1.8),       # Input (top)
        (-1.5, -0.8),   # Excitatory (bottom-left)
        (1.5, -0.8),    # Inhibitory (bottom-right)
    ])
    max_count = max(zone_counts)
    node_radii = [0.3 + 0.35 * np.sqrt(c / max_count) for c in zone_counts]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='white')
    ax.set_facecolor('white')

    max_w = agg_w.max()
    layout_center = positions.mean(axis=0)

    # Pre-defined self-loop directions (hand-tuned to stay in bounds)
    # Input: loop goes UP, E: loop goes DOWN-LEFT, I: loop goes DOWN-RIGHT
    self_loop_dirs = [
        np.array([0, 1]),          # Input → up
        np.array([-0.85, -0.55]),  # E → down-left
        np.array([0.85, -0.55]),   # I → down-right
    ]

    # Draw self-loops first (behind everything)
    for i in range(3):
        if agg_w[i, i] < max_w * 0.005:
            continue

        pi = positions[i]
        ri = node_radii[i]
        w_frac = agg_w[i, i] / max_w
        lw = 1.5 + 4.5 * w_frac
        alpha = 0.3 + 0.5 * w_frac

        outward = self_loop_dirs[i]
        outward = outward / np.linalg.norm(outward)

        # Smaller, tighter loop
        loop_r = ri * 0.55 + 0.08
        loop_center = pi + outward * (ri + loop_r * 0.7)

        # Open arc (not full circle) — 300 degrees
        theta_start = np.arctan2(-outward[1], -outward[0])
        theta = np.linspace(theta_start + 0.5, theta_start + 2 * np.pi - 0.5, 80)
        lx = loop_center[0] + loop_r * np.cos(theta)
        ly = loop_center[1] + loop_r * np.sin(theta)

        ax.plot(lx, ly, color=zone_colors[i], lw=lw, alpha=alpha,
                solid_capstyle='round', zorder=1)

        # Arrowhead
        ax.annotate('', xy=(lx[-1], ly[-1]),
                    xytext=(lx[-5], ly[-5]),
                    arrowprops=dict(arrowstyle='-|>',
                                    color=zone_colors[i],
                                    lw=lw * 0.7,
                                    mutation_scale=9 + 7 * w_frac),
                    zorder=1)

        # Label outside the loop
        label_pos = loop_center + outward * (loop_r + 0.15)
        ax.text(label_pos[0], label_pos[1],
                f'{int(agg_n[i, i]):,}',
                ha='center', va='center', fontsize=6.5, color='#777',
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Draw inter-zone edges
    # Collect unique pairs: for E↔I draw one double-arrow, for others draw directed
    drawn_pairs = set()
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if agg_w[i, j] < max_w * 0.005:
                continue

            is_ei_pair = {i, j} == {1, 2}

            # For E↔I: draw once as a single double-headed straight line
            if is_ei_pair:
                pair_key = (min(i, j), max(i, j))
                if pair_key in drawn_pairs:
                    continue
                drawn_pairs.add(pair_key)

                pi = positions[1]  # E
                pj = positions[2]  # I
                ri = node_radii[1]
                rj = node_radii[2]
                direction = pj - pi
                dist = np.linalg.norm(direction)
                unit = direction / dist

                # Combined weight for line thickness
                w_frac = max(agg_w[1, 2], agg_w[2, 1]) / max_w
                lw = 1.0 + 3.0 * w_frac
                alpha = 0.5 + 0.4 * w_frac

                # Leave room for arrowheads at each end
                arrow_len = 0.18
                line_start = pi + unit * (ri + 0.06 + arrow_len)
                line_end = pj - unit * (rj + 0.06 + arrow_len)
                tip_start = pi + unit * (ri + 0.06)   # E-side arrowhead tip
                tip_end = pj - unit * (rj + 0.06)     # I-side arrowhead tip

                # Gradient line: red on E side, blend to blue on I side
                n_seg = 60
                t_arr = np.linspace(0, 1, n_seg + 1)
                pts = np.column_stack([
                    line_start[0] + t_arr * (line_end[0] - line_start[0]),
                    line_start[1] + t_arr * (line_end[1] - line_start[1])
                ])
                segs = [[(pts[k, 0], pts[k, 1]), (pts[k+1, 0], pts[k+1, 1])]
                        for k in range(n_seg)]
                colors_grad = []
                c_e = np.array(matplotlib.colors.to_rgb(C_RES_E))
                c_i = np.array(matplotlib.colors.to_rgb(C_RES_I))
                for k in range(n_seg):
                    frac = (k + 0.5) / n_seg
                    c = (1 - frac) * c_e + frac * c_i
                    colors_grad.append((*c, alpha))
                lc = LineCollection(segs, colors=colors_grad, linewidths=lw, zorder=1,
                                    capstyle='round')
                ax.add_collection(lc)

                # Arrowheads at the true terminus of each end
                ax.annotate('', xy=tuple(tip_end), xytext=tuple(line_end),
                            arrowprops=dict(arrowstyle='-|>', color=C_RES_I,
                                            lw=lw * 0.6, mutation_scale=14),
                            zorder=2)
                ax.annotate('', xy=tuple(tip_start), xytext=tuple(line_start),
                            arrowprops=dict(arrowstyle='-|>', color=C_RES_E,
                                            lw=lw * 0.6, mutation_scale=14),
                            zorder=2)

                # Labels: E→I count above line, I→E count below line
                perp = np.array([-unit[1], unit[0]])
                mid = (line_start + line_end) / 2
                ax.text(mid[0] + perp[0] * 0.18, mid[1] + perp[1] * 0.18,
                        f'E→I: {int(agg_n[1, 2]):,}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=C_RES_E,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
                ax.text(mid[0] - perp[0] * 0.18, mid[1] - perp[1] * 0.18,
                        f'I→E: {int(agg_n[2, 1]):,}',
                        ha='center', va='top', fontsize=8, fontweight='bold', color=C_RES_I,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])
                continue

            # Non-E↔I edges: straight lines with gentle curve
            pi = positions[i]
            pj = positions[j]
            ri = node_radii[i]
            rj = node_radii[j]
            w_frac = agg_w[i, j] / max_w
            lw = 1.0 + 5.5 * w_frac
            alpha = 0.35 + 0.55 * w_frac

            direction = pj - pi
            dist = np.linalg.norm(direction)
            unit = direction / dist
            perp = np.array([-unit[1], unit[0]])

            start = pi + unit * (ri + 0.06)
            end = pj - unit * (rj + 0.13)

            # Gentle curve for non-E↔I (Input→E, Input→I)
            offset_sign = 1 if i < j else -1
            ctrl = (start + end) / 2 + perp * 0.3 * offset_sign

            t = np.linspace(0, 1, 60)
            curve = (np.outer((1 - t) ** 2, start) +
                     np.outer(2 * (1 - t) * t, ctrl) +
                     np.outer(t ** 2, end))
            ax.plot(curve[:, 0], curve[:, 1], color=zone_colors[i],
                    lw=lw, alpha=alpha, solid_capstyle='round', zorder=1)

            ax.annotate('', xy=tuple(end),
                        xytext=tuple(curve[-6]),
                        arrowprops=dict(arrowstyle='-|>',
                                        color=zone_colors[i],
                                        lw=lw * 0.7,
                                        mutation_scale=9 + 7 * w_frac),
                        zorder=2)

            # Label
            label_pos = ((1 - 0.5) ** 2 * start +
                         2 * (1 - 0.5) * 0.5 * ctrl +
                         0.5 ** 2 * end)
            label_offset = perp * 0.18 * offset_sign
            ax.text(label_pos[0] + label_offset[0],
                    label_pos[1] + label_offset[1],
                    f'{int(agg_n[i, j]):,}',
                    ha='center', va='center', fontsize=7.5, fontweight='bold', color='#666',
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    # Draw nodes (on top)
    for k in range(3):
        px, py = positions[k]
        r = node_radii[k]
        circle_fill = plt.Circle((px, py), r, color=zone_colors[k],
                                  alpha=0.12, zorder=3)
        ax.add_patch(circle_fill)
        circle_border = plt.Circle((px, py), r, fill=False,
                                    edgecolor=zone_colors[k], lw=2.2, zorder=4)
        ax.add_patch(circle_border)
        ax.text(px, py + 0.06, zone_short[k], ha='center', va='center',
                fontsize=10, fontweight='bold', color='#222', zorder=5,
                linespacing=1.1)
        ax.text(px, py - r + 0.1, f'n = {zone_counts[k]}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='#666', zorder=5)

    if standalone:
        ax.text(0, -2.0, '728 LIF neurons  |  distance-dependent connectivity',
                ha='center', va='center', fontsize=8, color='#999', style='italic')

    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.4, 3.4)
    ax.set_aspect('equal')
    ax.set_axis_off()

    if standalone:
        fig = ax.get_figure()
        path = OUT_DIR / 'fig_C_schematic.png'
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {path} + .pdf")


# =========================================================================
# Combined: A + B + C in one row
# =========================================================================
def fig_combined(net):
    fig = plt.figure(figsize=(24, 8), facecolor='white')

    # A: sphere projection
    ax_a = fig.add_axes([0.01, 0.04, 0.27, 0.85])
    fig_A_sphere_projection(net, ax=ax_a)
    ax_a.set_title('A   Spatial embedding', fontsize=12, fontweight='bold',
                    loc='left', pad=10)

    # B: connectivity matrix (wider for colorbar)
    ax_b = fig.add_axes([0.31, 0.04, 0.32, 0.85])
    fig_B_connectivity_matrix(net, ax=ax_b, fig_ref=fig)
    ax_b.set_title('B   Synaptic weight matrix', fontsize=12, fontweight='bold',
                    loc='left', pad=10)

    # C: zone schematic
    ax_c = fig.add_axes([0.70, 0.04, 0.28, 0.85])
    fig_C_zone_schematic(net, ax=ax_c)
    ax_c.set_title('C   Zone-level connectivity', fontsize=12, fontweight='bold',
                    loc='left', pad=10)

    fig.suptitle('Network structure', fontsize=18, fontweight='bold', y=0.99)

    path = OUT_DIR / 'fig_network_structure.png'
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path} + .pdf")


if __name__ == '__main__':
    print("Loading network...")
    net = load_network()
    print(f"  {net['n']} neurons, {len(net['input_idx'])} input, "
          f"{len(net['res_idx'])} reservoir")
    print(f"  {len(get_all_edges(net))} total connections")

    print("\nGenerating figures...")
    fig_A_sphere_projection(net)
    fig_B_connectivity_matrix(net)
    fig_C_zone_schematic(net)
    fig_combined(net)
    print("\nDone!")
