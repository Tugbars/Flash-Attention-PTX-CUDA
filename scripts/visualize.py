#!/usr/bin/env python3
"""
Flash Attention Visualization
Reads binary dumps from demo.cu and generates publication-quality plots.

Usage: python visualize.py
Requires: numpy, matplotlib

Outputs:
  figures/attention_heatmap.png   — Per-head attention weight heatmaps
  figures/attention_detail.png    — Zoomed attention patterns
  figures/error_analysis.png      — GPU vs CPU error distribution
  figures/performance.png         — Optimization progression chart
  figures/head_comparison.png     — How different heads attend differently
"""

import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.gridspec as gridspec
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)

# ============================================================================
# Binary reader
# ============================================================================
def read_binary(filename):
    """Read binary file written by demo.cu: header + float32 data."""
    with open(filename, 'rb') as f:
        ndims = np.frombuffer(f.read(4), dtype=np.int32)[0]
        shape = []
        for _ in range(ndims):
            shape.append(np.frombuffer(f.read(4), dtype=np.int32)[0])
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape(shape)


# ============================================================================
# Style configuration
# ============================================================================
COLORS = {
    'bg':      '#0a0a0f',
    'bg2':     '#12121a',
    'fg':      '#e8e6e1',
    'dim':     '#6b6878',
    'accent':  '#00ff88',
    'accent2': '#ff6b35',
    'accent3': '#4ecdc4',
    'hot':     '#ff2d55',
    'gold':    '#ffd700',
}

# Custom colormaps
attention_cmap = LinearSegmentedColormap.from_list('attention', [
    COLORS['bg'], '#1a1a3a', '#2d1b69', '#6b21a8', '#ff6b35', '#ffd700', '#ffffff'
])

error_cmap = LinearSegmentedColormap.from_list('error', [
    COLORS['accent'], '#1a3a2a', '#3a2a1a', COLORS['accent2'], COLORS['hot']
])

def style_axis(ax, title=None):
    """Apply dark theme to axis."""
    ax.set_facecolor(COLORS['bg2'])
    ax.tick_params(colors=COLORS['dim'], labelsize=8)
    ax.spines['bottom'].set_color(COLORS['dim'])
    ax.spines['left'].set_color(COLORS['dim'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, color=COLORS['fg'], fontsize=11, fontweight='bold', pad=10)


# ============================================================================
# Figure 1: Attention Heatmaps (all heads)
# ============================================================================
def plot_attention_heatmap(P, save_path):
    """Plot attention weight heatmaps for all heads."""
    B, H, S, _ = P.shape
    b = 0  # First batch

    fig, axes = plt.subplots(1, H, figsize=(4 * H, 4), facecolor=COLORS['bg'])
    fig.suptitle('Attention Weights by Head (Causal Mask)',
                 color=COLORS['fg'], fontsize=14, fontweight='bold', y=1.02)

    for h in range(H):
        ax = axes[h] if H > 1 else axes
        im = ax.imshow(P[b, h], cmap=attention_cmap, aspect='auto',
                       interpolation='nearest', vmin=0)
        style_axis(ax, f'Head {h}')
        ax.set_xlabel('Key Position', color=COLORS['dim'], fontsize=8)
        if h == 0:
            ax.set_ylabel('Query Position', color=COLORS['dim'], fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================================
# Figure 2: Attention Detail (zoomed)
# ============================================================================
def plot_attention_detail(P, save_path):
    """Zoomed view of attention patterns showing causal structure."""
    B, H, S, _ = P.shape
    b, h = 0, 0

    fig = plt.figure(figsize=(14, 5), facecolor=COLORS['bg'])
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.3)

    # Full attention matrix
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(P[b, h], cmap=attention_cmap, aspect='auto', vmin=0)
    style_axis(ax1, f'Full Attention (Head 0)')
    ax1.set_xlabel('Key Position', color=COLORS['dim'], fontsize=8)
    ax1.set_ylabel('Query Position', color=COLORS['dim'], fontsize=8)

    # Zoomed: first 32x32
    ax2 = fig.add_subplot(gs[1])
    zoom = min(32, S)
    im2 = ax2.imshow(P[b, h, :zoom, :zoom], cmap=attention_cmap, aspect='auto', vmin=0)
    style_axis(ax2, f'Zoomed (0:{zoom})')
    ax2.set_xlabel('Key Position', color=COLORS['dim'], fontsize=8)
    ax2.set_ylabel('Query Position', color=COLORS['dim'], fontsize=8)

    # Row-wise entropy
    ax3 = fig.add_subplot(gs[2])
    entropy = np.zeros(S)
    for i in range(S):
        p = P[b, h, i, :i+1]
        p = p[p > 1e-10]
        entropy[i] = -np.sum(p * np.log2(p))
    ax3.plot(entropy, color=COLORS['accent'], linewidth=1.5, alpha=0.9)
    ax3.fill_between(range(S), entropy, alpha=0.15, color=COLORS['accent'])
    style_axis(ax3, 'Attention Entropy')
    ax3.set_xlabel('Query Position', color=COLORS['dim'], fontsize=8)
    ax3.set_ylabel('Bits', color=COLORS['dim'], fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================================
# Figure 3: Error Analysis
# ============================================================================
def plot_error_analysis(O_ref, O_gpu, save_path):
    """Analyze GPU vs CPU output differences."""
    B, H, S, D = O_ref.shape

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=COLORS['bg'])
    fig.suptitle('GPU vs CPU Reference — Error Analysis',
                 color=COLORS['fg'], fontsize=14, fontweight='bold', y=1.02)

    # Per-element absolute error distribution
    ax = axes[0]
    err = np.abs(O_ref - O_gpu).flatten()
    ax.hist(err, bins=100, color=COLORS['accent'], alpha=0.8, edgecolor='none')
    ax.axvline(np.median(err), color=COLORS['gold'], linestyle='--', linewidth=1,
               label=f'Median: {np.median(err):.5f}')
    ax.set_yscale('log')
    style_axis(ax, 'Absolute Error Distribution')
    ax.set_xlabel('|ref - gpu|', color=COLORS['dim'], fontsize=8)
    ax.legend(fontsize=7, facecolor=COLORS['bg2'], edgecolor=COLORS['dim'],
              labelcolor=COLORS['fg'])

    # Per-row max error
    ax = axes[1]
    row_err = np.max(np.abs(O_ref[0, 0] - O_gpu[0, 0]), axis=1)
    ax.bar(range(S), row_err, color=COLORS['accent3'], alpha=0.8, width=1.0)
    style_axis(ax, 'Max Error per Row (Head 0)')
    ax.set_xlabel('Sequence Position', color=COLORS['dim'], fontsize=8)
    ax.set_ylabel('Max |error|', color=COLORS['dim'], fontsize=8)

    # Error heatmap (first head)
    ax = axes[2]
    err_map = np.abs(O_ref[0, 0] - O_gpu[0, 0])
    im = ax.imshow(err_map.T, cmap=error_cmap, aspect='auto',
                   interpolation='nearest')
    style_axis(ax, 'Error Heatmap (Head 0)')
    ax.set_xlabel('Sequence Position', color=COLORS['dim'], fontsize=8)
    ax.set_ylabel('Dimension', color=COLORS['dim'], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================================
# Figure 4: Performance Progression
# ============================================================================
def plot_performance(csv_path, save_path):
    """Optimization progression bar chart."""
    versions, tflops_vals, pct_vals = [], [], []
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            versions.append(parts[0])
            tflops_vals.append(float(parts[1]))
            pct_vals.append(float(parts[2]))

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS['bg'])

    colors = ['#555555', '#887766', COLORS['accent2'], COLORS['accent3'],
              COLORS['accent'], COLORS['gold']]
    colors = colors[:len(versions)]

    bars = ax.barh(range(len(versions)), tflops_vals, color=colors, height=0.6,
                   edgecolor='none', alpha=0.9)

    # Value labels
    for i, (bar, tf, pct) in enumerate(zip(bars, tflops_vals, pct_vals)):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f'{tf:.1f} TFLOPS ({pct:.0f}%)',
                color=COLORS['fg'], va='center', fontsize=9, fontweight='bold',
                fontfamily='monospace')

    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(versions, fontfamily='monospace')
    ax.invert_yaxis()
    style_axis(ax, 'Flash Attention Optimization Progression')
    ax.set_xlabel('TFLOPS (FP16)', color=COLORS['dim'], fontsize=10)
    ax.set_xlim(0, max(tflops_vals) * 1.35)

    # Speedup annotation
    if len(tflops_vals) >= 2:
        speedup = tflops_vals[-1] / tflops_vals[0]
        ax.annotate(f'{speedup:.0f}× total speedup',
                    xy=(tflops_vals[-1], len(versions)-1),
                    xytext=(tflops_vals[-1] * 0.6, 0.5),
                    color=COLORS['gold'], fontsize=11, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=COLORS['gold'], lw=1.5))

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================================
# Figure 5: Head Comparison
# ============================================================================
def plot_head_comparison(P, save_path):
    """Compare how different heads distribute attention."""
    B, H, S, _ = P.shape
    b = 0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor=COLORS['bg'])
    fig.suptitle('Per-Head Attention Behavior',
                 color=COLORS['fg'], fontsize=14, fontweight='bold', y=1.02)

    colors_h = [COLORS['accent'], COLORS['accent2'], COLORS['accent3'], COLORS['gold']]

    # Top-left: Attention span (how far back each position attends)
    ax = axes[0, 0]
    for h in range(min(H, 4)):
        # Weighted average attention distance
        positions = np.arange(S)
        avg_dist = np.zeros(S)
        for i in range(S):
            weights = P[b, h, i, :i+1]
            if weights.sum() > 0:
                avg_dist[i] = np.average(np.arange(i+1), weights=weights)
                avg_dist[i] = i - avg_dist[i]  # distance from current
        ax.plot(avg_dist, color=colors_h[h], alpha=0.8, linewidth=1.2,
                label=f'Head {h}')
    style_axis(ax, 'Average Attention Distance')
    ax.set_xlabel('Query Position', color=COLORS['dim'], fontsize=8)
    ax.set_ylabel('Avg. lookback distance', color=COLORS['dim'], fontsize=8)
    ax.legend(fontsize=7, facecolor=COLORS['bg2'], edgecolor=COLORS['dim'],
              labelcolor=COLORS['fg'])

    # Top-right: Attention entropy per head
    ax = axes[0, 1]
    for h in range(min(H, 4)):
        entropy = np.zeros(S)
        for i in range(S):
            p = P[b, h, i, :i+1]
            p = p[p > 1e-10]
            entropy[i] = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
        ax.plot(entropy, color=colors_h[h], alpha=0.8, linewidth=1.2,
                label=f'Head {h}')
    style_axis(ax, 'Attention Entropy (bits)')
    ax.set_xlabel('Query Position', color=COLORS['dim'], fontsize=8)
    ax.legend(fontsize=7, facecolor=COLORS['bg2'], edgecolor=COLORS['dim'],
              labelcolor=COLORS['fg'])

    # Bottom-left: Max attention weight
    ax = axes[1, 0]
    for h in range(min(H, 4)):
        max_w = np.max(P[b, h], axis=1)
        ax.plot(max_w, color=colors_h[h], alpha=0.8, linewidth=1.2,
                label=f'Head {h}')
    style_axis(ax, 'Max Attention Weight')
    ax.set_xlabel('Query Position', color=COLORS['dim'], fontsize=8)
    ax.legend(fontsize=7, facecolor=COLORS['bg2'], edgecolor=COLORS['dim'],
              labelcolor=COLORS['fg'])

    # Bottom-right: Diagonal dominance (how much attention goes to recent tokens)
    ax = axes[1, 1]
    window = 8
    for h in range(min(H, 4)):
        local = np.zeros(S)
        for i in range(S):
            start = max(0, i - window + 1)
            local[i] = np.sum(P[b, h, i, start:i+1])
        ax.plot(local, color=colors_h[h], alpha=0.8, linewidth=1.2,
                label=f'Head {h}')
    ax.axhline(y=1.0, color=COLORS['dim'], linestyle=':', linewidth=0.5)
    style_axis(ax, f'Local Attention (last {window} tokens)')
    ax.set_xlabel('Query Position', color=COLORS['dim'], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, facecolor=COLORS['bg2'], edgecolor=COLORS['dim'],
              labelcolor=COLORS['fg'])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'], edgecolor='none')
    plt.close()
    print(f"  Saved {save_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  Flash Attention Visualization")
    print("=" * 60)

    # Check for data files
    required = ['attention_weights.bin', 'output_gpu.bin', 'output_ref.bin']
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print(f"\nError: Missing data files: {missing}")
        print("Run ./flash_demo first to generate them.")
        sys.exit(1)

    os.makedirs('figures', exist_ok=True)

    # Load data
    print("\nLoading data...")
    P = read_binary('attention_weights.bin')
    O_gpu = read_binary('output_gpu.bin')
    O_ref = read_binary('output_ref.bin')
    print(f"  Attention weights: {P.shape}")
    print(f"  GPU output:        {O_gpu.shape}")
    print(f"  CPU reference:     {O_ref.shape}")

    # Generate figures
    print("\nGenerating figures...")
    plot_attention_heatmap(P, 'figures/attention_heatmap.png')
    plot_attention_detail(P, 'figures/attention_detail.png')
    plot_error_analysis(O_ref, O_gpu, 'figures/error_analysis.png')
    plot_head_comparison(P, 'figures/head_comparison.png')

    if os.path.exists('perf_data.csv'):
        plot_performance('perf_data.csv', 'figures/performance.png')

    print(f"\n{'=' * 60}")
    print(f"  All figures saved to figures/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
