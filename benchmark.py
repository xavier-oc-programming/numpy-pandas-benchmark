import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from pathlib import Path

PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)
SIZES = [100_000, 1_000_000, 10_000_000]
SEED = 42
N_RUNS = 5


# ── data generation ───────────────────────────────────────────────────────────

def generate_numpy(size):
    """Return a structured ndarray with four named fields."""
    rng = np.random.default_rng(SEED)
    return np.array(
        list(zip(
            rng.uniform(10, 1000, size).astype(np.float64),   # price
            rng.integers(1, 101, size).astype(np.int32),       # quantity
            rng.integers(0, 5, size).astype(np.int8),          # category
            rng.uniform(0, 1, size).astype(np.float32),        # score
        )),
        dtype=[
            ('price',    np.float64),
            ('quantity', np.int32),
            ('category', np.int8),
            ('score',    np.float32),
        ]
    )


def generate_pandas(size):
    """Return a DataFrame built from the same seed — identical data to NumPy."""
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        'price':    rng.uniform(10, 1000, size).astype(np.float64),
        'quantity': rng.integers(1, 101, size).astype(np.int32),
        'category': rng.integers(0, 5, size).astype(np.int8),
        'score':    rng.uniform(0, 1, size).astype(np.float32),
    })


# ── timing ────────────────────────────────────────────────────────────────────

def time_it(func, n_runs=N_RUNS):
    """Call func() n_runs times with perf_counter; return mean elapsed seconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return float(np.mean(times))


# ── benchmarks ────────────────────────────────────────────────────────────────

def run_benchmarks(size):
    """Time all 7 operations for one dataset size; return structured results dict."""
    print(f'  Generating data ({size:,} rows)...')
    arr = generate_numpy(size)
    df  = generate_pandas(size)

    results = {}

    # 1. sort
    np_sort_time = time_it(lambda: np.sort(arr['price']))
    pd_sort_time = time_it(lambda: df.sort_values('price'))
    speedup = pd_sort_time / np_sort_time
    results['sort'] = {
        'numpy': np_sort_time,
        'pandas': pd_sort_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    # 2. filter — boolean mask selection
    np_filter_time = time_it(lambda: arr[arr['price'] > 500])
    pd_filter_time = time_it(lambda: df[df['price'] > 500])
    speedup = pd_filter_time / np_filter_time
    results['filter'] = {
        'numpy': np_filter_time,
        'pandas': pd_filter_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    # 3. sum
    np_sum_time = time_it(lambda: np.sum(arr['price']))
    pd_sum_time = time_it(lambda: df['price'].sum())
    speedup = pd_sum_time / np_sum_time
    results['sum'] = {
        'numpy': np_sum_time,
        'pandas': pd_sum_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    # 4. mean
    np_mean_time = time_it(lambda: np.mean(arr['price']))
    pd_mean_time = time_it(lambda: df['price'].mean())
    speedup = pd_mean_time / np_mean_time
    results['mean'] = {
        'numpy': np_mean_time,
        'pandas': pd_mean_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    # 5. std
    np_std_time = time_it(lambda: np.std(arr['price']))
    pd_std_time = time_it(lambda: df['price'].std())
    speedup = pd_std_time / np_std_time
    results['std'] = {
        'numpy': np_std_time,
        'pandas': pd_std_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    # 6. fillna — np.where replaces NaN; no NaNs in this dataset but cost is identical
    np_fillna_time = time_it(lambda: np.where(np.isnan(arr['score']), 0, arr['score']))
    pd_fillna_time = time_it(lambda: df['score'].fillna(0))
    speedup = pd_fillna_time / np_fillna_time
    results['fillna'] = {
        'numpy': np_fillna_time,
        'pandas': pd_fillna_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    # 7. groupby — manual loop vs Pandas split-apply-combine
    np_groupby_time = time_it(
        lambda: np.array([arr['price'][arr['category'] == i].mean() for i in range(5)])
    )
    pd_groupby_time = time_it(lambda: df.groupby('category')['price'].mean())
    speedup = pd_groupby_time / np_groupby_time
    results['groupby'] = {
        'numpy': np_groupby_time,
        'pandas': pd_groupby_time,
        'speedup': round(speedup, 3),
        'winner': 'NumPy' if speedup > 1 else 'Pandas',
    }

    return results


# ── report ────────────────────────────────────────────────────────────────────

def print_table(size, results):
    print(f'\n{"─" * 72}')
    print(f'  {size:,} rows')
    print(f'{"─" * 72}')
    print(f'  {"Operation":<10}  {"NumPy (ms)":>11}  {"Pandas (ms)":>12}  {"Speedup":>9}  {"Winner":<8}')
    print(f'  {"─"*10}  {"─"*11}  {"─"*12}  {"─"*9}  {"─"*8}')
    for op, r in results.items():
        np_ms  = r['numpy']  * 1000
        pd_ms  = r['pandas'] * 1000
        print(
            f'  {op:<10}  {np_ms:>11.3f}  {pd_ms:>12.3f}  '
            f'{r["speedup"]:>8.2f}x  {r["winner"]:<8}'
        )
    print(f'{"─" * 72}')


# ── visualisations ────────────────────────────────────────────────────────────

def make_speed_bars(all_results):
    """Grouped bar chart: NumPy vs Pandas per operation, one chart per size."""
    ops = ['sort', 'filter', 'sum', 'mean', 'std', 'fillna', 'groupby']
    size_labels = {100_000: '100k', 1_000_000: '1M', 10_000_000: '10M'}
    file_stems  = {100_000: '01_speed_bars_100k', 1_000_000: '02_speed_bars_1m', 10_000_000: '03_speed_bars_10m'}

    for size in SIZES:
        results = all_results[size]
        np_ms  = [results[op]['numpy']  * 1000 for op in ops]
        pd_ms  = [results[op]['pandas'] * 1000 for op in ops]

        x = np.arange(len(ops))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars_np = ax.bar(x - width / 2, np_ms, width, label='NumPy',  color='#4C72B0')
        bars_pd = ax.bar(x + width / 2, pd_ms, width, label='Pandas', color='#DD8452')

        ax.set_title(f'NumPy vs Pandas — Execution Time ({size_labels[size]} rows)', fontsize=13)
        ax.set_ylabel('Time (ms)')
        ax.set_xticks(x)
        ax.set_xticklabels(ops)
        ax.legend()
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f'{file_stems[size]}.png', dpi=150)
        plt.close(fig)
        print(f'  Saved {file_stems[size]}.png')


def make_speedup_heatmap(all_results):
    """Heatmap: operations × sizes, colour = speedup factor (NumPy / Pandas ratio)."""
    ops   = ['sort', 'filter', 'sum', 'mean', 'std', 'fillna', 'groupby']
    sizes = [100_000, 1_000_000, 10_000_000]
    size_labels = ['100k', '1M', '10M']

    # speedup > 1 means NumPy is faster; < 1 means Pandas is faster
    data = np.array([
        [all_results[s][op]['speedup'] for s in sizes]
        for op in ops
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        data,
        ax=ax,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',   # red = Pandas faster, green = NumPy faster
        center=1.0,
        xticklabels=size_labels,
        yticklabels=ops,
        linewidths=0.5,
        cbar_kws={'label': 'Speedup factor (NumPy ÷ Pandas time)'},
    )
    ax.set_title('NumPy Speedup over Pandas (>1 = NumPy faster)', fontsize=12)
    ax.set_xlabel('Dataset size')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '04_speedup_heatmap.png', dpi=150)
    plt.close(fig)
    print('  Saved 04_speedup_heatmap.png')


def make_scaling_lines(all_results):
    """Line chart: time vs dataset size per operation, one line each for NumPy/Pandas."""
    ops   = ['sort', 'filter', 'sum', 'mean', 'std', 'fillna', 'groupby']
    sizes = [100_000, 1_000_000, 10_000_000]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()

    for i, op in enumerate(ops):
        ax = axes[i]
        np_ms = [all_results[s][op]['numpy']  * 1000 for s in sizes]
        pd_ms = [all_results[s][op]['pandas'] * 1000 for s in sizes]

        ax.plot(sizes, np_ms, marker='o', label='NumPy',  color='#4C72B0')
        ax.plot(sizes, pd_ms, marker='s', label='Pandas', color='#DD8452')
        ax.set_title(op, fontsize=11)
        ax.set_ylabel('Time (ms)')
        ax.set_xscale('log')
        ax.set_xticks(sizes)
        ax.set_xticklabels(['100k', '1M', '10M'])
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    # hide the unused 8th subplot
    axes[-1].set_visible(False)

    fig.suptitle('Scaling: Execution Time vs Dataset Size', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '05_scaling_lines.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved 05_scaling_lines.png')


def make_memory_comparison(all_results):
    """Bar chart: memory usage of NumPy structured array vs Pandas DataFrame per size."""
    sizes = [100_000, 1_000_000, 10_000_000]
    size_labels = ['100k', '1M', '10M']

    # Each row: price(8) + quantity(4) + category(1) + score(4) = 17 bytes per row
    bytes_per_row = 17
    np_mb  = [s * bytes_per_row / 1_048_576 for s in sizes]

    # Pandas adds per-column overhead; empirically generate and measure
    pd_mb = []
    for s in sizes:
        df = generate_pandas(s)
        pd_mb.append(df.memory_usage(deep=True).sum() / 1_048_576)

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, np_mb, width, label='NumPy structured array', color='#4C72B0')
    ax.bar(x + width / 2, pd_mb, width, label='Pandas DataFrame',       color='#DD8452')
    ax.set_title('Memory Usage: NumPy vs Pandas', fontsize=13)
    ax.set_ylabel('Memory (MB)')
    ax.set_xticks(x)
    ax.set_xticklabels(size_labels)
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / '06_memory_comparison.png', dpi=150)
    plt.close(fig)
    print('  Saved 06_memory_comparison.png')


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = {}

    for size in SIZES:
        print(f'\nBenchmarking {size:,} rows...')
        all_results[size] = run_benchmarks(size)
        print_table(size, all_results[size])

    print('\nGenerating charts...')
    make_speed_bars(all_results)
    make_speedup_heatmap(all_results)
    make_scaling_lines(all_results)
    make_memory_comparison(all_results)

    # Serialise results — convert int keys to strings for JSON compliance
    serialisable = {str(k): v for k, v in all_results.items()}
    with open('results.json', 'w') as f:
        json.dump(serialisable, f, indent=2)
    print('\nResults saved to results.json')
    print('Done.')
