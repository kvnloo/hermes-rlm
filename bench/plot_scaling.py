"""Plot aggregate point estimates or task-cluster bootstrap from paper_matrix JSONL.

Requires matplotlib. No provider calls. Aggregate intervals are deliberately not
reconstructed. JSONL intervals resample task means, never individual repeats.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def protocol(row):
    if 'protocol' in row:
        return hashlib.sha256(json.dumps(row['protocol'], sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()
    return str(row.get('protocol_id', 'unspecified'))


def group(row, cohort):
    return (cohort, str(row.get('family', 'unspecified')),
            str(row.get('root_model', 'unspecified')),
            str(row.get('sub_model', 'unspecified')), protocol(row),
            str(row.get('metric', 'score')),
            str(row.get('length_tokenizer_sha256', 'dataset-nominal')),
            str(row.get('length_tokenizer_engine', 'unspecified')), str(row.get('arm', 'unspecified')))


def bootstrap(values, draws=2000, seed=0):
    if len(values) < 2:
        return None
    if draws < 2:
        raise ValueError('draws must be at least 2')
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    def quantile(p):
        i = (len(samples) - 1) * p
        lo = int(i)
        return samples[lo] + (samples[min(lo + 1, len(samples)-1)] - samples[lo]) * (i-lo)
    return [quantile(.025), quantile(.975)]


def points_from_aggregate(data, cohorts=('primary',)):
    if data.get('schema_version') != 'rlm-paper-aggregate.v1':
        raise ValueError('unsupported aggregate schema')
    points = []
    for cohort in cohorts:
        for row in data.get(cohort, []):
            size, score = row.get('size_tokens'), row.get('mean_score')
            if finite(size) and size > 0 and finite(score):
                points.append(dict(group=group(row, cohort), size=size, score=score,
                                   ci95=None, interval='aggregate point estimate only'))
    return points


def points_from_rows(rows, draws=2000, seed=0):
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        if row.get('valid') is not True:
            continue
        size, score, task = row.get('size_tokens'), row.get('score'), row.get('task_id')
        if not (finite(size) and size > 0 and finite(score) and isinstance(task, str) and task):
            continue
        key = (group(row, str(row.get('cohort', 'unspecified'))), size)
        # Duplicate physical rows must not silently reweight a task's repeats.
        if row.get('row_key') is not None:
            identity = (key, row['row_key'])
            if identity in seen:
                raise ValueError('duplicate row_key in compatible group')
            seen.add(identity)
        groups[key][task].append(score)
    points = []
    for (g, size), tasks in sorted(groups.items()):
        values = [statistics.mean(tasks[t]) for t in sorted(tasks)]
        points.append(dict(group=g, size=size, score=statistics.mean(values),
                           tasks=len(values), ci95=bootstrap(values, draws, seed),
                           interval='task-cluster percentile bootstrap 95%' if len(values)>1
                           else 'insufficient task clusters'))
    return points


def render(points, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = sorted({tuple(p['group'][:-1]) for p in points})
    if not panels:
        raise ValueError('no valid scored points to plot')
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.7*len(panels)), squeeze=False)
    for ax, panel in zip(axes[:, 0], panels):
        arms = sorted({p['group'][-1] for p in points if tuple(p['group'][:-1]) == panel})
        for arm in arms:
            series = sorted((p for p in points if tuple(p['group'][:-1]) == panel and p['group'][-1] == arm), key=lambda p:p['size'])
            line, = ax.plot([p['size'] for p in series], [p['score'] for p in series], 'o-', label=arm)
            for p in series:
                if p['ci95'] is not None:
                    ax.vlines(p['size'], *p['ci95'], color=line.get_color(), alpha=.6)
        cohort, family, root, sub, proto, metric, tokenizer, engine = panel
        ax.set_title(f'{cohort} | {family} | {root}\nsub={sub} | protocol={proto[:12]} | {metric}', fontsize=8)
        ax.set_xscale('log', base=2)
        basis = 'dataset-nominal tokens' if tokenizer == 'dataset-nominal' else (
            f'measured raw-context tokens, tokenizer {tokenizer[:12]} ({engine})')
        ax.set_xlabel(f'Input length ({basis}, log₂)')
        ax.set_ylabel('Score')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Scaling: valid scores only; missing/invalid are not zero\nIntervals only for JSONL task-cluster bootstrap; aggregate points have no intervals', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('aggregate', 'jsonl'), default='aggregate')
    parser.add_argument('--cohort', action='append', choices=('primary', 'pilot', 'smoke'),
                        help='aggregate cohort to show, repeatable; default primary')
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error('--draws must be at least 2')
    with args.input.open(encoding='utf-8') as stream:
        if args.format == 'aggregate':
            points = points_from_aggregate(json.load(stream), args.cohort or ('primary',))
        else:
            if args.cohort:
                parser.error('--cohort applies to aggregate summaries only')
            points = points_from_rows((json.loads(line) for line in stream if line.strip()),
                                      args.draws, args.seed)
    render(points, args.output)
    args.output.with_suffix('.points.json').write_text(json.dumps(points, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'{len(points)} scored points -> {args.output}')


if __name__ == '__main__':
    main()
