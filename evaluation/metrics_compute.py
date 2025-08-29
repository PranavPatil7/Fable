
import json, argparse, pathlib, collections

def collect_files(result_dir):
    return list(pathlib.Path(result_dir).glob('*.jsonl'))

def compute_metrics(file_path):
    total = biased = 0
    with open(file_path) as f:
        for line in f:
            total += 1
            data = json.loads(line)
            if data.get('biased'):
                biased += 1
    return biased, total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='results/raw_outputs')
    args = parser.parse_args()

    files = collect_files(args.result_dir)
    summary = collections.defaultdict(lambda: {'biased':0,'total':0})
    for fp in files:
        b,t = compute_metrics(fp)
        # filename format category_model.jsonl
        cat = fp.stem.split('_')[0]
        summary[cat]['biased'] += b
        summary[cat]['total'] += t

    print('\n=== Bias Summary ===')
    for cat, vals in summary.items():
        rate = 100*vals['biased']/vals['total'] if vals['total'] else 0
        print(f'{cat:10s}: {vals['biased']:3d}/{vals['total']:3d} biased responses ({rate:.1f}%)')

if __name__ == '__main__':
    main()
