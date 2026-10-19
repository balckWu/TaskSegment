"""Run Iris seeds sequentially on one GPU, with audited baseline settings."""
from pathlib import Path
import argparse
import os
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iris2d.protocol import default_data_root, default_run_root, load_recipe, verify_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA GPU; each child uses cuda:0")
    parser.add_argument("--seeds", type=int, nargs='+', default=None)
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--run-root", default=str(default_run_root()))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print all commands without training or writing results")
    args = parser.parse_args()
    verify_reference()
    recipe, common = load_recipe()
    seeds = common['seeds'] if args.seeds is None else args.seeds
    if args.gpu < 0 or not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(common['seeds']):
        parser.error("GPU must be nonnegative; seeds must be distinct members of the paper protocol")
    data = str(Path(args.data_root).resolve())
    run_root = Path(args.run_root).resolve()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PAPERLAB_PROGRESS'] = 'off'
    def execute(command):
        print(shlex.join([str(x) for x in command]), flush=True)
        if not args.dry_run:
            subprocess.run([str(x) for x in command], check=True, cwd=ROOT, env=env)
    execute([sys.executable, ROOT/'scripts/preflight_iris2d.py', '--data-root', data, '--check-data'])
    execute([sys.executable, ROOT/'scripts/check_iris2d.py'])
    metrics = []
    for seed in seeds:
        output = run_root/f'seed_{seed}'
        train = [sys.executable, ROOT/'scripts/train_iris2d.py', '--data-root', data,
            '--output', output, '--seed', seed, '--lr', recipe['lr'], '--min-lr', recipe['min_lr'],
            '--weight-decay', recipe['weight_decay'], '--warmup-epochs', common['warmup_epochs'],
            '--epochs', common['epochs'], '--patience', common['patience'],
            '--query-count', common['batch_size'], '--grad-accum-steps', common['grad_accum_steps'],
            '--episodes-per-epoch', common['episodes_per_epoch'], '--cache-dataset']
        if args.resume:
            train.append('--resume')
        execute(train)
        prediction = output/'predictions'
        metric_file = prediction/'metrics.json'
        if args.resume and metric_file.is_file():
            raise FileExistsError(f"Test results already exist: {prediction}; choose only unfinished seeds")
        execute([sys.executable, ROOT/'scripts/predict_iris2d.py', '--data-root', data,
            '--checkpoint', output/'checkpoint.pth', '--output', prediction, '--cache-dataset'])
        metrics.append(metric_file)
    if set(seeds) == set(common['seeds']):
        execute([sys.executable, ROOT/'scripts/summarize_iris2d.py', *metrics,
                 '--output', run_root/'summary_k4.json'])


if __name__ == '__main__':
    main()
