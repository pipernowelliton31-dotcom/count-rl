"""Evaluate a saved Student LoRA on fixed plain-prompt ODE cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.cases import build_eval_cases
from .core.config import load_experiment_config
from .core.io_utils import atomic_json, now_stamp
from .core.reward import RewardEvaluator
from .core.runtime import Flux2Runtime
from .evaluation.exact_count import evaluate_plain_ode
from .trainers.common import build_evaluator

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Optional evaluation subset for smoke tests')
    parser.add_argument('--tag', default='eval_lora')
    args = parser.parse_args(argv)
    cfg = load_experiment_config(path=args.config)
    cases = build_eval_cases(cfg)
    if args.limit:
        cases = cases[:args.limit]
    runtime = Flux2Runtime(cfg)
    runtime.precompute_prompts(list(dict.fromkeys([''] + [c.plain_prompt for c in cases])))
    runtime.lora.load(args.lora)
    runtime.snapshot_old()
    evaluator = build_evaluator(cfg)
    root = Path(cfg['experiment']['output_root']) / str(cfg['method']) / ('eval_' + now_stamp().replace(':', ''))
    root.mkdir(parents=True, exist_ok=True)
    summary = evaluate_plain_ode(runtime, evaluator, cases, root, tag=args.tag)
    atomic_json(root / 'evaluation_manifest.json', {'config': cfg['_config_path'], 'lora': str(Path(args.lora).resolve()), 'n_cases': len(cases), 'summary': {k:v for k,v in summary.items() if k != 'rows'}})
    print(json.dumps({'output': str(root), 'n': summary['n'], 'exact_accuracy': summary['exact_accuracy'], 'mae': summary['mae'], 'by_target': summary['by_target']}, indent=2), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
