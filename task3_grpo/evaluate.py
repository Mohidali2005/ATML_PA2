from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer
from task2_ppo.evaluate import EVAL_PROMPTS, generate_and_score

REWARD_MAX_LENGTH = 1280


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_policy(bundle, name):
    """Run the common held out evaluation for one policy and save it

    Every policy is scored on the same prompts with the same decoding
    settings and the same seed as the other tasks so differences between
    policies are not caused by the evaluation itself
    """
    cfg = bundle["cfg"]
    set_seed(int(cfg["seed"]))
    rows = bundle["rows"][:EVAL_PROMPTS]
    eval_cfg = {**cfg,"eval_max_response_length":cfg["cache_generation_cap"],"reward_max_length":REWARD_MAX_LENGTH}

    mean_kl,mean_entropy,records = generate_and_score(bundle["policy"],bundle["tokenizer"],bundle["reward"],rows,eval_cfg)
    rewards = [r["reward_score"] for r in records]
    lengths = [r["response_length"] for r in records]

    result = {
        "name":name,
        "n_prompts":len(records),
        "mean_reward":float(np.mean(rewards)),
        "std_reward":float(np.std(rewards)),
        "mean_kl":mean_kl,
        "mean_entropy":mean_entropy,
        "mean_response_length":float(np.mean(lengths)),
        "std_response_length":float(np.std(lengths)),
        "truncation_rate":float(np.mean([r["truncated"] for r in records])),
        "reward_length_correlation":safe_corr(rewards,lengths),
    }
    print(result)

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    tables_dir.mkdir(parents=True,exist_ok=True)
    save_json(tables_dir/f"{name}_eval.json",result)
    pd.DataFrame(records).to_csv(tables_dir/f"{name}_generations.csv",index=False)
    return result


def evaluate_if_missing(config_path, adapter, name):
    """Evaluate an adapter unless its saved metrics already exist on disk"""
    cfg = load_yaml(config_path)
    eval_path = repo_path(cfg["results_dir"])/"tables"/f"{name}_eval.json"
    if eval_path.exists():
        return load_json(eval_path)
    bundle = load_evaluation_bundle(config_path,adapter)
    result = evaluate_policy(bundle,name)
    del bundle
    clear_gpu()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    evaluate_policy(bundle, args.name)


if __name__ == "__main__":
    main()
