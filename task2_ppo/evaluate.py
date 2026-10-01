from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import safe_corr
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode

EVAL_PROMPTS = 64
GENERATION_BATCH_SIZE = 2


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def generate_and_score(policy, tokenizer, reward_bundle, rows, cfg):
    """Generate one response per held out prompt and score every response

    Returns the token weighted sampled kl from the reference policy and
    the sampled token entropy plus a per prompt record holding the reward
    and length so paired comparisons across policies stay possible
    """
    reward_model,reward_tokenizer = reward_bundle
    gen_cfg = cfg["generation"]
    kl_total = 0.0
    entropy_total = 0.0
    token_total = 0.0
    records = []

    for start in range(0,len(rows),GENERATION_BATCH_SIZE):
        chunk = rows[start:start+GENERATION_BATCH_SIZE]
        prompts = [prompt_messages(row) for row in chunk]
        gen = batch_generate(
            policy,tokenizer,prompts,
            max_prompt_length=int(cfg["max_prompt_length"]),
            max_new_tokens=int(cfg["eval_max_response_length"]),
            temperature=gen_cfg["temperature"],top_p=gen_cfg["top_p"],do_sample=gen_cfg["do_sample"],
        )

        with torch.no_grad():
            policy_logp,_ = response_token_logprobs(policy,gen["sequences"],gen["attention_mask"],gen["prompt_width"],gen["response_ids"])
            with reference_mode(policy):
                ref_logp,_ = response_token_logprobs(policy,gen["sequences"],gen["attention_mask"],gen["prompt_width"],gen["response_ids"])

        mask = gen["response_mask"]
        tokens = mask.sum(-1).clamp_min(1.0)
        kl_per_prompt = (((policy_logp-ref_logp)*mask).sum(-1)/tokens).tolist()
        entropy_per_prompt = (-(policy_logp*mask).sum(-1)/tokens).tolist()
        kl_total += ((policy_logp-ref_logp)*mask).sum().item()
        entropy_total += (-(policy_logp*mask)).sum().item()
        token_total += mask.sum().item()

        rewards = score_reward_pairs(
            reward_model,reward_tokenizer,prompts,gen["responses"],
            max_length=int(cfg["reward_max_length"]),
        ).tolist()

        for i,row in enumerate(chunk):
            records.append({
                "prompt_id":row.get("prompt_id"),
                "prompt":prompts[i][-1]["content"],
                "response":gen["responses"][i],
                "reward_score":rewards[i],
                "response_length":gen["response_lengths"][i],
                "truncated":gen["truncated"][i],
                "kl":kl_per_prompt[i],
                "entropy":entropy_per_prompt[i],
            })

    return kl_total/token_total,entropy_total/token_total,records


def evaluate_policy(bundle, name):
    """Run the common held out evaluation for one policy and save it

    Every policy is scored on the same prompts with the same decoding
    settings and the same seed so differences between policies are not
    caused by the evaluation itself. Saves the aggregate metrics and the
    per prompt generations to disk
    """
    cfg = bundle["cfg"]
    set_seed(int(cfg["seed"]))
    rows = bundle["rows"][:EVAL_PROMPTS]

    mean_kl,mean_entropy,records = generate_and_score(bundle["policy"],bundle["tokenizer"],bundle["reward"],rows,cfg)
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
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    evaluate_policy(bundle, args.name)


if __name__ == "__main__":
    main()
