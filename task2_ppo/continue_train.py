from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import masked_mean, sample_entropy, sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.evaluate import evaluate_policy, load_evaluation_bundle
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss

CURVE_PANELS = [
    ("reward","learned reward","steelblue"),
    ("kl","sampled kl from reference","darkorange"),
    ("policy_loss","policy loss","firebrick"),
    ("value_loss","value loss","seagreen"),
    ("entropy","sampled token entropy","steelblue"),
    ("clip_fraction","clip fraction","darkorange"),
    ("policy_grad_norm","policy gradient norm","firebrick"),
    ("response_length","response length","seagreen"),
]


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def critic_autocast():
    """Return a half precision autocast context on gpu and a plain context on cpu"""
    if torch.cuda.is_available():
        return torch.autocast("cuda",dtype=torch.float16)
    return nullcontext()


def response_values(value_model, sequences, attention_mask, prompt_width, steps):
    """Return the critic value of the state just before each response token"""
    with critic_autocast():
        all_values = token_values(value_model,sequences,attention_mask)
    return all_values[:,prompt_width-1:prompt_width-1+steps].float()


def explained_variance(values, returns, mask):
    """Measure how much of the return variance the critic explains on valid tokens

    A score of one means perfect prediction and zero means no better than
    the mean return while negative scores mean the critic is worse than that
    """
    valid = mask.bool()
    target = returns[valid]
    residual = target-values[valid]
    return 1.0-residual.pow(2).mean().item()/max(target.var(unbiased=False).item(),1e-8)


def collect_rollout(bundle, prompts):
    """Sample responses from the current policy and cache what an update needs

    The old policy and reference log probabilities and the critic values are
    computed once here so every ppo epoch compares against the same snapshot
    """
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    gen_cfg = cfg["generation"]

    policy.config.use_cache = True
    gen = batch_generate(
        policy,bundle["tokenizer"],prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_response_length"]),
        temperature=gen_cfg["temperature"],top_p=gen_cfg["top_p"],do_sample=gen_cfg["do_sample"],
    )
    policy.config.use_cache = False

    # cloning because tensors made under inference mode cannot be saved for backward
    sequences = gen["sequences"].clone()
    response_ids = gen["response_ids"].clone()
    attention_mask = gen["attention_mask"]
    width = gen["prompt_width"]

    with torch.no_grad():
        old_logp,_ = response_token_logprobs(policy,sequences,attention_mask,width,response_ids)
        with reference_mode(policy):
            ref_logp,_ = response_token_logprobs(policy,sequences,attention_mask,width,response_ids)
        values = response_values(bundle["value_model"],sequences,attention_mask,width,response_ids.shape[1])
        reward = score_reward_pairs(
            bundle["reward_model"],bundle["reward_tokenizer"],prompts,gen["responses"],
            max_length=int(cfg["reward_max_length"]),
        ).to(old_logp.device)

    missing_eos = torch.tensor([0.0 if done else 1.0 for done in gen["terminated_with_eos"]],device=reward.device)
    return {
        "sequences":sequences,
        "response_ids":response_ids,
        "attention_mask":attention_mask,
        "width":width,
        "mask":gen["response_mask"],
        "old_logp":old_logp,
        "ref_logp":ref_logp,
        "values":values,
        "reward":reward,
        "effective_reward":reward-float(cfg["missing_eos_penalty"])*missing_eos,
        "lengths":gen["response_lengths"],
        "missing_eos":missing_eos,
    }


def ppo_update(bundle, prompts):
    """Run one rollout and the configured number of ppo epochs on it

    Returns the per update diagnostics that get logged. The critic is
    trained on the same returns the advantages came from
    """
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    max_grad_norm = float(cfg["max_grad_norm"])

    batch = collect_rollout(bundle,prompts)
    mask = batch["mask"]
    rewards = shaped_rewards(batch["effective_reward"],batch["old_logp"],batch["ref_logp"],mask,float(cfg["kl_beta"]))
    advantages,returns = compute_gae(rewards,batch["values"],mask,gamma=float(cfg["gamma"]),lam=float(cfg["gae_lambda"]))
    advantages = normalize_advantages(advantages,mask)
    critic_ev = explained_variance(batch["values"],returns,mask)

    epoch_stats = {"policy_loss":[],"value_loss":[],"clip_fraction":[],"policy_grad_norm":[],"value_grad_norm":[]}
    for _ in range(int(cfg["ppo_epochs"])):
        new_logp,_ = response_token_logprobs(policy,batch["sequences"],batch["attention_mask"],batch["width"],batch["response_ids"])
        policy_loss,_,clip_fraction = ppo_policy_loss(new_logp,batch["old_logp"],advantages,mask,float(cfg["clip_epsilon"]))
        bundle["policy_optimizer"].zero_grad()
        policy_loss.backward()
        policy_grad = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy),max_grad_norm)
        bundle["policy_optimizer"].step()

        new_values = response_values(value_model,batch["sequences"],batch["attention_mask"],batch["width"],batch["response_ids"].shape[1])
        value_loss = value_mse_loss(new_values,returns,mask)
        bundle["value_optimizer"].zero_grad()
        (float(cfg["value_coef"])*value_loss).backward()
        value_grad = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model),max_grad_norm)
        bundle["value_optimizer"].step()

        epoch_stats["policy_loss"].append(policy_loss.item())
        epoch_stats["value_loss"].append(value_loss.item())
        epoch_stats["clip_fraction"].append(clip_fraction.item())
        epoch_stats["policy_grad_norm"].append(policy_grad.item())
        epoch_stats["value_grad_norm"].append(value_grad.item())

    # measuring how far the whole update moved the policy on the sampled tokens
    with torch.no_grad():
        post_logp,_ = response_token_logprobs(policy,batch["sequences"],batch["attention_mask"],batch["width"],batch["response_ids"])
    ratio_deviation = masked_mean((post_logp-batch["old_logp"]).abs(),mask).item()

    record = {key:float(np.mean(values)) for key,values in epoch_stats.items()}
    record.update({
        "reward":batch["reward"].mean().item(),
        "effective_reward":batch["effective_reward"].mean().item(),
        "kl":sampled_kl(batch["old_logp"],batch["ref_logp"],mask).item(),
        "entropy":sample_entropy(batch["old_logp"],mask).item(),
        "response_length":float(np.mean(batch["lengths"])),
        "truncated_fraction":batch["missing_eos"].mean().item(),
        "critic_explained_variance":critic_ev,
        "post_update_ratio_deviation":ratio_deviation,
    })
    return record


def plot_training_curve(history, run_name, figures_dir):
    """Plot every logged ppo diagnostic against the update index for one run"""
    table = pd.DataFrame(history)
    fig,axes = plt.subplots(2,4,figsize=(16,7))
    for ax,(key,label,color) in zip(axes.flat,CURVE_PANELS):
        ax.plot(table["update"],table[key],marker="o",markersize=3,color=color)
        ax.set_xlabel("update")
        ax.set_ylabel(label)
        ax.set_title(label)
    fig.tight_layout()
    out_path = figures_dir/f"{run_name}_training_curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard"):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    value_model = bundle["value_model"]

    # keeping dropout off so the ratio starts at exactly one on every update
    policy.eval()
    value_model.eval()
    # keeping the critic parameters in fp32 so adam does not underflow
    for param in trainable_parameters(value_model):
        param.data = param.data.float()

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    # one fixed prompt order so every fork sees the same prompts in the same order
    rows = bundle["prompt_rows"]
    order = np.random.RandomState(int(cfg["seed"])).permutation(len(rows))
    per_update = int(cfg["prompts_per_update"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()

    history = []
    for update in range(int(cfg["updates"])):
        picked = [rows[i] for i in order[update*per_update:(update+1)*per_update]]
        record = ppo_update(bundle,[prompt_messages(row) for row in picked])
        record["update"] = update
        record["prompt_id"] = picked[0].get("prompt_id")
        history.append(record)
        print(f"update {update} reward {record['reward']:.3f} kl {record['kl']:.4f} entropy {record['entropy']:.3f} "
              f"clip {record['clip_fraction']:.3f} length {record['response_length']:.0f}")

    wall_seconds = elapsed()
    peak_vram_gb = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None

    policy.save_pretrained(out)
    print(f"saved adapter to {out}")

    table = pd.DataFrame(history)
    summary = {
        "run_name":run_name,
        "updates":int(cfg["updates"]),
        "clip_epsilon":float(cfg["clip_epsilon"]),
        "kl_beta":float(cfg["kl_beta"]),
        "wall_clock_seconds":wall_seconds,
        "peak_vram_gb":peak_vram_gb,
        "mean_clip_fraction":float(table["clip_fraction"].mean()),
        "peak_policy_grad_norm":float(table["policy_grad_norm"].max()),
        "mean_post_update_ratio_deviation":float(table["post_update_ratio_deviation"].mean()),
        "max_post_update_ratio_deviation":float(table["post_update_ratio_deviation"].max()),
        "mean_critic_explained_variance":float(table["critic_explained_variance"].mean()),
    }
    save_json(tables_dir/f"{run_name}_training_curve.json",history)
    save_json(tables_dir/f"{run_name}_run_summary.json",summary)
    plot_training_curve(history,run_name,figures_dir)
    return summary


def run_fork(config_path, clip_epsilon, kl_beta):
    """Train and evaluate one short ppo fork from the supplied midpoint

    Every fork loads the identical policy and critic checkpoints and uses
    the same prompt order and update budget. A finished fork is read back
    from disk so the shared centre fork is only ever trained once and an
    interrupted sweep can resume
    """
    cfg = load_yaml(config_path)
    run_name = f"eps_{float(clip_epsilon)}_kl_{float(kl_beta)}"
    tables_dir = repo_path(cfg["results_dir"])/"tables"
    eval_path = tables_dir/f"{run_name}_eval.json"
    summary_path = tables_dir/f"{run_name}_run_summary.json"

    if not eval_path.exists():
        output_path = f"outputs/task2_ppo/{run_name}"
        if not summary_path.exists():
            run_ppo(config_path,output_path,updates=int(cfg["fork_updates"]),clip_epsilon=clip_epsilon,kl_beta=kl_beta,run_name=run_name)
            clear_gpu()
        bundle = load_evaluation_bundle(config_path,output_path)
        evaluate_policy(bundle,run_name)
        del bundle
        clear_gpu()

    result = load_json(eval_path)
    result.update({key:value for key,value in load_json(summary_path).items() if key not in result})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
