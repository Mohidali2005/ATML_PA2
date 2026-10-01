from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import sample_entropy, sampled_kl
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters
from task3_grpo.evaluate import evaluate_policy, load_evaluation_bundle
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

REWARD_MAX_LENGTH = 1280
INFORMATIVE_TOLERANCE = 1e-6

CURVE_PANELS = [
    ("reward","learned reward","steelblue"),
    ("kl","sampled kl from reference","darkorange"),
    ("group_reward_std","within group reward std","firebrick"),
    ("policy_loss","policy loss","seagreen"),
    ("entropy","sampled token entropy","steelblue"),
    ("grad_norm","policy gradient norm","darkorange"),
    ("response_length","response length","firebrick"),
    ("truncated_fraction","truncated fraction","seagreen"),
]


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def collect_group_rollout(bundle, prompts):
    """Sample K completions for every prompt and cache what an update needs

    The old policy and reference log probabilities are computed once here so
    every epoch compares against the same snapshot. Completions that hit the
    generation cap keep their reward in the group statistics but are masked
    out of the training loss when the config asks for it
    """
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    gen_cfg = cfg["generation"]
    k = int(cfg["num_generations"])

    grouped_prompts = [prompt for prompt in prompts for _ in range(k)]
    group_ids = torch.arange(len(prompts)).repeat_interleave(k)

    policy.config.use_cache = True
    gen = batch_generate(
        policy,bundle["tokenizer"],grouped_prompts,
        max_prompt_length=int(cfg["max_prompt_length"]),
        max_new_tokens=int(cfg["max_completion_length"]),
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
        reward = score_reward_pairs(
            bundle["reward_model"],bundle["reward_tokenizer"],grouped_prompts,gen["responses"],
            max_length=REWARD_MAX_LENGTH,
        ).to(old_logp.device)

    train_mask = gen["response_mask"]
    if cfg["mask_truncated_completions"]:
        train_mask = mask_truncated_sequences(train_mask,gen["truncated"])
    return {
        "sequences":sequences,
        "response_ids":response_ids,
        "attention_mask":attention_mask,
        "width":width,
        "response_mask":gen["response_mask"],
        "train_mask":train_mask,
        "old_logp":old_logp,
        "ref_logp":ref_logp,
        "reward":reward,
        "group_ids":group_ids.to(old_logp.device),
        "lengths":gen["response_lengths"],
        "truncated":gen["truncated"],
    }


def group_statistics(rewards, group_ids):
    """Return the mean within group reward std and the uninformative group fraction"""
    stds = torch.stack([rewards[group_ids == gid].std(unbiased=False) for gid in group_ids.unique()])
    return stds.mean().item(),(stds <= INFORMATIVE_TOLERANCE).float().mean().item()


def sequence_gradient_norms(bundle, batch, advantages, loss_type):
    """Measure the policy gradient norm each completion contributes on its own

    Every completion gets its own forward and backward pass with the kl term
    switched off so the norm only reflects how the normalization scales that
    completion. The policy weights are not changed by this measurement
    """
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    rows = []
    for i in range(len(batch["lengths"])):
        row = slice(i,i+1)
        new_logp,_ = response_token_logprobs(
            policy,batch["sequences"][row],batch["attention_mask"][row],batch["width"],batch["response_ids"][row],
        )
        loss,_ = grpo_policy_loss(
            new_logp,batch["old_logp"][row],advantages[row],batch["train_mask"][row],batch["ref_logp"][row],
            float(cfg["clip_epsilon"]),0.0,loss_type,int(cfg["max_completion_length"]),
        )
        bundle["optimizer"].zero_grad()
        loss.backward()
        squared = sum(p.grad.float().pow(2).sum().item() for p in trainable_parameters(policy) if p.grad is not None)
        rows.append({
            "response_length":batch["lengths"][i],
            "abs_advantage":advantages[i].abs().item(),
            "grad_norm":squared**0.5,
            "truncated":batch["truncated"][i],
        })
    bundle["optimizer"].zero_grad()
    return rows


def grpo_update(bundle, prompts, loss_type, track_gradients):
    """Run one rollout and the configured number of epochs on it

    Returns the per update diagnostics that get logged and, when asked, the
    per completion gradient norms measured before the policy moves
    """
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    max_grad_norm = float(cfg["max_grad_norm"])

    batch = collect_group_rollout(bundle,prompts)
    advantages = group_relative_advantages(batch["reward"],batch["group_ids"])
    group_std,uninformative = group_statistics(batch["reward"],batch["group_ids"])
    sequence_rows = sequence_gradient_norms(bundle,batch,advantages,loss_type) if track_gradients else []

    epoch_stats = {"policy_loss":[],"clip_fraction":[],"grad_norm":[]}
    for _ in range(int(cfg["policy_epochs"])):
        new_logp,_ = response_token_logprobs(policy,batch["sequences"],batch["attention_mask"],batch["width"],batch["response_ids"])
        loss,stats = grpo_policy_loss(
            new_logp,batch["old_logp"],advantages,batch["train_mask"],batch["ref_logp"],
            float(cfg["clip_epsilon"]),float(cfg["kl_beta"]),loss_type,int(cfg["max_completion_length"]),
        )
        bundle["optimizer"].zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy),max_grad_norm)
        bundle["optimizer"].step()
        epoch_stats["policy_loss"].append(stats["policy_term"].item())
        epoch_stats["clip_fraction"].append(stats["clip_fraction"].item())
        epoch_stats["grad_norm"].append(grad_norm.item())

    mask = batch["response_mask"]
    record = {key:float(np.mean(values)) for key,values in epoch_stats.items()}
    record.update({
        "reward":batch["reward"].mean().item(),
        "kl":sampled_kl(batch["old_logp"],batch["ref_logp"],mask).item(),
        "entropy":sample_entropy(batch["old_logp"],mask).item(),
        "response_length":float(np.mean(batch["lengths"])),
        "truncated_fraction":float(np.mean(batch["truncated"])),
        "group_reward_std":group_std,
        "uninformative_fraction":uninformative,
    })
    return record,sequence_rows


def plot_training_curve(history, run_name, figures_dir):
    """Plot every logged grpo diagnostic against the update index for one run"""
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


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", track_gradients: bool = False):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    # staying in train mode so gradient checkpointing keeps memory low for long completions
    policy.train()
    # switching dropout off so the ratio starts at exactly one on every update
    for module in policy.modules():
        if isinstance(module,torch.nn.Dropout):
            module.eval()

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    # one fixed prompt order so every run sees the same prompts in the same order
    rows = bundle["prompt_rows"]
    order = np.random.RandomState(int(cfg["seed"])).permutation(len(rows))
    per_update = int(cfg["prompts_per_update"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()

    history = []
    sequence_rows = []
    for update in range(int(cfg["updates"])):
        picked = [rows[i] for i in order[update*per_update:(update+1)*per_update]]
        record,per_sequence = grpo_update(bundle,[prompt_messages(row) for row in picked],loss_type,track_gradients)
        record["update"] = update
        record["prompt_id"] = picked[0].get("prompt_id")
        history.append(record)
        sequence_rows.extend({"update":update,"loss_type":loss_type,**row} for row in per_sequence)
        print(f"update {update} reward {record['reward']:.3f} kl {record['kl']:.4f} entropy {record['entropy']:.3f} "
              f"group std {record['group_reward_std']:.3f} length {record['response_length']:.0f}")

    wall_seconds = elapsed()
    peak_vram_gb = torch.cuda.max_memory_allocated()/1e9 if torch.cuda.is_available() else None

    policy.save_pretrained(out)
    print(f"saved adapter to {out}")

    table = pd.DataFrame(history)
    summary = {
        "run_name":run_name,
        "loss_type":loss_type,
        "updates":int(cfg["updates"]),
        "num_generations":int(cfg["num_generations"]),
        "wall_clock_seconds":wall_seconds,
        "peak_vram_gb":peak_vram_gb,
        "mean_group_reward_std":float(table["group_reward_std"].mean()),
        "uninformative_fraction":float(table["uninformative_fraction"].mean()),
        "mean_clip_fraction":float(table["clip_fraction"].mean()),
        "peak_grad_norm":float(table["grad_norm"].max()),
        "mean_grad_norm":float(table["grad_norm"].mean()),
    }
    save_json(tables_dir/f"{run_name}_training_curve.json",history)
    save_json(tables_dir/f"{run_name}_run_summary.json",summary)
    if sequence_rows:
        pd.DataFrame(sequence_rows).to_csv(tables_dir/f"{run_name}_sequence_gradients.csv",index=False)
    plot_training_curve(history,run_name,figures_dir)
    return summary


def run_fork(config_path, loss_type):
    """Train and evaluate one short grpo fork from the supplied midpoint

    Both normalization forks load the identical midpoint adapter and use the
    same prompt order and update budget. A finished fork is read back from
    disk so an interrupted comparison can resume
    """
    cfg = load_yaml(config_path)
    run_name = f"norm_{loss_type}"
    tables_dir = repo_path(cfg["results_dir"])/"tables"
    eval_path = tables_dir/f"{run_name}_eval.json"
    summary_path = tables_dir/f"{run_name}_run_summary.json"

    if not eval_path.exists():
        output_path = f"outputs/task3_grpo/{run_name}"
        if not summary_path.exists():
            run_grpo(config_path,output_path,updates=int(cfg["fork_updates"]),loss_type=loss_type,run_name=run_name,track_gradients=True)
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
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
