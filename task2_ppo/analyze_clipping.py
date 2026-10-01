from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.logging_utils import load_json, save_json, set_seed
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer, trainable_parameters
from task2_ppo.continue_train import run_fork
from task2_ppo.evaluate import evaluate_if_missing
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards

INNER_STEPS = 10
EPS_COLORS = {0.05:"firebrick",0.2:"steelblue",0.5:"seagreen"}


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def reconstruct_cached_batch(rows, cfg, tokenizer):
    """Rebuild the token level tensors of the cached rollout batch

    The cache stores the response text and the per token log probabilities
    but not the token ids so each prompt is rendered and each response is
    tokenized again and checked against the stored token count. The shaped
    rewards and normalized advantages are computed once over the whole
    padded batch exactly as a live update would compute them
    """
    eval_rows = {row["prompt_id"]:row for row in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    steps = max(int(row["response_tokens"]) for row in rows)
    count = len(rows)
    old_logp = torch.zeros(count,steps)
    ref_logp = torch.zeros(count,steps)
    values = torch.zeros(count,steps)
    mask = torch.zeros(count,steps)
    task_reward = torch.zeros(count)
    prompt_ids = []
    response_ids = []

    for i,row in enumerate(rows):
        n = int(row["response_tokens"])
        old_logp[i,:n] = row["old_logprobs"]
        ref_logp[i,:n] = row["ref_logprobs"]
        values[i,:n] = row["values"]
        mask[i,:n] = 1.0
        task_reward[i] = float(row["effective_terminal_reward"])

        rendered = tokenizer.apply_chat_template(prompt_messages(eval_rows[row["prompt_id"]]),tokenize=False,add_generation_prompt=True)
        prompt_ids.append(tokenizer(rendered,truncation=True,max_length=int(cfg["max_prompt_length"]))["input_ids"])
        ids = tokenizer(row["response"],add_special_tokens=False)["input_ids"]
        if row["terminated_with_eos"]:
            ids = ids+[tokenizer.eos_token_id]
        if len(ids) != n:
            raise ValueError(f"cached response {i} retokenized to {len(ids)} tokens but the cache stores {n}")
        response_ids.append(ids)

    rewards = shaped_rewards(task_reward,old_logp,ref_logp,mask,float(cfg["kl_beta"]))
    advantages,_ = compute_gae(rewards,values,mask,gamma=float(cfg["gamma"]),lam=float(cfg["gae_lambda"]))
    advantages = normalize_advantages(advantages,mask)

    items = []
    for i in range(count):
        n = int(rows[i]["response_tokens"])
        input_ids = torch.tensor([prompt_ids[i]+response_ids[i]])
        items.append({
            "input_ids":input_ids,
            "attention_mask":torch.ones_like(input_ids),
            "width":len(prompt_ids[i]),
            "response_ids":torch.tensor([response_ids[i]]),
            "old_logp":old_logp[i:i+1,:n],
            "advantages":advantages[i:i+1,:n],
            "mask":mask[i:i+1,:n],
        })
    batch = {"old_logp":old_logp,"ref_logp":ref_logp,"advantages":advantages,"mask":mask}
    return batch,items


def drift_stress_test(batch, epsilons):
    """Measure clipping when an update repeats the drift the midpoint already made

    The ratio between the rollout policy and the reference policy is what
    one more update would produce if it moved as far again. It needs no
    model and shows how much of the batch each clip range would touch
    """
    rows = []
    for eps in epsilons:
        loss,ratio,clip_fraction = ppo_policy_loss(batch["old_logp"],batch["ref_logp"],batch["advantages"],batch["mask"],eps)
        rows.append({
            "epsilon":eps,
            "clip_fraction":clip_fraction.item(),
            "clipped_surrogate":-loss.item(),
            "unclipped_surrogate":masked_mean(ratio*batch["advantages"],batch["mask"]).item(),
            "ratio_deviation":masked_mean((ratio-1.0).abs(),batch["mask"]).item(),
        })
    return rows


def pass_over_batch(policy, items, eps, backward):
    """Score every cached sequence at the current weights and optionally take its gradient

    Each sequence is run alone just as it was during rollout and weighted by
    its token count so the accumulated gradient is the gradient of the mean
    over all tokens in the batch
    """
    device = next(policy.parameters()).device
    total = float(sum(item["old_logp"].shape[1] for item in items))
    sums = {"clip_fraction":0.0,"upper_fraction":0.0,"lower_fraction":0.0,"clipped_surrogate":0.0,"unclipped_surrogate":0.0,"ratio_deviation":0.0}

    for item in items:
        n = item["old_logp"].shape[1]
        old_logp = item["old_logp"].to(device)
        advantages = item["advantages"].to(device)
        mask = item["mask"].to(device)
        with torch.set_grad_enabled(backward):
            new_logp,_ = response_token_logprobs(policy,item["input_ids"].to(device),item["attention_mask"].to(device),item["width"],item["response_ids"].to(device))
            loss,ratio,clip_fraction = ppo_policy_loss(new_logp,old_logp,advantages,mask,eps)
        if backward:
            (loss*n/total).backward()
        sums["clip_fraction"] += clip_fraction.item()*n
        sums["upper_fraction"] += masked_mean((ratio > 1.0+eps).float(),mask).item()*n
        sums["lower_fraction"] += masked_mean((ratio < 1.0-eps).float(),mask).item()*n
        sums["clipped_surrogate"] += -loss.item()*n
        sums["unclipped_surrogate"] += masked_mean(ratio*advantages,mask).item()*n
        sums["ratio_deviation"] += masked_mean((ratio-1.0).abs(),mask).item()*n
    return {key:value/total for key,value in sums.items()}


def clip_trajectory(policy, snapshot, items, eps, cfg):
    """Take several optimizer steps on the cached batch at one clip range

    The adapter is reset to the supplied midpoint weights first so every
    clip range starts from the same policy. Step zero is measured before
    any update and so shows the numerical noise floor of the ratio
    """
    with torch.no_grad():
        for name,param in policy.named_parameters():
            if name in snapshot:
                param.copy_(snapshot[name])

    optimizer = AdamW(trainable_parameters(policy),lr=float(cfg["policy_learning_rate"]))
    curve = []
    for step in range(INNER_STEPS+1):
        update = step < INNER_STEPS
        optimizer.zero_grad()
        stats = pass_over_batch(policy,items,eps,update)
        stats["step"] = step
        stats["epsilon"] = eps
        curve.append(stats)
        if update:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(policy),float(cfg["max_grad_norm"]))
            optimizer.step()
        print(f"epsilon {eps} step {step} clip_fraction {stats['clip_fraction']:.4f} ratio_deviation {stats['ratio_deviation']:.4f}")
    return curve


def plot_cached_batch(curves, stress, figures_dir):
    """Plot the live clipping geometry of the cached batch and the drift stress test"""
    table = pd.DataFrame(curves)
    fig,axes = plt.subplots(2,2,figsize=(11,8))

    for eps,group in table.groupby("epsilon"):
        color = EPS_COLORS[float(eps)]
        axes[0,0].plot(group["step"],group["clip_fraction"],marker="o",color=color)
        axes[0,1].plot(group["step"],group["clipped_surrogate"],marker="o",color=color)
        axes[0,1].plot(group["step"],group["unclipped_surrogate"],linestyle="--",color=color)
        axes[1,0].plot(group["step"],group["ratio_deviation"],marker="o",color=color)

    axes[0,0].set_ylabel("affected token fraction")
    axes[0,0].set_title("tokens outside the clip range")
    axes[0,1].set_ylabel("surrogate value")
    axes[0,1].set_title("clipped and unclipped surrogate")
    axes[1,0].set_ylabel("mean absolute ratio deviation")
    axes[1,0].set_title("policy movement on the batch")
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.set_xlabel("optimizer step on the cached batch")

    stress_table = pd.DataFrame(stress)
    positions = np.arange(len(stress_table))
    axes[1,1].bar(positions,stress_table["clip_fraction"],color=[EPS_COLORS[float(e)] for e in stress_table["epsilon"]])
    axes[1,1].set_xticks(positions)
    axes[1,1].set_xticklabels([str(e) for e in stress_table["epsilon"]])
    axes[1,1].set_xlabel("clip epsilon")
    axes[1,1].set_ylabel("affected token fraction")
    axes[1,1].set_title("drift stress test")

    handles = [plt.Line2D([0],[0],color=color,marker="o",label=f"epsilon {eps}") for eps,color in EPS_COLORS.items()]
    handles.append(plt.Line2D([0],[0],color="dimgray",label="clipped surrogate"))
    handles.append(plt.Line2D([0],[0],color="dimgray",linestyle="--",label="unclipped surrogate"))
    fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=9)
    fig.tight_layout(rect=(0,0.06,1,1))
    out_path = figures_dir/"clipping_cached_batch.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_fork_results(fork_rows, midpoint, figures_dir):
    """Plot held out behavior and update stability of the three clip range forks"""
    table = pd.DataFrame(fork_rows).sort_values("clip_epsilon")
    positions = np.arange(len(table))
    metrics = [
        ("mean_reward","held out reward score",True),
        ("mean_kl","sampled kl from reference",True),
        ("mean_response_length","mean response length",True),
        ("mean_post_update_ratio_deviation","mean ratio deviation per update",False),
    ]

    fig,axes = plt.subplots(1,4,figsize=(16,4))
    for ax,(key,label,has_midpoint) in zip(axes,metrics):
        ax.plot(positions,table[key],marker="o",color="steelblue")
        if has_midpoint:
            ax.axhline(midpoint[key],linestyle="--",color="darkorange")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(e) for e in table["clip_epsilon"]])
        ax.set_xlabel("clip epsilon")
        ax.set_ylabel(label)
        ax.set_title(label)

    handles = [
        plt.Line2D([0],[0],color="steelblue",marker="o",label="short fork"),
        plt.Line2D([0],[0],color="darkorange",linestyle="--",label="supplied midpoint"),
    ]
    fig.legend(handles=handles,loc="lower center",ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,0.08,1,1))
    out_path = figures_dir/"clipping_forks.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_fork_trajectories(fork_rows, tables_dir, figures_dir):
    """Plot the per update clip fraction movement and gradient norm of each fork"""
    panels = [
        ("clip_fraction","clip fraction"),
        ("post_update_ratio_deviation","ratio deviation after the update"),
        ("policy_grad_norm","policy gradient norm"),
    ]
    fig,axes = plt.subplots(1,3,figsize=(13,4))
    for row in fork_rows:
        history = pd.DataFrame(load_json(tables_dir/f"{row['name']}_training_curve.json"))
        for ax,(key,label) in zip(axes,panels):
            ax.plot(history["update"],history[key],marker="o",markersize=3,color=EPS_COLORS[float(row["clip_epsilon"])])
    for ax,(key,label) in zip(axes,panels):
        ax.set_xlabel("update")
        ax.set_ylabel(label)
        ax.set_title(label)

    handles = [plt.Line2D([0],[0],color=color,marker="o",label=f"epsilon {eps}") for eps,color in EPS_COLORS.items()]
    fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,0.08,1,1))
    out_path = figures_dir/"clipping_fork_trajectories.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))

    set_seed(int(cfg["seed"]))
    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)
    epsilons = [float(e) for e in cfg["clip_values"]]

    batch,items = reconstruct_cached_batch(rows,cfg,load_tokenizer(cfg["base_model"]))
    stress = drift_stress_test(batch,epsilons)

    policy = load_policy(cfg,adapter_path=cfg["paths"]["ppo_midpoint_policy"],trainable=True)
    policy.eval()
    snapshot = {name:param.detach().clone() for name,param in policy.named_parameters() if param.requires_grad}
    curves = []
    for eps in epsilons:
        curves.extend(clip_trajectory(policy,snapshot,items,eps,cfg))
    del policy
    clear_gpu()

    pd.DataFrame(curves).to_csv(tables_dir/"clipping_cached_batch.csv",index=False)
    pd.DataFrame(stress).to_csv(tables_dir/"clipping_drift_stress.csv",index=False)
    save_json(tables_dir/"clipping_cached_batch_setup.json",{
        "n_sequences":len(items),
        "n_tokens":int(batch["mask"].sum().item()),
        "inner_steps":INNER_STEPS,
        "learning_rate":float(cfg["policy_learning_rate"]),
        "kl_beta":float(cfg["kl_beta"]),
    })
    plot_cached_batch(curves,stress,figures_dir)

    midpoint = evaluate_if_missing(args.config,cfg["paths"]["ppo_midpoint_policy"],"midpoint")
    fork_rows = [run_fork(args.config,eps,cfg["kl_beta"]) for eps in epsilons]
    table = pd.DataFrame(fork_rows)
    table.to_csv(tables_dir/"clipping_study.csv",index=False)
    print(table)

    plot_fork_results(fork_rows,midpoint,figures_dir)
    plot_fork_trajectories(fork_rows,tables_dir,figures_dir)


if __name__ == "__main__":
    main()
