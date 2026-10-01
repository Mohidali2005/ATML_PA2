from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.ablate_kl import paired_examples
from task3_grpo.continue_train import run_fork
from task3_grpo.evaluate import evaluate_if_missing

LOSS_COLORS = {"grpo":"steelblue","dr_grpo":"firebrick"}
LOSS_LABELS = {"grpo":"canonical grpo","dr_grpo":"dr grpo"}


def length_gradient_statistics(tables_dir, loss_types):
    """Measure how each normalization allocates gradient across short and long completions

    Completions masked for hitting the generation cap contribute no gradient
    and are left out. The gradient norm is divided by the absolute advantage
    so the numbers reflect the normalization alone. Short and long are split
    at the median length pooled over both forks
    """
    frames = {name:pd.read_csv(tables_dir/f"norm_{name}_sequence_gradients.csv") for name in loss_types}
    trained = {name:frame[~frame["truncated"]].copy() for name,frame in frames.items()}
    for frame in trained.values():
        frame["normalized_grad"] = frame["grad_norm"]/frame["abs_advantage"].clip(lower=1e-6)
    median = float(pd.concat(trained.values())["response_length"].median())

    stats = {}
    for name,frame in trained.items():
        informative = frame[frame["abs_advantage"] > 1e-6]
        short = informative[informative["response_length"] <= median]["normalized_grad"]
        long = informative[informative["response_length"] > median]["normalized_grad"]
        stats[name] = {
            "n_trained_completions":int(len(informative)),
            "length_median_split":median,
            "length_gradient_spearman":float(informative["response_length"].corr(informative["normalized_grad"],method="spearman")),
            "short_mean_normalized_grad":float(short.mean()),
            "long_mean_normalized_grad":float(long.mean()),
            "long_to_short_grad_ratio":float(long.mean()/short.mean()),
        }
    return stats,trained


def plot_norm_study(fork_rows, midpoint, figures_dir):
    """Plot held out reward drift length and truncation for both normalizations"""
    metrics = [
        ("mean_reward","held out reward score"),
        ("mean_kl","sampled kl from reference"),
        ("mean_response_length","mean response length"),
        ("truncation_rate","truncation rate"),
    ]
    fig,axes = plt.subplots(1,4,figsize=(16,4))
    for ax,(key,label) in zip(axes,metrics):
        for position,row in enumerate(fork_rows):
            ax.plot(position,row[key],marker="o",markersize=8,color=LOSS_COLORS[row["loss_type"]])
        ax.axhline(midpoint[key],linestyle="--",color="darkorange")
        ax.set_xticks(range(len(fork_rows)))
        ax.set_xticklabels([LOSS_LABELS[row["loss_type"]] for row in fork_rows])
        ax.set_xlim(-0.5,len(fork_rows)-0.5)
        ax.set_ylabel(label)
        ax.set_title(label)
    handles = [
        plt.Line2D([0],[0],color="steelblue",marker="o",linestyle="",label="canonical grpo fork"),
        plt.Line2D([0],[0],color="firebrick",marker="o",linestyle="",label="dr grpo fork"),
        plt.Line2D([0],[0],color="darkorange",linestyle="--",label="supplied midpoint"),
    ]
    fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,0.08,1,1))
    out_path = figures_dir/"norm_study.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_norm_trajectories(fork_rows, tables_dir, figures_dir):
    """Plot the per update training trajectories of both normalization forks"""
    panels = [
        ("reward","learned reward"),
        ("kl","sampled kl from reference"),
        ("entropy","sampled token entropy"),
        ("response_length","response length"),
    ]
    fig,axes = plt.subplots(2,2,figsize=(11,8))
    for row in fork_rows:
        history = pd.DataFrame(load_json(tables_dir/f"{row['name']}_training_curve.json"))
        for ax,(key,label) in zip(axes.flat,panels):
            ax.plot(history["update"],history[key],marker="o",markersize=3,color=LOSS_COLORS[row["loss_type"]])
    for ax,(key,label) in zip(axes.flat,panels):
        ax.set_xlabel("update")
        ax.set_ylabel(label)
        ax.set_title(label)
    handles = [plt.Line2D([0],[0],color=color,marker="o",label=LOSS_LABELS[name]) for name,color in LOSS_COLORS.items()]
    fig.legend(handles=handles,loc="lower center",ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,0.06,1,1))
    out_path = figures_dir/"norm_trajectories.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_gradient_allocation(trained, stats, figures_dir):
    """Plot each completion's normalized gradient norm against its length"""
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    for name,frame in trained.items():
        informative = frame[frame["abs_advantage"] > 1e-6]
        axes[0].scatter(informative["response_length"],informative["normalized_grad"],s=22,color=LOSS_COLORS[name],alpha=0.7)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("gradient norm per unit advantage")
    axes[0].set_title("per completion gradient norm")
    axes[0].set_xlabel("completion length in tokens")

    names = list(trained)
    short = [stats[n]["short_mean_normalized_grad"] for n in names]
    long = [stats[n]["long_mean_normalized_grad"] for n in names]
    positions = np.arange(len(names))
    axes[1].plot(positions,short,marker="o",markersize=8,color="seagreen",linestyle="")
    axes[1].plot(positions,long,marker="s",markersize=8,color="darkorange",linestyle="")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([LOSS_LABELS[n] for n in names])
    axes[1].set_xlim(-0.5,len(names)-0.5)
    axes[1].set_ylabel("mean gradient norm per unit advantage")
    axes[1].set_title("short versus long completions")
    axes[1].set_xlabel("normalization")

    handles = [
        plt.Line2D([0],[0],color="steelblue",marker="o",linestyle="",label="canonical grpo completions"),
        plt.Line2D([0],[0],color="firebrick",marker="o",linestyle="",label="dr grpo completions"),
        plt.Line2D([0],[0],color="seagreen",marker="o",linestyle="",label="short completions"),
        plt.Line2D([0],[0],color="darkorange",marker="s",linestyle="",label="long completions"),
    ]
    fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=9)
    fig.tight_layout(rect=(0,0.09,1,1))
    out_path = figures_dir/"norm_gradient_allocation.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    loss_types = ["grpo","dr_grpo"]
    midpoint = evaluate_if_missing(args.config,cfg["paths"]["grpo_midpoint_policy"],"midpoint")
    fork_rows = [run_fork(args.config,name) for name in loss_types]

    stats,trained = length_gradient_statistics(tables_dir,loss_types)
    for row in fork_rows:
        row.update(stats[row["loss_type"]])
    table = pd.DataFrame(fork_rows)
    table.to_csv(tables_dir/"norm_comparison.csv",index=False)
    print(table.T)

    plot_norm_study(fork_rows,midpoint,figures_dir)
    plot_norm_trajectories(fork_rows,tables_dir,figures_dir)
    plot_gradient_allocation(trained,stats,figures_dir)

    for row in fork_rows:
        examples = paired_examples("midpoint",row["name"],tables_dir)
        save_json(tables_dir/f"{row['name']}_examples.json",examples)


if __name__ == "__main__":
    main()
