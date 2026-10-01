from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import run_fork
from task2_ppo.evaluate import evaluate_if_missing

BETA_COLORS = {0.0:"firebrick",0.1:"steelblue",0.2:"seagreen"}
EXAMPLES_PER_SIDE = 3


def paired_examples(midpoint_name, fork_name, tables_dir):
    """Pair each fork response with the midpoint response to the same prompt

    Sorting by the change in reward surfaces the prompts where the learned
    reward rose the most and the ones where it fell the most so the text
    can be read to judge whether reward and quality moved together
    """
    midpoint = pd.read_csv(tables_dir/f"{midpoint_name}_generations.csv")
    fork = pd.read_csv(tables_dir/f"{fork_name}_generations.csv")
    paired = midpoint.merge(fork,on="prompt_id",suffixes=("_midpoint","_fork"))
    paired["reward_change"] = paired["reward_score_fork"]-paired["reward_score_midpoint"]
    paired["length_change"] = paired["response_length_fork"]-paired["response_length_midpoint"]
    paired = paired.sort_values("reward_change",ascending=False)
    paired.to_csv(tables_dir/f"{fork_name}_vs_midpoint.csv",index=False)

    keep = ["prompt_id","prompt_midpoint","response_midpoint","response_fork","reward_score_midpoint","reward_score_fork","reward_change","response_length_midpoint","response_length_fork"]
    return {
        "largest_reward_gain":paired.head(EXAMPLES_PER_SIDE)[keep].to_dict("records"),
        "largest_reward_drop":paired.tail(EXAMPLES_PER_SIDE)[keep].to_dict("records"),
    }


def plot_kl_study(fork_rows, midpoint, figures_dir):
    """Plot held out reward drift entropy and length against the kl coefficient"""
    table = pd.DataFrame(fork_rows).sort_values("kl_beta")
    positions = np.arange(len(table))
    metrics = [
        ("mean_reward","held out reward score"),
        ("mean_kl","sampled kl from reference"),
        ("mean_entropy","sampled token entropy"),
        ("mean_response_length","mean response length"),
    ]

    fig,axes = plt.subplots(1,4,figsize=(16,4))
    for ax,(key,label) in zip(axes,metrics):
        ax.plot(positions,table[key],marker="o",color="steelblue")
        ax.axhline(midpoint[key],linestyle="--",color="darkorange")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(b) for b in table["kl_beta"]])
        ax.set_xlabel("kl coefficient")
        ax.set_ylabel(label)
        ax.set_title(label)

    handles = [
        plt.Line2D([0],[0],color="steelblue",marker="o",label="short fork"),
        plt.Line2D([0],[0],color="darkorange",linestyle="--",label="supplied midpoint"),
    ]
    fig.legend(handles=handles,loc="lower center",ncol=2,fontsize=9)
    fig.tight_layout(rect=(0,0.08,1,1))
    out_path = figures_dir/"kl_study.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_kl_trajectories(fork_rows, tables_dir, figures_dir):
    """Plot the per update training trajectories of each kl coefficient fork"""
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
            ax.plot(history["update"],history[key],marker="o",markersize=3,color=BETA_COLORS[float(row["kl_beta"])])
    for ax,(key,label) in zip(axes.flat,panels):
        ax.set_xlabel("update")
        ax.set_ylabel(label)
        ax.set_title(label)

    handles = [plt.Line2D([0],[0],color=color,marker="o",label=f"kl coefficient {beta}") for beta,color in BETA_COLORS.items()]
    fig.legend(handles=handles,loc="lower center",ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,0.06,1,1))
    out_path = figures_dir/"kl_trajectories.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    midpoint = evaluate_if_missing(args.config,cfg["paths"]["ppo_midpoint_policy"],"midpoint")
    fork_rows = [run_fork(args.config,cfg["clip_epsilon"],beta) for beta in cfg["kl_values"]]
    table = pd.DataFrame(fork_rows)
    table.to_csv(tables_dir/"kl_study.csv",index=False)
    print(table)

    plot_kl_study(fork_rows,midpoint,figures_dir)
    plot_kl_trajectories(fork_rows,tables_dir,figures_dir)

    for row in fork_rows:
        examples = paired_examples("midpoint",row["name"],tables_dir)
        save_json(tables_dir/f"{row['name']}_examples.json",examples)


if __name__ == "__main__":
    main()
