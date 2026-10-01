from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path

INFORMATIVE_TOLERANCE = 1e-6
WEAK_STD = 0.25
SHUFFLES = 200
BIN_NAMES = ["hard","medium","easy"]
BIN_COLORS = {"all":"black","hard":"firebrick","medium":"darkorange","easy":"seagreen"}


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng=None):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    groups = []
    for pid,rows in by_prompt.items():
        order = np.arange(len(rows)) if rng is None else rng.permutation(len(rows))
        for start in range(0,len(rows)-len(rows)%k,k):
            members = [rows[i] for i in order[start:start+k]]
            groups.append({
                "source_index":pid,
                "rewards":np.array([m["reward"] for m in members],dtype=np.float64),
                "clipped":np.array([m["clipped_at_max"] for m in members],dtype=bool),
            })
    return groups


def prompt_difficulty_bins(by_prompt):
    """Assign each prompt to a difficulty tercile by its mean cached reward

    The rule is applied once to all eight cached completions of a prompt so
    every group size sees the identical bins. A low mean reward means hard
    """
    means = {pid:float(np.mean([r["reward"] for r in rows])) for pid,rows in by_prompt.items()}
    ranked = sorted(means,key=means.get)
    size = len(ranked)/3
    return {pid:BIN_NAMES[min(int(i//size),2)] for i,pid in enumerate(ranked)}


def group_metrics(groups):
    """Summarize how much usable relative signal a list of groups carries"""
    stds = np.array([g["rewards"].std() for g in groups])
    centered = np.concatenate([g["rewards"]-g["rewards"].mean() for g in groups])
    informative = stds > INFORMATIVE_TOLERANCE
    usable = sum(int((~g["clipped"]).sum()) for g,ok in zip(groups,informative) if ok)
    total = sum(len(g["rewards"]) for g in groups)
    return {
        "n_groups":len(groups),
        "informative_rate":float(informative.mean()),
        "weak_rate":float((stds < WEAK_STD).mean()),
        "mean_group_std":float(stds.mean()),
        "signal_variance":float(centered.var()),
        "usable_fraction":usable/total,
    }


def summarize(by_prompt, bins, k):
    """Return metrics for one group size overall and inside each difficulty bin

    The fixed partition keeps each prompt's completions in generation order.
    The shuffled columns average the same metrics over many random partitions
    so one lucky split cannot drive the comparison
    """
    fixed = regroup_equal_generation_budget(by_prompt,k)
    rng = np.random.RandomState(6304)
    shuffled = [regroup_equal_generation_budget(by_prompt,k,rng) for _ in range(SHUFFLES)]
    rows = []
    for name in ["all"]+BIN_NAMES:
        pick = lambda groups: [g for g in groups if name == "all" or bins[g["source_index"]] == name]
        row = {"k":k,"difficulty_bin":name,**group_metrics(pick(fixed))}
        draws = pd.DataFrame([group_metrics(pick(groups)) for groups in shuffled])
        for key in ["informative_rate","weak_rate","mean_group_std","signal_variance"]:
            row[f"{key}_shuffled"] = float(draws[key].mean())
        rows.append(row)
    return rows


def plot_group_size_study(table, figures_dir):
    """Plot informativeness and signal strength against group size per difficulty bin"""
    panels = [
        ("mean_group_std_shuffled","mean within group reward std"),
        ("weak_rate_shuffled",f"fraction of groups with std below {WEAK_STD}"),
        ("signal_variance_shuffled","variance of the centered reward signal"),
    ]
    fig,axes = plt.subplots(1,3,figsize=(15,4.2))
    for ax,(key,label) in zip(axes,panels):
        for name,color in BIN_COLORS.items():
            part = table[table["difficulty_bin"] == name]
            ax.plot(part["k"],part[key],marker="o",color=color,linestyle="--" if name == "all" else "-")
        ax.set_xticks(sorted(table["k"].unique()))
        ax.set_xlabel("group size K")
        ax.set_ylabel(label)
        ax.set_title(label,fontsize=10)
    handles = [plt.Line2D([0],[0],color=color,marker="o",label=f"{name} prompts" if name != "all" else "all prompts") for name,color in BIN_COLORS.items()]
    fig.legend(handles=handles,loc="lower center",ncol=4,fontsize=9)
    fig.tight_layout(rect=(0,0.08,1,1))
    out_path = figures_dir/"group_size_study.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    bins = prompt_difficulty_bins(by_prompt)
    print("Prompts per difficulty bin:", {name:sum(1 for b in bins.values() if b == name) for name in BIN_NAMES})
    table = pd.DataFrame([row for k in cfg["group_sizes"] for row in summarize(by_prompt,bins,int(k))])
    table.to_csv(tables_dir/"group_size_study.csv",index=False)
    print(table.round(3).to_string())
    plot_group_size_study(table,figures_dir)


if __name__ == "__main__":
    main()
