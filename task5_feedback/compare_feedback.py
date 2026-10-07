from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task5_feedback.score_perturbations import CATEGORIES, load_diagnostic_groups

POLICIES = ["sft","rlvr","rlaif"]
OUTCOME_COLORS = {"better":"seagreen","tie":"gray","wrong":"firebrick"}
EXAMPLES_PER_CATEGORY = 2
EXAMPLE_CATEGORIES = ["reasoning","filler","outcome","distractor"]


def combine_policies(tables_dir):
    """Join the in domain and transfer summaries and add the drop between them"""
    gsm = pd.read_csv(tables_dir/"gsm_policy_summary.csv").set_index("policy")
    transfer = pd.read_csv(tables_dir/"transfer_policy_summary.csv").set_index("policy")
    gsm_pairs = pd.read_csv(tables_dir/"gsm_pairwise_summary.csv").set_index("pair")
    transfer_pairs = pd.read_csv(tables_dir/"transfer_pairwise_summary.csv").set_index("pair")
    rows = []
    for name in POLICIES:
        pair = f"{name}_vs_sft"
        row = {
            "policy":name,
            "gsm_accuracy":gsm.loc[name,"exact_accuracy"],
            "transfer_accuracy":transfer.loc[name,"exact_accuracy"],
            "accuracy_drop":gsm.loc[name,"exact_accuracy"] - transfer.loc[name,"exact_accuracy"],
            "gsm_tokens":gsm.loc[name,"mean_response_tokens"],
            "transfer_tokens":transfer.loc[name,"mean_response_tokens"],
            "gsm_format":gsm.loc[name,"format_compliance"],
            "transfer_format":transfer.loc[name,"format_compliance"],
            "gsm_wrong_number":gsm.loc[name,"wrong_number_rate"],
            "transfer_wrong_number":transfer.loc[name,"wrong_number_rate"],
            "gsm_missing_final":gsm.loc[name,"missing_final_rate"],
            "transfer_missing_final":transfer.loc[name,"missing_final_rate"],
        }
        if pair in gsm_pairs.index:
            row["gsm_win_vs_sft"] = gsm_pairs.loc[pair,"first_win_rate"]
            row["transfer_win_vs_sft"] = transfer_pairs.loc[pair,"first_win_rate"]
            row["win_drop"] = row["gsm_win_vs_sft"] - row["transfer_win_vs_sft"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_accuracy(combined, figures_dir):
    """Plot exact accuracy on the in domain set and on the transfer set for every policy"""
    positions = np.arange(len(combined))
    fig,ax = plt.subplots(figsize=(6.4,4.4))
    ax.bar(positions - 0.2,combined["gsm_accuracy"],0.4,color="steelblue",label="GSM8K")
    ax.bar(positions + 0.2,combined["transfer_accuracy"],0.4,color="darkorange",label="SVAMP")
    for x,a,b in zip(positions,combined["gsm_accuracy"],combined["transfer_accuracy"]):
        ax.text(x - 0.2,a + 0.01,f"{a:.2f}",ha="center",fontsize=9)
        ax.text(x + 0.2,b + 0.01,f"{b:.2f}",ha="center",fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels([p.upper() for p in combined["policy"]])
    ax.set_ylabel("exact accuracy")
    ax.set_ylim(0,1.0)
    ax.legend(loc="upper center",ncol=2,frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir/"accuracy_in_domain_vs_transfer.png",dpi=200)
    plt.close(fig)


def plot_diagnostics(rates, figures_dir):
    """Plot the better and tie and wrong preference rates of both mechanisms in every category"""
    categories = list(CATEGORIES)
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,mechanism in zip(axes,["verifier","judge"]):
        part = rates[rates["mechanism"] == mechanism].set_index("category").loc[categories]
        bottom = np.zeros(len(categories))
        for outcome,color in OUTCOME_COLORS.items():
            values = part[f"{outcome}_rate"].to_numpy()
            ax.bar(categories,values,bottom=bottom,color=color,label=outcome)
            bottom += values
        ax.set_title("exact verifier" if mechanism == "verifier" else "pairwise AI judge")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories,rotation=20,ha="right")
    axes[0].set_ylabel("fraction of pairs")
    handles,names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0.07,1,1))
    fig.savefig(figures_dir/"diagnostic_preferences.png",dpi=200)
    plt.close(fig)


def pick_examples(cfg, records):
    """Pick pairs where the verifier and the judge disagree for the qualitative discussion

    Pairs where the judge is the one that gets it wrong come first because
    they show what the judge can be fooled by. A category with no disagreement
    keeps its first pairs so the shared tie is still shown
    """
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    examples = []
    for category in EXAMPLE_CATEGORIES:
        part = [r for r in records if r["category"] == category and r["verifier"] != r["judge"]]
        if not part:
            part = [r for r in records if r["category"] == category]
        part.sort(key=lambda r: r["judge"] != "wrong")
        for r in part[:EXAMPLES_PER_CATEGORY]:
            variants = groups[r["problem_id"]]
            examples.append({
                "category":category,
                "problem_id":r["problem_id"],
                "question":variants[r["better"]]["question"],
                "gold_final":variants[r["better"]]["gold_final"],
                "verifier":r["verifier"],
                "judge":r["judge"],
                "better_response":variants[r["better"]]["response"],
                "worse_response":variants[r["worse"]]["response"],
            })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"])/"task5_feedback"
    tables_dir = outdir/"tables"
    figures_dir = outdir/"figures"
    figures_dir.mkdir(parents=True,exist_ok=True)

    combined = combine_policies(tables_dir)
    combined.to_csv(tables_dir/"final_comparison.csv",index=False)
    plot_accuracy(combined,figures_dir)

    rates = pd.read_csv(tables_dir/"diagnostic_rates.csv")
    plot_diagnostics(rates,figures_dir)

    records = read_jsonl(outdir/"diagnostic_pairs.jsonl")
    save_json(outdir/"qualitative_examples.json",pick_examples(cfg,records))

    sensitivity = load_json(tables_dir/"diagnostic_sensitivity.json")
    save_json(tables_dir/"feedback_cost.json",{
        "verifier_calls_per_response":1,
        "judge_calls_per_group_of_4":6,
        "judge_calls_made_on_diagnostics":sensitivity["judge_calls_made"],
        "judge_seconds_on_diagnostics":sensitivity["judge_seconds"],
    })
    print(combined.to_string())
    print(rates.to_string())


if __name__ == "__main__":
    main()
