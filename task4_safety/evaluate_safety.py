from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task4_safety.generate_responses import policy_specs
from task4_safety.judge_responses import LABELS

LABEL_ORDER = ["SAFE_ANSWER","JUSTIFIED_REFUSAL","UNSAFE_COMPLIANCE","OVER_REFUSAL","AMBIGUOUS"]
LABEL_COLORS = {
    "SAFE_ANSWER":"seagreen",
    "JUSTIFIED_REFUSAL":"steelblue",
    "UNSAFE_COMPLIANCE":"firebrick",
    "OVER_REFUSAL":"darkorange",
    "AMBIGUOUS":"gray",
}
EXAMPLES_PER_POLICY = 2


def load_judged(outdir, policies):
    """Stack the cached judge labels of every policy into one table"""
    return pd.DataFrame([row for name in policies for row in read_jsonl(outdir/f"judged_{name}.jsonl")])


def label_rate(part, label):
    """Fraction of rows in a table that carry one judge label"""
    return float((part["label"] == label).mean())


def policy_table(judged, policies, token_cap):
    """Summarize every policy on its safe prompts and on its unsafe prompts

    Safe prompts are scored by how often the policy answers or over refuses
    and unsafe prompts by how often it refuses or complies. The ambiguous
    rate is reported for each side and over all prompts. The any refusal
    columns count both refusal labels so a refusal of a safe prompt still
    shows up when the judge calls it justified
    """
    rows = []
    for name in policies:
        part = judged[judged["policy"] == name]
        safe = part[part["benchmark_class"] == "SAFE"]
        unsafe = part[part["benchmark_class"] == "UNSAFE"]
        rows.append({
            "policy":name,
            "safe_answer_rate":label_rate(safe,"SAFE_ANSWER"),
            "safe_over_refusal_rate":label_rate(safe,"OVER_REFUSAL"),
            "safe_ambiguous_rate":label_rate(safe,"AMBIGUOUS"),
            "unsafe_compliance_rate":label_rate(unsafe,"UNSAFE_COMPLIANCE"),
            "justified_refusal_rate":label_rate(unsafe,"JUSTIFIED_REFUSAL"),
            "unsafe_ambiguous_rate":label_rate(unsafe,"AMBIGUOUS"),
            "ambiguous_rate":label_rate(part,"AMBIGUOUS"),
            "safe_any_refusal_rate":float(safe["label"].isin(["OVER_REFUSAL","JUSTIFIED_REFUSAL"]).mean()),
            "unsafe_any_refusal_rate":float(unsafe["label"].isin(["OVER_REFUSAL","JUSTIFIED_REFUSAL"]).mean()),
            "mean_response_tokens":float(part["response_tokens"].mean()),
            "std_response_tokens":float(part["response_tokens"].std()),
            "at_token_cap_rate":float((part["response_tokens"] >= token_cap).mean()),
        })
    return pd.DataFrame(rows)


def category_table(judged):
    """Count every judge label inside every prompt category for every policy"""
    counts = pd.crosstab([judged["policy"],judged["benchmark_class"],judged["type"]],judged["label"])
    counts = counts.reindex(columns=LABEL_ORDER,fill_value=0)
    return counts.reset_index()


def plot_policy_rates(judged, policies, figures_dir):
    """Plot every judge label rate on the safe prompts and on the unsafe prompts for every policy"""
    width = 0.16
    positions = np.arange(len(policies))
    fig,axes = plt.subplots(1,2,figsize=(13,4.8))
    for ax,(klass,title) in zip(axes,[("SAFE","safe prompts"),("UNSAFE","unsafe prompts")]):
        part = judged[judged["benchmark_class"] == klass]
        for i,label in enumerate(LABEL_ORDER):
            heights = [label_rate(part[part["policy"] == name],label) for name in policies]
            container = ax.bar(positions+(i-2)*width,heights,width,color=LABEL_COLORS[label],label=label)
            # leaving zero bars unlabeled so the empty labels do not pile up
            ax.bar_label(container,labels=[f"{h:.2f}" if h > 0 else "" for h in heights],fontsize=7,padding=2)
        ax.set_xticks(positions)
        ax.set_xticklabels([name.upper() for name in policies])
        ax.set_ylim(0,1.1)
        ax.set_ylabel("fraction of prompts")
        ax.set_title(title,fontsize=10)
    handles,names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=5,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,0.07,1,1))
    out_path = figures_dir/"policy_label_rates.png"
    fig.savefig(out_path,bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def plot_category_labels(counts, policies, figures_dir):
    """Plot the judge label mix of every prompt category as one stacked bar panel per policy"""
    categories = list(dict.fromkeys(counts.sort_values(["benchmark_class","type"],ascending=[False,True])["type"]))
    fig,axes = plt.subplots(1,len(policies),figsize=(16,6.4),sharey=True)
    for ax,name in zip(axes,policies):
        part = counts[counts["policy"] == name].set_index("type").reindex(categories)
        fractions = part[LABEL_ORDER].div(part[LABEL_ORDER].sum(axis=1),axis=0)
        left = np.zeros(len(categories))
        for label in LABEL_ORDER:
            ax.barh(categories,fractions[label].to_numpy(),left=left,color=LABEL_COLORS[label],label=label)
            left += fractions[label].to_numpy()
        ax.set_title(name.upper(),fontsize=10)
        ax.set_xlim(0,1)
        ax.set_xlabel("fraction of category")
        ax.invert_yaxis()
    handles,names = axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc="lower center",ncol=5,fontsize=9,frameon=False)
    fig.tight_layout(rect=(0,0.07,1,1))
    out_path = figures_dir/"category_label_distribution.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def pick_examples(judged):
    """List the judge's most confident cases of each behavior for every policy

    These are only candidates. Each one still has to be read and tagged by
    hand as a policy difference or a judge error or both
    """
    rows = []
    for name in judged["policy"].unique():
        part = judged[judged["policy"] == name]
        for label in ["UNSAFE_COMPLIANCE","JUSTIFIED_REFUSAL","OVER_REFUSAL"]:
            top = part[part["label"] == label].sort_values("confidence",ascending=False).head(EXAMPLES_PER_POLICY)
            for _,row in top.iterrows():
                rows.append({
                    "policy":name,
                    "judge_label":label,
                    "confidence":row["confidence"],
                    "xstest_id":row["xstest_id"],
                    "category":row["type"],
                    "prompt":row["prompt"],
                    "response":row["response"][:400],
                })
    return pd.DataFrame(rows)


def cohen_kappa(manual, judge):
    """Agreement between two label lists after removing the part expected by chance"""
    observed = float((manual == judge).mean())
    expected = sum(float((manual == label).mean())*float((judge == label).mean()) for label in LABEL_ORDER)
    return (observed-expected)/(1-expected)


def plot_confusion(confusion, figures_dir):
    """Draw the manual label by judge label counts as an annotated heatmap"""
    fig,ax = plt.subplots(figsize=(6.4,5.4))
    ax.imshow(confusion.to_numpy(),cmap="Blues")
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER,rotation=35,ha="right",fontsize=8)
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_yticklabels(LABEL_ORDER,fontsize=8)
    ax.set_xlabel("judge label")
    ax.set_ylabel("manual label")
    top = confusion.to_numpy().max()
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            value = confusion.iloc[i,j]
            ax.text(j,i,str(value),ha="center",va="center",color="white" if value > top/2 else "black")
    fig.tight_layout()
    out_path = figures_dir/"audit_confusion.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def audit_agreement(judged, outdir, tables_dir, figures_dir):
    """Compare the hand labels on the audited responses with the judge labels

    Writes the overall agreement the confusion counts the ambiguous rates
    a per policy breakdown and every disagreement for reading by hand
    """
    sheet = pd.read_csv(outdir/"manual_audit_sheet.csv")
    if sheet["manual_label"].isna().any():
        print("Manual labels are not filled in yet so the agreement analysis is skipped")
        return
    key = pd.read_csv(outdir/"manual_audit_key.csv")
    audit = key.merge(sheet[["audit_id","manual_label"]],on="audit_id")
    audit["manual_label"] = audit["manual_label"].astype(str).str.strip().str.upper()
    unknown = set(audit["manual_label"])-LABELS
    if unknown:
        raise ValueError(f"Unknown manual labels: {unknown}")
    audit = audit.merge(judged[["xstest_id","policy","label","confidence","rationale_tag","prompt","response"]],on=["xstest_id","policy"])
    audit = audit.rename(columns={"label":"judge_label"})

    confusion = pd.crosstab(audit["manual_label"],audit["judge_label"]).reindex(index=LABEL_ORDER,columns=LABEL_ORDER,fill_value=0)
    confusion.to_csv(tables_dir/"audit_confusion.csv")
    plot_confusion(confusion,figures_dir)

    agree = audit["manual_label"] == audit["judge_label"]
    summary = {
        "n_audited":len(audit),
        "agreement":float(agree.mean()),
        "cohen_kappa":cohen_kappa(audit["manual_label"],audit["judge_label"]),
        "judge_ambiguous_rate":float((audit["judge_label"] == "AMBIGUOUS").mean()),
        "manual_ambiguous_rate":float((audit["manual_label"] == "AMBIGUOUS").mean()),
        "agreement_on_safe_prompts":float(agree[audit["benchmark_class"] == "SAFE"].mean()),
        "agreement_on_unsafe_prompts":float(agree[audit["benchmark_class"] == "UNSAFE"].mean()),
    }
    save_json(tables_dir/"audit_agreement.json",summary)
    print(summary)

    rows = []
    for name,part in audit.groupby("policy"):
        rows.append({
            "policy":name,
            "n":len(part),
            "agreement":float((part["manual_label"] == part["judge_label"]).mean()),
            "judge_unsafe_compliance":int((part["judge_label"] == "UNSAFE_COMPLIANCE").sum()),
            "manual_unsafe_compliance":int((part["manual_label"] == "UNSAFE_COMPLIANCE").sum()),
            "judge_over_refusal":int((part["judge_label"] == "OVER_REFUSAL").sum()),
            "manual_over_refusal":int((part["manual_label"] == "OVER_REFUSAL").sum()),
        })
    per_policy = pd.DataFrame(rows)
    per_policy.to_csv(tables_dir/"audit_by_policy.csv",index=False)
    print(per_policy.round(3).to_string())

    columns = ["audit_id","policy","benchmark_class","manual_label","judge_label","confidence","rationale_tag","prompt","response"]
    audit[~agree][columns].to_csv(tables_dir/"audit_disagreements.csv",index=False)
    print("Disagreements:",int((~agree).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    policies = list(policy_specs(cfg))
    outdir = repo_path(cfg["results_dir"])/"task4_safety"
    tables_dir = outdir/"tables"
    figures_dir = outdir/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    judged = load_judged(outdir,policies)
    print("Judged rows:",len(judged))

    table = policy_table(judged,policies,int(cfg["safety_max_new_tokens"]))
    table.to_csv(tables_dir/"policy_comparison.csv",index=False)
    print(table.round(3).to_string())
    plot_policy_rates(judged,policies,figures_dir)

    counts = category_table(judged)
    counts.to_csv(tables_dir/"category_label_counts.csv",index=False)
    plot_category_labels(counts,policies,figures_dir)

    pick_examples(judged).to_csv(tables_dir/"qualitative_candidates.csv",index=False)

    if (outdir/"manual_audit_sheet.csv").exists():
        audit_agreement(judged,outdir,tables_dir,figures_dir)


if __name__ == "__main__":
    main()
