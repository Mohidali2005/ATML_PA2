from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import pandas as pd
from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from task1_dpo.evaluate import evaluate_policy, load_evaluation_bundle
from task1_dpo.train import run_training

BETA_COLOR = "darkorange"
STANDARD_COLOR = "steelblue"


def run_one_beta(config_path, cfg, beta):
    """Train and evaluate one short dpo fork at the given beta

    Every fork starts from the same original policy initialization and
    trains on the same short example subset so beta is the only thing
    that changes between the three conditions
    """
    run_name = f"beta_{beta}"
    output_path = f"outputs/task1_dpo/{run_name}"
    run_training(config_path,run_name,output_path=output_path,beta=beta,max_examples=int(cfg["short_ablation_examples"]))
    bundle = load_evaluation_bundle(config_path,output_path)
    return evaluate_policy(bundle,run_name,beta=beta)


def plot_beta_sweep(rows, figures_dir):
    """Plot how preference fitting reference drift reward and length move with beta

    The standard full epoch run is drawn as a single star off the beta
    sweep line since it used a different training budget and is only
    shown here for context rather than as part of the controlled sweep
    """
    sweep = sorted([r for r in rows if r["name"] != "standard"],key=lambda r:r["beta"])
    standard = next(r for r in rows if r["name"] == "standard")

    metrics = [
        ("preference_accuracy","held out preference accuracy"),
        ("mean_kl","sampled kl from reference"),
        ("mean_reward","reward model score"),
        ("mean_response_length","mean response length"),
    ]

    fig,axes = plt.subplots(2,2,figsize=(10,8))
    for ax,(key,label) in zip(axes.flat,metrics):
        betas = [r["beta"] for r in sweep]
        values = [r[key] for r in sweep]
        ax.plot(betas,values,marker="o",color=BETA_COLOR,label="short beta sweep")
        ax.scatter([standard["beta"]],[standard[key]],marker="*",s=180,color=STANDARD_COLOR,label="standard full epoch",zorder=5)
        ax.set_xlabel("beta")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8,loc="best")

    fig.tight_layout()
    out_path = figures_dir/"beta_sweep.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    figures_dir.mkdir(parents=True,exist_ok=True)

    standard_result = load_json(tables_dir/"standard_eval.json")
    rows = [standard_result]
    for beta in cfg["betas"]:
        rows.append(run_one_beta(args.config,cfg,beta))

    table = pd.DataFrame(rows)
    table.to_csv(tables_dir/"beta_sweep.csv",index=False)
    print(table)

    plot_beta_sweep(rows,figures_dir)


if __name__ == "__main__":
    main()
