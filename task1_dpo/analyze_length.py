from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate, response_sequence_logprobs
from common.logging_utils import save_json
from common.metrics import word_limit_compliance
from common.models import load_policy, load_tokenizer
from task1_dpo.train import fitting_rows, make_collate, run_training

MODEL_COLORS = {"standard":"steelblue","length_balanced":"darkorange"}
STRATA = ["preferred_longer","length_matched","rejected_longer"]
STRATA_LABELS = {"preferred_longer":"preferred longer","length_matched":"length matched","rejected_longer":"rejected longer"}


def per_stratum_accuracy(policy, tokenizer, rows, max_length):
    """Compute held out preference accuracy separately for each length stratum

    Grouping the per pair correctness by the length stratum field turns a
    policy that only learned to prefer longer text into a visible gap
    between the preferred longer and rejected longer strata instead of
    hiding it inside one averaged number
    """
    device = next(policy.parameters()).device
    loader = DataLoader(rows,batch_size=8,shuffle=False,collate_fn=make_collate(tokenizer,max_length))

    correct = []
    with torch.no_grad():
        for chosen,rejected in loader:
            chosen = {k:v.to(device) for k,v in chosen.items()}
            rejected = {k:v.to(device) for k,v in rejected.items()}
            policy_chosen_logp,_,_ = response_sequence_logprobs(policy,chosen)
            policy_rejected_logp,_,_ = response_sequence_logprobs(policy,rejected)
            correct.extend((policy_chosen_logp > policy_rejected_logp).tolist())

    by_stratum = {stratum:[] for stratum in STRATA}
    for row,is_correct in zip(rows,correct):
        by_stratum[row["length_stratum"]].append(is_correct)
    return {stratum:float(sum(values))/len(values) for stratum,values in by_stratum.items()}


def evaluate_word_limits(policy, tokenizer, rows, cfg):
    """Generate one response per word limit prompt and check the stated limit

    Returns the mean generated length and the fraction of responses that
    stayed within the word limit named in the prompt itself
    """
    gen_cfg = cfg["generation"]
    prompts = [row["messages"] for row in rows]
    gen = batch_generate(
        policy,tokenizer,prompts,
        max_prompt_length=int(cfg["max_sequence_length"]),
        max_new_tokens=int(cfg["max_generation_tokens"]),
        temperature=gen_cfg["temperature"],top_p=gen_cfg["top_p"],do_sample=gen_cfg["do_sample"],
    )
    compliant = []
    for row,response in zip(rows,gen["responses"]):
        prompt_text = row["messages"][0]["content"]
        compliant.append(word_limit_compliance(prompt_text,response))
    mean_length = float(np.mean(gen["response_lengths"]))
    compliance_rate = float(np.mean(compliant))
    return mean_length,compliance_rate,gen["responses"]


def plot_length_analysis(per_stratum, word_limit, figures_dir):
    """Plot per stratum preference accuracy alongside word limit compliance

    The three length strata sit side by side for both models so a length
    biased policy shows up as a gap between the preferred longer and
    rejected longer bars rather than being averaged away
    """
    models = list(MODEL_COLORS.keys())
    fig,axes = plt.subplots(1,3,figsize=(15,4.5))

    x = np.arange(len(STRATA))
    bar_width = 0.35
    for i,model_name in enumerate(models):
        values = [per_stratum[model_name][stratum] for stratum in STRATA]
        axes[0].bar(x+i*bar_width,values,width=bar_width,label=model_name,color=MODEL_COLORS[model_name])
    axes[0].set_xticks(x+bar_width/2)
    axes[0].set_xticklabels([STRATA_LABELS[s] for s in STRATA])
    axes[0].set_ylabel("preference accuracy")
    axes[0].set_ylim(0,1.05)
    axes[0].set_title("accuracy by length stratum")
    axes[0].legend(fontsize=8)

    compliance_values = [word_limit[m][1] for m in models]
    axes[1].bar(models,compliance_values,color=[MODEL_COLORS[m] for m in models])
    axes[1].set_ylabel("word limit compliance rate")
    axes[1].set_ylim(0,1.05)
    axes[1].set_title("compliance on word limit prompts")

    length_values = [word_limit[m][0] for m in models]
    axes[2].bar(models,length_values,color=[MODEL_COLORS[m] for m in models])
    axes[2].set_ylabel("mean response length in tokens")
    axes[2].set_title("length on word limit prompts")

    fig.tight_layout()
    out_path = figures_dir/"length_confounding.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tokenizer = load_tokenizer(cfg["base_model"])
    max_length = int(cfg["max_sequence_length"])
    stratified = fitting_rows(tokenizer,read_jsonl(cfg["paths"]["dpo_length_eval"]),max_length)
    print("Length-stratified eval rows:", len(stratified))

    run_training(args.config,"length_balanced",dataset_path=cfg["paths"]["dpo_length_train"],output_path=cfg["length_output"])

    word_limit_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])

    policies = {"standard":cfg["standard_output"],"length_balanced":cfg["length_output"]}
    per_stratum = {}
    word_limit = {}
    for name,adapter_path in policies.items():
        policy = load_policy(cfg,adapter_path=adapter_path,trainable=False)
        per_stratum[name] = per_stratum_accuracy(policy,tokenizer,stratified,max_length)
        mean_length,compliance_rate,_ = evaluate_word_limits(policy,tokenizer,word_limit_rows,cfg)
        word_limit[name] = (mean_length,compliance_rate)
        print(name,per_stratum[name],"compliance_rate",compliance_rate,"mean_length",mean_length)

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    save_json(tables_dir/"length_stratum_accuracy.json",per_stratum)
    save_json(tables_dir/"word_limit_compliance.json",{name:{"mean_length":ml,"compliance_rate":cr} for name,(ml,cr) in word_limit.items()})

    plot_length_analysis(per_stratum,word_limit,figures_dir)


if __name__ == "__main__":
    main()
