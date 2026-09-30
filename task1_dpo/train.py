from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import save_json, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def pair_logprobs(model, chosen, rejected):
    """Return the policy and reference sequence log probabilities for one batch

    The reference pass reuses the same weights with the lora adapter turned
    off so no second copy of the base model needs to be kept in memory
    """
    policy_chosen_logp,_,_ = response_sequence_logprobs(model,chosen)
    policy_rejected_logp,_,_ = response_sequence_logprobs(model,rejected)
    with torch.no_grad(),reference_mode(model):
        ref_chosen_logp,_,_ = response_sequence_logprobs(model,chosen)
        ref_rejected_logp,_,_ = response_sequence_logprobs(model,rejected)
    return policy_chosen_logp,policy_rejected_logp,ref_chosen_logp,ref_rejected_logp


def plot_training_curve(history, run_name, figures_dir):
    """Plot the per step dpo loss and preference accuracy for one training run"""
    table = pd.DataFrame(history)
    fig,axes = plt.subplots(1,2,figsize=(10,4))

    axes[0].plot(table["global_step"],table["loss"],color="steelblue")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("dpo loss")
    axes[0].set_title(f"{run_name} training loss")

    axes[1].plot(table["global_step"],table["preference_accuracy"],color="darkorange")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("batch preference accuracy")
    axes[1].set_ylim(0,1.05)
    axes[1].set_title(f"{run_name} batch preference accuracy")

    fig.tight_layout()
    out_path = figures_dir/f"{run_name}_training_curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"saved {out_path}")


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    grad_accum_steps = int(cfg["grad_accum_steps"])
    max_grad_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg["epochs"])

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    figures_dir = repo_path(cfg["results_dir"])/"figures"
    tables_dir.mkdir(parents=True,exist_ok=True)
    figures_dir.mkdir(parents=True,exist_ok=True)

    history = []
    global_step = 0
    accum_count = 0
    optimizer.zero_grad()

    for epoch in range(epochs):
        for chosen,rejected in loader:
            chosen = {k:v.to(device) for k,v in chosen.items()}
            rejected = {k:v.to(device) for k,v in rejected.items()}

            policy_chosen_logp,policy_rejected_logp,ref_chosen_logp,ref_rejected_logp = pair_logprobs(model,chosen,rejected)
            loss,diagnostics = dpo_loss(policy_chosen_logp,policy_rejected_logp,ref_chosen_logp,ref_rejected_logp,beta)
            (loss/grad_accum_steps).backward()
            accum_count += 1

            if accum_count == grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model),max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                history.append({
                    "epoch":epoch,
                    "global_step":global_step,
                    "loss":loss.item(),
                    "preference_accuracy":diagnostics["preference_accuracy"].item(),
                })
                if global_step%10 == 0:
                    print(f"epoch {epoch} step {global_step} loss {loss.item():.3f} preference_accuracy {diagnostics['preference_accuracy'].item():.3f}")

        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model),max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            global_step += 1
            history.append({
                "epoch":epoch,
                "global_step":global_step,
                "loss":loss.item(),
                "preference_accuracy":diagnostics["preference_accuracy"].item(),
            })

        print(f"epoch {epoch} finished at step {global_step} loss {history[-1]['loss']:.3f}")

    model.save_pretrained(output)
    print(f"saved adapter to {output}")

    save_json(tables_dir/f"{run_name}_training_curve.json",history)
    plot_training_curve(history,run_name,figures_dir)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
