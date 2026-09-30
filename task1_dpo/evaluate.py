from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.metrics import sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate

GENERATION_BATCH_SIZE = 8


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def held_out_loss_and_accuracy(policy, tokenizer, rows, beta, max_length):
    """Score every held out pair with the policy and its own frozen reference

    Returns the mean dpo loss and the fraction of pairs the policy ranks
    correctly, both measured on pairs the model never trained on
    """
    device = next(policy.parameters()).device
    loader = DataLoader(rows,batch_size=8,collate_fn=make_collate(tokenizer,max_length))

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for chosen,rejected in loader:
            chosen = {k:v.to(device) for k,v in chosen.items()}
            rejected = {k:v.to(device) for k,v in rejected.items()}
            policy_chosen_logp,_,_ = response_sequence_logprobs(policy,chosen)
            policy_rejected_logp,_,_ = response_sequence_logprobs(policy,rejected)
            with reference_mode(policy):
                ref_chosen_logp,_,_ = response_sequence_logprobs(policy,chosen)
                ref_rejected_logp,_,_ = response_sequence_logprobs(policy,rejected)

            loss,_ = dpo_loss(policy_chosen_logp,policy_rejected_logp,ref_chosen_logp,ref_rejected_logp,beta)
            n = policy_chosen_logp.shape[0]
            total_loss += loss.item()*n
            total_correct += int((policy_chosen_logp > policy_rejected_logp).sum().item())
            total_count += n

    return total_loss/total_count,total_correct/total_count


def generate_and_score(policy, tokenizer, reward_bundle, rows, cfg):
    """Generate one response per held out prompt and score every response

    Returns the sampled kl from the reference policy the reward model
    score the response length statistics and a per example record for
    picking qualitative examples later
    """
    reward_model,reward_tokenizer = reward_bundle
    gen_cfg = cfg["generation"]
    kl_numerator = 0.0
    kl_denominator = 0.0
    reward_values = []
    lengths = []
    records = []

    for start in range(0,len(rows),GENERATION_BATCH_SIZE):
        chunk = rows[start:start+GENERATION_BATCH_SIZE]
        prompts = [prompt_messages_from_preference(row) for row in chunk]
        gen = batch_generate(
            policy,tokenizer,prompts,
            max_prompt_length=int(cfg["max_sequence_length"]),
            max_new_tokens=int(cfg["max_generation_tokens"]),
            temperature=gen_cfg["temperature"],top_p=gen_cfg["top_p"],do_sample=gen_cfg["do_sample"],
        )

        with torch.no_grad():
            policy_logp,_ = response_token_logprobs(policy,gen["sequences"],gen["attention_mask"],gen["prompt_width"],gen["response_ids"])
            with reference_mode(policy):
                ref_logp,_ = response_token_logprobs(policy,gen["sequences"],gen["attention_mask"],gen["prompt_width"],gen["response_ids"])
        kl_chunk = sampled_kl(policy_logp,ref_logp,gen["response_mask"])
        n_tokens = float(gen["response_mask"].sum().item())
        kl_numerator += kl_chunk.item()*n_tokens
        kl_denominator += n_tokens

        rewards = score_reward_pairs(reward_model,reward_tokenizer,prompts,gen["responses"]).tolist()
        reward_values.extend(rewards)
        lengths.extend(gen["response_lengths"])

        for row,response,reward,length in zip(chunk,gen["responses"],rewards,gen["response_lengths"]):
            records.append({
                "prompt_id":row.get("prompt_id"),
                "response":response,
                "reward_score":reward,
                "response_length":length,
            })

    mean_kl = kl_numerator/kl_denominator
    mean_reward = float(np.mean(reward_values))
    mean_length = float(np.mean(lengths))
    std_length = float(np.std(lengths))
    return mean_kl,mean_reward,mean_length,std_length,records


def evaluate_policy(bundle, name, beta=None):
    """Run the full held out evaluation for one trained dpo policy

    Computes the held out dpo loss and preference accuracy from the
    training style pairs plus the sampled kl the reward model score and
    the response length statistics from fresh generations on the same
    held out prompts then saves everything to disk. The beta passed in
    should match whatever beta the given adapter was actually trained
    with since it is only used to score the held out pairs not to train
    """
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    tokenizer = bundle["tokenizer"]
    rows = bundle["rows"]
    beta = float(cfg["beta"] if beta is None else beta)
    max_length = int(cfg["max_sequence_length"])

    dpo_loss_value,pref_accuracy = held_out_loss_and_accuracy(policy,tokenizer,rows,beta,max_length)
    mean_kl,mean_reward,mean_length,std_length,records = generate_and_score(policy,tokenizer,bundle["reward"],rows,cfg)

    result = {
        "name":name,
        "beta":beta,
        "dpo_loss":dpo_loss_value,
        "preference_accuracy":pref_accuracy,
        "mean_kl":mean_kl,
        "mean_reward":mean_reward,
        "mean_response_length":mean_length,
        "std_response_length":std_length,
    }
    print(result)

    tables_dir = repo_path(cfg["results_dir"])/"tables"
    tables_dir.mkdir(parents=True,exist_ok=True)
    save_json(tables_dir/f"{name}_eval.json",result)
    pd.DataFrame(records).to_csv(tables_dir/f"{name}_generations.csv",index=False)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    evaluate_policy(bundle, args.name)


if __name__ == "__main__":
    main()
