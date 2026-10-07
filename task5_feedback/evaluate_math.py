from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final

GENERATION_BATCH_SIZE = 16
PAIRS = [("rlaif","sft"),("rlvr","sft"),("rlaif","rlvr")]


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def generate_for_policy(cfg, rows, name: str, tokenizer):
    """Generate one greedy response per problem and score it with the exact verifier"""
    model = load_frozen_policy(cfg,name)
    records = []
    for start in range(0,len(rows),GENERATION_BATCH_SIZE):
        chunk = rows[start:start + GENERATION_BATCH_SIZE]
        gen = batch_generate(
            model,
            tokenizer,
            [r["messages"] for r in chunk],
            max_prompt_length=512,
            max_new_tokens=int(cfg["math_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for row,response,n_tok,cut in zip(chunk,gen["responses"],gen["response_lengths"],gen["truncated"]):
            records.append({
                "policy":name,
                "index":len(records),
                "question":row["question"],
                "gold_final":str(row["gold_final"]),
                "response":response,
                "response_tokens":int(n_tok),
                "truncated":bool(cut),
                "has_final_field":extract_designated_final(response) is not None,
                "exact":exact_reward(response,str(row["gold_final"])),
            })
    clear_gpu(model)
    return records


def policy_table(generated):
    """Summarize accuracy format length and failure types for every policy"""
    table = []
    for name,records in generated.items():
        df = pd.DataFrame(records)
        wrong = df[df["exact"] == 0]
        table.append({
            "policy":name,
            "n":len(df),
            "exact_accuracy":float(df["exact"].mean()),
            "format_compliance":float(df["has_final_field"].mean()),
            "mean_response_tokens":float(df["response_tokens"].mean()),
            "truncated_rate":float(df["truncated"].mean()),
            "wrong_number_rate":float(wrong["has_final_field"].sum()/len(df)),
            "missing_final_rate":float((~wrong["has_final_field"]).sum()/len(df)),
        })
    return pd.DataFrame(table)


def judge_pairs(judge, generated):
    """Ask the fixed judge which response is better for every policy pair and every problem"""
    records = []
    for first,second in PAIRS:
        for a,b in zip(generated[first],generated[second]):
            verdict = judge.compare(a["question"],a["response"],b["response"])
            records.append({
                "pair":f"{first}_vs_{second}",
                "index":a["index"],
                "verdict":verdict,
                "exact_first":a["exact"],
                "exact_second":b["exact"],
            })
    return records


def pair_table(records):
    """Report the judge win rate and the agreement between the judge and the exact verifier

    The win rate counts a tie as half a win for the first policy. Agreement
    over all problems counts a tie as agreeing with equal exact scores. The
    split columns only look at problems where exactly one response is correct
    """
    table = []
    for pair,part in pd.DataFrame(records).groupby("pair",sort=False):
        wins = (part["verdict"] == "A").sum()
        ties = (part["verdict"] == "TIE").sum()
        verifier = (part["exact_first"] - part["exact_second"]).apply(lambda d: "A" if d > 0 else ("B" if d < 0 else "TIE"))
        split = part[verifier != "TIE"]
        split_verifier = verifier[verifier != "TIE"]
        table.append({
            "pair":pair,
            "n":len(part),
            "first_win_rate":float((wins + 0.5*ties)/len(part)),
            "judge_tie_rate":float(ties/len(part)),
            "agreement_all":float((part["verdict"] == verifier).mean()),
            "n_verifier_split":len(split),
            "judge_picks_correct":float((split["verdict"] == split_verifier).mean()) if len(split) else float("nan"),
            "judge_tie_on_split":float((split["verdict"] == "TIE").mean()) if len(split) else float("nan"),
            "judge_picks_wrong":float(((split["verdict"] != split_verifier) & (split["verdict"] != "TIE")).mean()) if len(split) else float("nan"),
        })
    return pd.DataFrame(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, tokenizer = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))

    outdir = repo_path(cfg["results_dir"])/"task5_feedback"
    (outdir/"tables").mkdir(parents=True,exist_ok=True)
    generated = {}
    for name in policy_specs(cfg):
        out_path = outdir/f"{args.dataset}_generated_{name}.jsonl"
        # skipping a finished policy so an interrupted run can resume
        if out_path.exists():
            generated[name] = read_jsonl(out_path)
            print(f"{name}: already generated")
            continue
        generated[name] = generate_for_policy(cfg,rows,name,tokenizer)
        write_jsonl(out_path,generated[name])
        acc = sum(r["exact"] for r in generated[name])/len(rows)
        print(f"{name}: exact accuracy {acc:.3f}")

    judge = PairwiseAIJudge(cfg,outdir/"judge_cache.json")
    records = judge_pairs(judge,generated)
    write_jsonl(outdir/f"{args.dataset}_pairwise.jsonl",records)

    policies = policy_table(generated)
    pairs = pair_table(records)
    policies.to_csv(outdir/"tables"/f"{args.dataset}_policy_summary.csv",index=False)
    pairs.to_csv(outdir/"tables"/f"{args.dataset}_pairwise_summary.csv",index=False)
    save_json(outdir/"tables"/f"{args.dataset}_summary.json",{
        "policies":policies.to_dict("records"),
        "pairs":pairs.to_dict("records"),
    })
    print(policies.to_string())
    print(pairs.to_string())


if __name__ == "__main__":
    main()
