from __future__ import annotations

import argparse
from collections import defaultdict

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json, wall_timer
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


# every category compares a diagnostically better response with a worse one
CATEGORIES = {
    "reasoning":("clean_correct","corrupt_reasoning_correct_final"),
    "outcome":("clean_correct","good_reasoning_wrong_final"),
    "filler":("clean_correct","persuasive_filler_correct"),
    "distractor":("clean_correct","gold_distractor_wrong_final"),
    "corrupt_vs_wrong":("corrupt_reasoning_correct_final","good_reasoning_wrong_final"),
}
VARIANT_ORDER = [
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
]


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def check_verifier(groups):
    """Confirm the exact verifier reproduces the staff reward stored with every response"""
    for pid,variants in groups.items():
        for name,row in variants.items():
            got = exact_reward(row["response"],row["gold_final"])
            if got != float(row["expected_exact_reward"]):
                raise ValueError(f"Verifier mismatch on problem {pid} variant {name}")


def score_pair(judge, variants, better, worse):
    """Compare one better and one worse response with the verifier and with the judge

    Each mechanism returns better when it prefers the better response and
    wrong when it prefers the worse one and tie when it cannot separate them
    """
    good = variants[better]
    bad = variants[worse]
    gold = good["gold_final"]
    diff = exact_reward(good["response"],gold) - exact_reward(bad["response"],gold)
    verifier = "better" if diff > 0 else ("wrong" if diff < 0 else "tie")
    verdict = judge.compare(good["question"],good["response"],bad["response"])
    pairwise = {"A":"better","B":"wrong","TIE":"tie"}[verdict]
    return verifier,pairwise


def rate_table(records):
    """Report the better and tie and wrong rates of both mechanisms for every category"""
    table = []
    for category in CATEGORIES:
        for mechanism in ["verifier","judge"]:
            part = [r[mechanism] for r in records if r["category"] == category]
            table.append({
                "category":category,
                "mechanism":mechanism,
                "n":len(part),
                "better_rate":part.count("better")/len(part),
                "tie_rate":part.count("tie")/len(part),
                "wrong_rate":part.count("wrong")/len(part),
            })
    return pd.DataFrame(table)


def variant_table(judge, groups):
    """Average the exact reward and the group pairwise reward of every response variant

    The group reward ranks the five variants of one problem against each
    other so it is the same reward the supplied direct RLAIF policy trained on
    """
    rows = defaultdict(lambda: {"exact":[],"pairwise":[]})
    for pid,variants in groups.items():
        texts = [variants[v]["response"] for v in VARIANT_ORDER]
        rewards = judge.group_rewards(variants[VARIANT_ORDER[0]]["question"],texts)
        for v,reward in zip(VARIANT_ORDER,rewards):
            rows[v]["exact"].append(exact_reward(variants[v]["response"],variants[v]["gold_final"]))
            rows[v]["pairwise"].append(reward)
    return pd.DataFrame([
        {"variant":v,"mean_exact_reward":sum(rows[v]["exact"])/len(rows[v]["exact"]),"mean_pairwise_reward":sum(rows[v]["pairwise"])/len(rows[v]["pairwise"])}
        for v in VARIANT_ORDER
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    check_verifier(groups)

    outdir = repo_path(cfg["results_dir"])/"task5_feedback"
    (outdir/"tables").mkdir(parents=True,exist_ok=True)
    judge = PairwiseAIJudge(cfg,outdir/"judge_cache.json")
    cached_before = len(judge.cache)
    elapsed = wall_timer()
    records = []
    for pid,variants in groups.items():
        for category,(better,worse) in CATEGORIES.items():
            verifier,pairwise = score_pair(judge,variants,better,worse)
            records.append({
                "problem_id":pid,
                "category":category,
                "better":better,
                "worse":worse,
                "verifier":verifier,
                "judge":pairwise,
            })
    variants_table = variant_table(judge,groups)
    seconds = elapsed()
    write_jsonl(outdir/"diagnostic_pairs.jsonl",records)

    rates = rate_table(records)
    rates.to_csv(outdir/"tables"/"diagnostic_rates.csv",index=False)
    variants_table.to_csv(outdir/"tables"/"diagnostic_variant_rewards.csv",index=False)
    pick = lambda mechanism,category: float(rates[(rates["mechanism"] == mechanism) & (rates["category"] == category)]["better_rate"].iloc[0])
    # every pair where the final answer differs between the two responses
    outcome_pairs = ["outcome","distractor","corrupt_vs_wrong"]
    pooled = lambda mechanism: sum(pick(mechanism,c) for c in outcome_pairs)/len(outcome_pairs)
    save_json(outdir/"tables"/"diagnostic_sensitivity.json",{
        "s_reason_verifier":pick("verifier","reasoning"),
        "s_reason_judge":pick("judge","reasoning"),
        "s_outcome_verifier":pick("verifier","outcome"),
        "s_outcome_judge":pick("judge","outcome"),
        "s_outcome_pooled_verifier":pooled("verifier"),
        "s_outcome_pooled_judge":pooled("judge"),
        "judge_calls_made":len(judge.cache) - cached_before,
        "judge_seconds":seconds,
    })
    print(rates.to_string())
    print(variants_table.to_string())


if __name__ == "__main__":
    main()
