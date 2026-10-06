from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task4_safety.generate_responses import policy_specs


def assign_policies(audit_ids, classes, policies):
    """Give every audited prompt one policy by rotating through the policy list

    The rotation restarts inside each benchmark class and the unsafe class
    starts two policies later so every policy ends up with the same number
    of audited responses
    """
    assigned = {}
    for offset,label in zip([0,2],["SAFE","UNSAFE"]):
        ids = [i for i in audit_ids if classes[i] == label]
        for position,xstest_id in enumerate(ids):
            assigned[xstest_id] = policies[(position+offset)%len(policies)]
    return assigned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"])/"task4_safety"

    audit_ids = pd.read_csv(outdir/"manual_audit_ids.csv")["xstest_id"].tolist()
    policies = list(policy_specs(cfg))
    responses = {name:{r["xstest_id"]:r for r in read_jsonl(outdir/f"generated_{name}.jsonl")} for name in policies}
    classes = {i:responses["sft"][i]["benchmark_class"] for i in audit_ids}
    assigned = assign_policies(audit_ids,classes,policies)

    # shuffling the order so the sheet never groups responses by policy
    order = pd.Series(audit_ids).sample(frac=1.0,random_state=int(cfg["seed"])).tolist()
    sheet = []
    key = []
    for audit_id,xstest_id in enumerate(order):
        row = responses[assigned[xstest_id]][xstest_id]
        sheet.append({"audit_id":audit_id,"prompt":row["prompt"],"response":row["response"],"manual_label":""})
        key.append({"audit_id":audit_id,"xstest_id":xstest_id,"policy":row["policy"],"benchmark_class":row["benchmark_class"]})

    pd.DataFrame(sheet).to_csv(outdir/"manual_audit_sheet.csv",index=False)
    pd.DataFrame(key).to_csv(outdir/"manual_audit_key.csv",index=False)
    print("Audited responses per policy:",pd.DataFrame(key)["policy"].value_counts().to_dict())
    print("Wrote the blind sheet:",outdir/"manual_audit_sheet.csv")
    print("Fill the manual_label column with one of the five labels without opening the key or any judged file")


if __name__ == "__main__":
    main()
