# LLM Post-Training: DPO, PPO, GRPO, Safety Calibration and Feedback Sources

Controlled experiments on post-training a `Qwen/Qwen2.5-1.5B-Instruct` policy with LoRA adapters. The repository covers five studies and a NeurIPS-format report.

| Study | Topic | Folder |
|---|---|---|
| 1 | Direct Preference Optimization: standard run, beta sweep, length confounding | `task1_dpo/` |
| 2 | Proximal Policy Optimization: continuation, clipping study, KL study | `task2_ppo/` |
| 3 | Group Relative Policy Optimization: continuation, group size study, canonical versus Dr. GRPO normalization | `task3_grpo/` |
| 4 | Safety calibration: judged responses on XSTest and a manual audit | `task4_safety/` |
| 5 | Verifiable versus AI feedback on GSM8K, SVAMP transfer and a controlled diagnostic set | `task5_feedback/` |

All experiments use seed 6304. Training and generation were run on a single Colab T4 GPU.

## Repository layout

```text
README.md
requirements.txt
configs/            one yaml per study plus base.yaml and assets.yaml
common/             data, model, generation and metric helpers shared across studies
scripts/            asset download, validation and environment checks
task1_dpo/ ... task5_feedback/
data/ cached/ checkpoints/ manifests/    course assets, downloaded and not tracked
outputs/            trained adapters, not tracked
results/            JSON and CSV logs, generated responses, tables and figures
report_latex/       report source (LaTeX), figures and compiled PDF
```

## Setup

```bash
git clone https://github.com/Mohidali2005/ATML_PA2.git
cd ATML_PA2
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
python -m scripts.check_environment
```

Pinned versions: Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, Tokenizers 0.22.1.

The fixed datasets, cached rollouts and supplied checkpoints come from the asset repository
`https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets` (revision `0b350481fb03f5525a35bcdec4131bd4fe487f98`).
The base, reward and judge models are downloaded from Hugging Face at runtime.

Run every command from the repository root. Training and judging need a GPU.

## Reproducing the experiments

### Study 1: DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Study 2: PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Study 3: GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Study 4: Safety calibration

The four policies are the base model and the standard DPO, PPO and GRPO adapters from Studies 1 to 3.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.build_audit_sheet --config configs/feedback.yaml
python -m task4_safety.build_audit_sheet --config configs/feedback.yaml --remaining
```

The audit covers 60 fixed prompts, with the responses of all four policies (240 rows). The two sheets are
blind: they show only the prompt and response. The `manual_label` column is filled by hand, the two sheets and
their keys are merged into `results/task4_safety/manual_audit_sheet.csv` and `manual_audit_key.csv`, and then:

```bash
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

The committed audit files hold the final manual labels.

### Study 5: RLVR versus RLAIF

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

### Report

```bash
cd report_latex
python make_figures.py
bash build.sh
```

Figures are rebuilt from the files in `results/`.

## Implementation notes

### Corrected objective code

- `task1_dpo/dpo.py`: the DPO logit uses `beta * (policy_margin - ref_margin)`. The starter added the two margins.
- `task2_ppo/ppo.py`: the clipped surrogate takes `torch.minimum` of the two terms. The starter used `torch.maximum`.
- `task3_grpo/grpo.py`: `group_relative_advantages` normalizes with the mean and standard deviation of each prompt's own group. The starter used batch-wide statistics.

### Experimental protocol

- Short ablation forks (DPO beta sweep, PPO clipping and KL forks, GRPO normalization fork) all start from the same supplied midpoint checkpoint, with matched prompts, update budgets, seed and evaluation.
- Evaluation uses fixed prompt IDs and the decoding settings in `configs/`.
- Prompts longer than the 768 token limit are filtered, and responses are truncated from the right so the prompt stays intact.
- Peak VRAM and wall-clock time are recorded for the standard PPO and GRPO continuations.
- Safety results and feedback-source diagnostics are never used to tune the earlier policies.

## Attribution

- Starter code, configurations, fixed datasets, cached rollouts and supplied checkpoints are course material. This includes the GSM8K verifier (`task5_feedback/rlvr.py`), the pairwise judge (`task5_feedback/rlaif.py`) and the safety judge loader, prompt and parser (`task4_safety/judge_responses.py`).
- Public models: Qwen2.5-1.5B-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct and the UltraFeedback reward model `yavuz-ai/qwen2.5-1.5b-rm-ultrafeedback`.
- Libraries: PyTorch, Transformers, PEFT, TRL, bitsandbytes, pandas, matplotlib.
