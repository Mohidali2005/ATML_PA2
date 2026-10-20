import csv
import json
import math
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent/"results"
OUT = HERE/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10.5,"axes.titlesize":11,"axes.labelsize":10.5,"legend.fontsize":9.5,"axes.spines.top":False,"axes.spines.right":False})


def read_csv(path):
    with open(path,encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


def paired_reward_change(path):
    """Return the mean and standard error of the per prompt reward change against the start policy"""
    rows = read_csv(path)
    change = [float(r["reward_change"]) for r in rows]
    return statistics.mean(change),statistics.stdev(change)/math.sqrt(len(change))


def task1_figure():
    """Plot the beta sweep with standard errors and the per stratum accuracy"""
    tables = RESULTS/"task1_dpo"/"tables"
    sweep = read_csv(tables/"beta_sweep.csv")
    fig,axes = plt.subplots(1,2,figsize=(7.5,2.6))
    forks = [r for r in sweep if r["name"].startswith("beta_")]
    std = [r for r in sweep if r["name"]=="standard"][0]
    betas = [float(r["beta"]) for r in forks]

    rewards = []
    errors = []
    for r in forks+[std]:
        gen = read_csv(tables/f"{r['name']}_generations.csv")
        values = [float(g["reward_score"]) for g in gen]
        rewards.append(statistics.mean(values))
        errors.append(statistics.stdev(values)/math.sqrt(len(values)))
    axes[0].errorbar(betas,rewards[:3],yerr=errors[:3],marker="o",color="darkorange",capsize=3,label="short fork")
    axes[0].errorbar([0.1],[rewards[3]],yerr=[errors[3]],marker="*",markersize=11,color="steelblue",capsize=3,linestyle="none",label="standard epoch")
    axes[0].set_xscale("log")
    axes[0].set_xticks(betas)
    axes[0].set_xticklabels(["0.03","0.1","0.3"])
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("held out reward score")
    axes[0].set_title("(a) reward is flat in beta")
    axes[0].legend(loc="lower right")

    n = 290
    accuracy = [float(r["preference_accuracy"]) for r in forks]
    se = math.sqrt(0.25/n)
    axes[1].errorbar(betas,accuracy,yerr=se,marker="o",color="darkorange",capsize=3)
    axes[1].errorbar([0.1],[float(std["preference_accuracy"])],yerr=se,marker="*",markersize=11,color="steelblue",capsize=3,linestyle="none")
    axes[1].axhline(0.5,color="gray",linestyle="--",linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xticks(betas)
    axes[1].set_xticklabels(["0.03","0.1","0.3"])
    axes[1].set_ylim(0.4,0.6)
    axes[1].set_xlabel("beta")
    axes[1].set_ylabel("held out preference accuracy")
    axes[1].set_title("(b) accuracy sits at chance")

    fig.tight_layout()
    fig.savefig(OUT/"task1_dpo_summary.png",dpi=220)
    plt.close(fig)


def task2_figure():
    """Plot clip fraction by epsilon, paired reward change by KL coefficient, and critic quality"""
    tables = RESULTS/"task2_ppo"/"tables"
    fig,axes = plt.subplots(1,3,figsize=(9.4,2.5))

    cached = read_csv(tables/"clipping_cached_batch.csv")
    stress = read_csv(tables/"clipping_drift_stress.csv")
    epsilons = [0.05,0.2,0.5]
    step_zero = [float([r for r in cached if float(r["epsilon"])==e and int(r["step"])==0][0]["clip_fraction"]) for e in epsilons]
    drift = [float([r for r in stress if float(r["epsilon"])==e][0]["clip_fraction"]) for e in epsilons]
    x = np.arange(3)
    axes[0].bar(x-0.19,step_zero,width=0.38,color="steelblue",label="cached batch")
    axes[0].bar(x+0.19,drift,width=0.38,color="firebrick",label="drift stress test")
    for i in range(3):
        if step_zero[i]>0.02:
            axes[0].text(x[i]-0.19,step_zero[i]+0.003,f"{step_zero[i]:.3f}",ha="center",fontsize=8.5)
        if drift[i]>0.02:
            axes[0].text(x[i]+0.19,drift[i]+0.003,f"{drift[i]:.3f}",ha="center",fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["0.05","0.20","0.50"])
    axes[0].set_xlabel("clip epsilon")
    axes[0].set_ylabel("share outside clip range")
    axes[0].set_ylim(0,0.17)
    axes[0].set_title("(a) tokens clipped")
    axes[0].legend(loc="upper right")

    betas = [0.0,0.1,0.2]
    stats = [paired_reward_change(tables/f"eps_0.2_kl_{b}_vs_midpoint.csv") for b in betas]
    axes[1].errorbar(range(3),[s[0] for s in stats],yerr=[s[1] for s in stats],marker="o",color="steelblue",capsize=4)
    axes[1].axhline(0,color="gray",linestyle="--",linewidth=1)
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["0.0","0.1","0.2"])
    axes[1].set_xlabel("KL coefficient")
    axes[1].set_ylabel("change vs midpoint")
    axes[1].set_ylim(-0.5,0.4)
    axes[1].set_title("(b) reward change")

    curve = read_json(tables/"standard_training_curve.json")
    update = [c["update"] for c in curve]
    axes[2].plot(update,[c["critic_explained_variance"] for c in curve],marker="o",markersize=3,color="firebrick")
    axes[2].axhline(0,color="gray",linestyle="--",linewidth=1)
    axes[2].set_xlabel("update")
    axes[2].set_ylabel("explained variance")
    axes[2].set_title("(c) critic quality")
    fig.tight_layout()
    fig.savefig(OUT/"task2_ppo_summary.png",dpi=220)
    plt.close(fig)


def task3_figure():
    """Plot group size effects by prompt difficulty and the per completion gradient norm by length"""
    tables = RESULTS/"task3_grpo"/"tables"
    study = read_csv(tables/"group_size_study.csv")
    fig,axes = plt.subplots(1,3,figsize=(9.4,2.5))
    colors = {"all":"black","hard":"firebrick","medium":"darkorange","easy":"seagreen"}
    for difficulty,color in colors.items():
        rows = sorted([r for r in study if r["difficulty_bin"]==difficulty],key=lambda r:int(r["k"]))
        ks = [int(r["k"]) for r in rows]
        style = "--" if difficulty=="all" else "-"
        axes[0].plot(ks,[float(r["mean_group_std_shuffled"]) for r in rows],marker="o",color=color,linestyle=style,label=difficulty)
        axes[1].plot(ks,[float(r["weak_rate_shuffled"]) for r in rows],marker="o",color=color,linestyle=style)
    for ax in axes[:2]:
        ax.set_xscale("log",base=2)
        ax.set_xticks([2,4,8])
        ax.set_xticklabels(["2","4","8"])
        ax.set_xlabel("group size K")
    axes[0].set_ylabel("mean group reward std")
    axes[0].set_title("(a) signal strength")
    axes[0].legend(loc="lower right")
    axes[1].set_ylabel("share with std below 0.25")
    axes[1].set_title("(b) weak groups")

    for name,color,label in [("norm_grpo","steelblue","canonical GRPO"),("norm_dr_grpo","firebrick","Dr. GRPO")]:
        rows = [r for r in read_csv(tables/f"{name}_sequence_gradients.csv") if r["truncated"]=="False"]
        length = [float(r["response_length"]) for r in rows]
        grad = [float(r["grad_norm"])/float(r["abs_advantage"]) for r in rows]
        axes[2].scatter(length,grad,s=14,color=color,alpha=0.8,label=label)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("completion length in tokens")
    axes[2].set_ylabel("gradient per unit advantage")
    axes[2].set_title("(c) gradient by length")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT/"task3_grpo_summary.png",dpi=220)
    plt.close(fig)


def task4_figure():
    """Plot the category level refusal rate for every policy and the judge versus manual confusion"""
    tables = RESULTS/"task4_safety"/"tables"
    counts = read_csv(tables/"category_label_counts.csv")
    policies = ["sft","dpo","ppo","grpo"]
    categories = []
    for r in counts:
        key = (r["benchmark_class"],r["type"])
        if key not in categories:
            categories.append(key)
    grid = np.zeros((len(categories),len(policies)))
    for i,(cls,cat) in enumerate(categories):
        for j,policy in enumerate(policies):
            row = [r for r in counts if r["policy"]==policy and r["type"]==cat][0]
            total = sum(int(row[k]) for k in ["SAFE_ANSWER","JUSTIFIED_REFUSAL","UNSAFE_COMPLIANCE","OVER_REFUSAL","AMBIGUOUS"])
            grid[i,j] = (int(row["JUSTIFIED_REFUSAL"])+int(row["OVER_REFUSAL"]))/total
    fig,axes = plt.subplots(1,2,figsize=(8.6,3.4),gridspec_kw={"width_ratios":[1.15,1]})
    image = axes[0].imshow(grid,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(["SFT","DPO","PPO","GRPO"])
    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels([("safe " if cls=="SAFE" else "")+cat.replace("contrast_","unsafe ").replace("_"," ") for cls,cat in categories],fontsize=8.5)
    axes[0].axhline(9.5,color="black",linewidth=1)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            axes[0].text(j,i,f"{grid[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if grid[i,j]>0.6 else "black")
    axes[0].set_title("(a) fraction refused by category")
    axes[0].spines[:].set_visible(False)

    labels = ["SAFE_ANSWER","JUSTIFIED_REFUSAL","OVER_REFUSAL","AMBIGUOUS"]
    short = ["safe\nanswer","justified\nrefusal","over\nrefusal","ambiguous"]
    confusion = {r["manual_label"]:r for r in read_csv(tables/"audit_confusion.csv")}
    matrix = np.array([[int(confusion[m][j]) for j in ["SAFE_ANSWER","JUSTIFIED_REFUSAL"]] for m in labels])
    axes[1].imshow(matrix,cmap="Oranges",vmin=0,vmax=26,aspect="auto")
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(["safe answer","justified refusal"])
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels(short)
    axes[1].set_xlabel("AI judge label")
    axes[1].set_ylabel("manual label")
    for i in range(4):
        for j in range(2):
            axes[1].text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=11,color="white" if matrix[i,j]>14 else "black")
    axes[1].set_title("(b) judge vs manual")
    axes[1].spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/"task4_safety_summary.png",dpi=220)
    plt.close(fig)


def task5_figure():
    """Plot better, tie, and wrong preference rates for the verifier and the judge by perturbation type"""
    rows = read_csv(RESULTS/"task5_feedback"/"tables"/"diagnostic_rates.csv")
    categories = ["reasoning","outcome","filler","distractor","corrupt_vs_wrong"]
    names = ["reasoning\ncorruption","wrong\nfinal","persuasive\nfiller","gold\ndistractor","corrupt vs\nwrong final"]
    fig,axes = plt.subplots(1,2,figsize=(9.4,2.5),sharey=True)
    for ax,mechanism,title in zip(axes,["verifier","judge"],["(a) exact verifier","(b) pairwise AI judge"]):
        better = [float([r for r in rows if r["category"]==c and r["mechanism"]==mechanism][0]["better_rate"]) for c in categories]
        tie = [float([r for r in rows if r["category"]==c and r["mechanism"]==mechanism][0]["tie_rate"]) for c in categories]
        wrong = [float([r for r in rows if r["category"]==c and r["mechanism"]==mechanism][0]["wrong_rate"]) for c in categories]
        x = np.arange(len(categories))
        ax.bar(x,better,color="seagreen",label="better response")
        ax.bar(x,tie,bottom=better,color="gray",label="tie")
        ax.bar(x,wrong,bottom=[b+t for b,t in zip(better,tie)],color="firebrick",label="wrong response")
        ax.set_xticks(x)
        ax.set_xticklabels(names,fontsize=8.5)
        ax.set_title(title)
    axes[0].set_ylabel("fraction of 20 pairs")
    axes[1].legend(loc="upper center",bbox_to_anchor=(0.5,-0.32),ncol=3,frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/"task5_diagnostics.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def appendix_trajectories():
    """Plot the per update trajectories of the standard PPO and GRPO continuations"""
    setups = {
        "ppo":("task2_ppo",[("reward","reward"),("kl","sampled KL"),("entropy","entropy"),("response_length","response length"),("policy_loss","policy loss"),("value_loss","value loss"),("clip_fraction","clip fraction"),("policy_grad_norm","policy gradient norm"),("critic_explained_variance","explained variance")]),
        "grpo":("task3_grpo",[("reward","reward"),("kl","sampled KL"),("entropy","entropy"),("response_length","response length"),("policy_loss","policy loss"),("grad_norm","gradient norm"),("clip_fraction","clip fraction"),("group_reward_std","within group reward std"),("uninformative_fraction","uninformative group fraction")]),
    }
    for task,(folder,keys) in setups.items():
        curve = read_json(RESULTS/folder/"tables"/"standard_training_curve.json")
        fig,axes = plt.subplots(3,3,figsize=(10.5,6.4))
        for ax,(key,label) in zip(axes.flat,keys):
            ax.plot([c["update"] for c in curve],[c[key] for c in curve],marker="o",markersize=3,color="steelblue")
            ax.set_xlabel("update")
            ax.set_ylabel(label)
        fig.tight_layout()
        fig.savefig(OUT/f"appendix_{task}_standard_trajectories.png",dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    task1_figure()
    task2_figure()
    task3_figure()
    task4_figure()
    task5_figure()
    appendix_trajectories()
    print("done")
