"""Summarize the prespecified v2 experiment without fitting additional models."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_q2_v2 as v2

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 10,
                     "axes.spines.right": False, "axes.spines.top": False})


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def csv_write(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    v2.base.write_csv(path, list(rows[0]), rows)


def metric_dict(row):
    return v2.safe_stats(row)


def bootstrap_metrics(pred, classes, valid, y, labels, sample_weights):
    """Per-bootstrap mean of condition metrics; grouped weights preserve repeated IDs."""
    p = pred.reshape(-1, len(y)).astype(np.float32)
    c = classes.reshape(p.shape)
    mask = np.broadcast_to(valid, pred.shape).reshape(p.shape).astype(np.float32)
    answer = []
    for start in range(0, sample_weights.shape[1], 100):
        w = sample_weights[:, start:start+100]
        count = mask @ w
        count[count == 0] = np.nan
        accuracy = (((c == labels).astype(np.float32) * mask) @ w) / count
        mae = ((np.abs(p - y) * mask) @ w) / count
        f1 = np.zeros_like(count)
        for k in range(3):
            tp = (((c == k) & (labels == k)).astype(np.float32) * mask) @ w
            denom = (((c == k).astype(np.float32) + (labels == k)) * mask) @ w
            f1 += np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0) / 3
        sx = (p * mask) @ w
        sy = (y * mask) @ w
        sxx = (p * p * mask) @ w
        syy = (y * y * mask) @ w
        sxy = (p * y * mask) @ w
        denom = np.sqrt(np.maximum(sxx - sx*sx/count, 0) * np.maximum(syy - sy*sy/count, 0))
        corr = np.divide(sxy - sx*sy/count, denom, out=np.full_like(denom, np.nan), where=denom > 1e-8)
        answer.append(np.stack([np.nanmean(x, axis=0) for x in (accuracy, f1, mae, corr)], axis=1))
    return np.concatenate(answer)


def select_bias(selection, y, cls):
    best = None
    for bias in [-.3, -.2, -.1, 0., .1, .2, .3]:
        pred = (selection["logits"] + np.array([0, bias, 0])).argmax(-1)
        stats = v2.batch_metrics(y, cls, selection["raw"], pred, selection["valid"])
        score = stats[0, 2] + stats[1:, 2].mean() + .6*(2-stats[0,1]-stats[1:,1].mean())
        key = (score, abs(bias), bias)
        if best is None or key < best[0]:
            best = key, bias
    return best[1]


def make_plots(output, summary, selected, targets, reg, classes, grid_mean, error_deltas):
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    variants = list(v2.VARIANTS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("mae", "macro_f1"), axes):
        for offset, condition, color in ((-.17,"clean","#4C78A8"),(.17,"stress","#F58518")):
            rows = [next(r for r in summary if r["variant"] == v and r["condition"] == condition) for v in variants]
            ax.bar(np.arange(6)+offset, [r[metric+"_mean"] for r in rows], width=.32,
                   yerr=[r[metric+"_sd"] for r in rows], label=condition, color=color, capsize=2)
        ax.set_xticks(range(6),variants)
        ax.set_ylabel(metric)
        ax.legend(frameon=False)
    fig.suptitle("Validation comparison: 5 training seeds; error bars = sample SD", fontsize=11)
    save_figure(fig,figs/"main_comparison.png")
    v2.base.plot_validation(figs, {"reg":targets["y"],"cls":targets["classes"]}, reg, classes)
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for combination in v2.COMBINATIONS:
        name="+".join(v2.MODES[j] for j in combination)
        for mi,ax in zip((2,1), axes):
            values=[]
            for rate in (.15,.35,.55):
                idx=[i for i,(m,p,r) in enumerate(v2.GRID) if m==combination and r==rate]
                values.append(np.nanmean(grid_mean[idx,mi]))
            ax.plot([.15,.35,.55],values,marker="o",label=name)
    axes[0].set(xlabel="Span fraction (aligned positions)",ylabel="MAE")
    axes[1].set(xlabel="Span fraction (aligned positions)",ylabel="Macro F1")
    axes[1].legend(fontsize=7,ncol=2)
    save_figure(fig,figs/"missing_length_effect.png")
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for mi,ax in zip((2,1),axes):
        matrix=grid_mean[:,mi].reshape(7,9)
        im=ax.imshow(matrix,aspect="auto",cmap="viridis")
        ax.set_yticks(range(7),["+".join(v2.MODES[j] for j in m) for m in v2.COMBINATIONS])
        ax.set_xticks(range(9),[f"{p[:1]} {r:.0%}" for p in v2.POSITIONS for r in (.15,.35,.55)],rotation=45)
        ax.set_title(("MAE" if mi==2 else "Macro F1")+" (common feasible samples)")
        fig.colorbar(im,ax=ax)
    save_figure(fig,figs/"missing_grid_common.png")
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].hist(np.abs(targets["y"]-reg),bins=25,color="#4C78A8")
    axes[0].set(xlabel="Absolute error without artificial masking",ylabel="Count")
    axes[1].hist(error_deltas,bins=30,color="#F58518")
    axes[1].set(xlabel="Masked minus clean absolute error",ylabel="Count")
    save_figure(fig,figs/"error_distributions.png")
    with (output/"attachment3_predictions_v2.csv").open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(v2.base.CLASS_NAMES,[sum(r["pred_polarity"]==c for r in rows) for c in v2.base.CLASS_NAMES])
    axes[0].set_ylabel("Predicted count (unlabeled)")
    axes[1].scatter(range(1,31),[float(r["pred_intensity"]) for r in rows])
    axes[1].axhline(0,color="gray",lw=1)
    axes[1].set(xlabel="Sample number",ylabel="Predicted intensity",ylim=(-3,3))
    save_figure(fig,figs/"attachment3_overview.png")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();output=args.output
    records=json.loads((output/"training_records.json").read_text(encoding="utf-8"))
    if len(records)!=30 or not (output/"TRAINING_AND_INFERENCE_COMPLETE.json").exists():
        raise ValueError("Formal experiment incomplete")
    with np.load(output/"validation_targets.npz") as data:
        targets={k:data[k] for k in data.files}
    y,labels=targets["y"],targets["classes"]
    frozen=json.loads((output/"frozen_model.json").read_text(encoding="utf-8"))
    selected=frozen["selected_variant"]
    masks=[]
    for s in range(5101,5106):
        with np.load(output/"predictions"/f"mask_{s}.npz") as f:
            masks.append({k:f[k] for k in f.files})
    valid=np.stack([m["valid"] for m in masks])
    common=valid.all(axis=(0,1))
    starts=np.stack([m["start"] for m in masks])
    ordered_starts=np.sort(starts,axis=0)
    unique_counts=1+(np.diff(ordered_starts,axis=0)!=0).sum(axis=0)
    if not common.any():
        raise ValueError("No common feasible samples; revise reporting, not the experiment")
    training_masks=[]
    for path in sorted((output/"cache").glob("train_*_mask.npz")):
        with np.load(path) as f:
            feasible=f["valid"]
            training_masks.append(dict(cache=path.stem,n=len(feasible),feasible=int(feasible.sum()),
                unchanged_due_to_infeasibility=int((~feasible).sum()),
                requested_mean=float(f["requested_rate"].mean()),
                actual_mean_span_fraction=float((f["width"][feasible]/f["length"][feasible]).mean())))
    v2.json_write(output/"mask_protocol.json",dict(common_feasible_n=int(common.sum()),
        validation_total=len(y),training_masks=training_masks,
        position_definition="interval center thirds; original ordered content indices",
        originally_observed_retained=True,infeasible_policy="flag; training view remains unchanged, diagnostic sample-condition excluded",
        selection_seed=4101,diagnostic_seeds=list(range(5101,5106)),
        length_unit="aligned content steps, not seconds",bootstrap_unit="video prefix before $_$"))
    v2.json_write(output/"figure_contract.json",dict(backend="Python matplotlib",archetype="quantitative grid",
        evidence="Compare matched models and describe missing-span effects without assuming improvement",
        figures={"main_comparison":"main_summary.csv; 5 training seeds; bars show sample SD",
                 "missing_grid_common":"missingness_common_samples.csv; 5 training x 5 mask seeds; shared feasible samples",
                 "missing_length_effect":"same common subset; average positions; descriptive curves",
                 "validation_overview":"frozen model seed2026; 728 validation samples; clean",
                 "error_distributions":"frozen model seed2026; paired errors; repeated masks are not independent samples",
                 "attachment3_overview":"30 unlabeled predictions; no test accuracy"},
        exports="PNG 300dpi, editable SVG and PDF except legacy validation panel PNG",interpretation="validation selection bias disclosed"))
    all_predictions={};main_rows=[];grid_rows=[];common_rows=[];decode_rows=[]
    for variant in v2.VARIANTS:
        all_predictions[variant]={k:[] for k in ("reg","classes","stats")}
        for seed in range(2026,2031):
            per=[]
            for s in range(5101,5106):
                with np.load(output/"predictions"/f"{variant}_{seed}_{s}.npz") as f:
                    per.append({k:f[k] for k in f.files})
            reg=np.stack([p["reg"] for p in per]);cls=np.stack([p["classes"] for p in per])
            stats=np.stack([p["stats"] for p in per])
            for k,a in (("reg",reg),("classes",cls),("stats",stats)):
                all_predictions[variant][k].append(a)
            main_rows.append(dict(variant=variant,seed=seed,condition="clean",**metric_dict(stats[0,0])))
            main_rows.append(dict(variant=variant,seed=seed,condition="stress",**metric_dict(np.nanmean(stats[:,1:],axis=(0,1)))))
            for im,s in enumerate(range(5101,5106)):
                for ig,(m,p,r) in enumerate(v2.GRID):
                    ok=valid[im,ig+1];manifest=masks[im]
                    paired_clean=v2.batch_metrics(y,labels,reg[im,0],cls[im,0],ok)[0]
                    orig=manifest["original_count"][ig][:,m]
                    removed=manifest["removed"][ig][:,m]
                    current=stats[im,ig+1]
                    grid_rows.append(dict(variant=variant,seed=seed,mask_seed=s,missing_mode="+".join(v2.MODES[j] for j in m),position=p,span_fraction=r,
                        n=int(ok.sum()),infeasible_n=int((~ok).sum()),naturally_absent_n=int(np.any(orig==0,axis=1).sum()),
                        support_negative=int((labels[ok]==0).sum()),support_neutral=int((labels[ok]==1).sum()),support_positive=int((labels[ok]==2).sum()),
                        mean_removed_steps_per_target_modality=float(removed[ok].mean()),
                        actual_removed_observation_rate=float((removed[ok]/orig[ok]).mean()),
                        actual_span_fraction=float((manifest["width"][ig][ok]/manifest["length"][ig][ok]).mean()),
                        unique_mask_realizations_mean=float(unique_counts[ig][valid[:,ig+1].all(axis=0)].mean()),
                        one_feasible_start_n=int((manifest["feasible_starts"][ig][ok]==1).sum()),
                        **metric_dict(current),delta_mae=float(current[2]-paired_clean[2]),delta_macro_f1=float(current[1]-paired_clean[1])))
                    common_stats=v2.batch_metrics(y,labels,reg[im,ig+1],cls[im,ig+1],common)[0]
                    common_rows.append(dict(variant=variant,seed=seed,mask_seed=s,grid_index=ig,n=int(common.sum()),**metric_dict(common_stats)))
            with np.load(output/"checkpoints"/f"{variant}_{seed}_selection.npz") as f:
                selection={k:f[k] for k in f.files}
            bias=select_bias(selection,y,labels)
            for rule in ("unified","dual_raw","dual_calibrated"):
                rule_stats=[];inconsistent=[]
                for im,part in enumerate(per):
                    if rule=="unified":
                        rr,cc=part["reg"],part["classes"]
                    else:
                        rr=part["raw"]
                        cc=(part["logits"]+np.array([0,bias if rule=="dual_calibrated" else 0,0])).argmax(-1)
                    rule_stats.append(v2.batch_metrics(y,labels,rr,cc,valid[im]))
                    inconsistent.append(((cc != np.sign(rr).astype(int)+1)&valid[im]).sum()/valid[im].sum())
                ss=np.stack(rule_stats)
                for condition,values in (("clean",ss[0,0]),("stress",ss[:,1:].mean((0,1)))):
                    decode_rows.append(dict(variant=variant,seed=seed,rule=rule,neutral_bias=bias if rule=="dual_calibrated" else 0,
                        condition=condition,inconsistency_rate_all_conditions=float(np.mean(inconsistent)),**metric_dict(values)))
        for key in all_predictions[variant]:
            all_predictions[variant][key]=np.stack(all_predictions[variant][key])
    csv_write(output/"main_results.csv",main_rows)
    csv_write(output/"missingness_grid_v2.csv",grid_rows)
    csv_write(output/"missingness_common_samples.csv",common_rows)
    csv_write(output/"decoding_comparison.csv",decode_rows)
    summary=[]
    for variant in v2.VARIANTS:
        for condition in ("clean","stress"):
            rows=[r for r in main_rows if r["variant"]==variant and r["condition"]==condition]
            row=dict(variant=variant,condition=condition,seeds=5)
            for key in v2.METRICS:
                values=[r[key] for r in rows]
                row[key+"_mean"]=float(np.mean(values));row[key+"_sd"]=float(np.std(values,ddof=1))
            summary.append(row)
    csv_write(output/"main_summary.csv",summary)
    # Identical group-resampling weights for all models, seeds and corruption realizations.
    groups,inverse=np.unique(targets["groups"],return_inverse=True)
    rng=np.random.default_rng(6201)
    group_weights=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000).T
    sample_weights=group_weights[inverse].astype(np.float32)
    boot={}
    for variant in v2.VARIANTS:
        cache_path=output/f"bootstrap_{variant}.npz"
        if cache_path.exists():
            with np.load(cache_path) as f:boot[variant]=f["values"]
        else:
            p=all_predictions[variant]
            boot[variant]=bootstrap_metrics(p["reg"][:,:,1:],p["classes"][:,:,1:],valid[None,:,1:],y,labels,sample_weights)
            v2.save_npz(cache_path,values=boot[variant])
        v2.log(f"bootstrap {variant} complete")
    comparisons=[]
    for left,right in (("M3","M2"),("M3","M1"),("M3","T1"),("M1","M0"),("M2","M0"),("T1","T0")):
        for index,key in enumerate(v2.METRICS):
            delta=boot[left][:,index]-boot[right][:,index]
            lower,upper=np.nanquantile(delta,[.025,.975])
            left_mean=next(r[key+"_mean"] for r in summary if r["variant"]==left and r["condition"]=="stress")
            right_mean=next(r[key+"_mean"] for r in summary if r["variant"]==right and r["condition"]=="stress")
            comparisons.append(dict(left=left,right=right,metric=key,difference=left_mean-right_mean,ci_low=float(lower),ci_high=float(upper),
                bootstrap_repeats=2000,group_count=len(groups),condition="stress grid mean",interval_excludes_zero=bool(lower>0 or upper<0)))
    csv_write(output/"paired_comparisons.csv",comparisons)
    csv_write(output/"ablation_results.csv",comparisons)
    # Detailed error attribution for the frozen seed, fixed by the plan.
    pred=all_predictions[selected];reg=pred["reg"][0];cls=pred["classes"][0]
    clean_reg=reg[0,0];clean_cls=cls[0,0];clean_error=np.abs(clean_reg-y)
    delta=np.abs(reg[:,1:]-y)-clean_error
    ok=valid[:,1:]
    score=np.where(ok,delta,-np.inf)
    order=np.argsort(score.reshape(-1))[::-1]
    selected_flat=[];seen=set()
    for flat in order:
        im,ig,i=np.unravel_index(flat,score.shape)
        if i not in seen and np.isfinite(score[im,ig,i]):
            selected_flat.append((im,ig,i,"largest_error_increase"));seen.add(i)
        if len(selected_flat)==6:break
    rng=np.random.default_rng(6202)
    for flat in rng.permutation(np.flatnonzero(ok)):
        im,ig,i=np.unravel_index(flat,score.shape)
        if i not in seen:
            selected_flat.append((im,ig,i,"fixed_random"));seen.add(i)
        if len(selected_flat)==12:break
    errors=[]
    for im,ig,i,reason in selected_flat:
        m,p,r=v2.GRID[ig];manifest=masks[im]
        errors.append(dict(sample_id=str(targets["ids"][i]),selection_reason=reason,mask_seed=5101+im,
            missing_mode="+".join(v2.MODES[j] for j in m),position=p,span_fraction=r,
            start_content_index=int(manifest["start"][ig,i]),span_steps=int(manifest["width"][ig,i]),
            true_intensity=float(y[i]),clean_intensity=float(clean_reg[i]),masked_intensity=float(reg[im,ig+1,i]),
            true_polarity=v2.base.CLASS_NAMES[int(labels[i])],clean_polarity=v2.base.CLASS_NAMES[int(clean_cls[i])],
            masked_polarity=v2.base.CLASS_NAMES[int(cls[im,ig+1,i])],error_increase=float(delta[im,ig,i]),
            raw_text=str(targets["raw_text"][i]),attribution="Observed change under synthetic masking; semantic cause requires manual review"))
    csv_write(output/"error_cases.csv",errors)
    error_summary={"by_class":{},"mask_outcomes":{},"common_feasible_n":int(common.sum()),"selected_variant":selected}
    for c,name in enumerate(v2.base.CLASS_NAMES):
        take=labels==c
        error_summary["by_class"][name]=dict(n=int(take.sum()),clean_mae=float(clean_error[take].mean()),recall=float((clean_cls[take]==c).mean()))
    clean_correct=clean_cls==labels;masked_correct=cls[:,1:]==labels
    for name,condition in (("correct_to_wrong",clean_correct & ~masked_correct),("wrong_to_correct",~clean_correct & masked_correct),
                           ("both_correct",clean_correct & masked_correct),("both_wrong",~clean_correct & ~masked_correct)):
        error_summary["mask_outcomes"][name]=int((condition&ok).sum())
    error_summary["outcome_unit"]="sample-condition realizations, not independent samples"
    v2.json_write(output/"validation_error_analysis.json",error_summary)
    selected_common=np.array([[r[k] for k in v2.METRICS] for r in common_rows if r["variant"]==selected]).reshape(5,5,63,4).mean((0,1))
    make_plots(output,summary,selected,targets,clean_reg,clean_cls,selected_common,delta[ok])
    write_paper(output,summary,comparisons,frozen,selected_common,error_summary,errors)
    v2.json_write(output/"REPORT_COMPLETE.json",dict(training_runs=30,diagnostic_predictions=150,grid_rows=len(grid_rows),
        common_feasible_n=int(common.sum()),bootstrap_repeats=2000,video_groups=len(groups),attachment3_rows=30,
        report_code_sha256=v2.base.sha256(Path(__file__))))
    v2.log("report complete")


def write_paper(output,summary,comparisons,frozen,grid,error_summary,errors):
    chosen=frozen["selected_variant"]
    lines=["# 问题2：局部连续模态缺失下的情感预测建模与验证", "",
        "> 第二轮实测结果稿。验证集同时用于选择与诊断，以下结果不代表独立测试集泛化性能。附件3无标签，只进行最终推理。", "",
        "## 数据及缺失定义", "",
        "使用附件2 aligned_50 的3395条训练样本学习模型参数，728条验证样本进行选择和诊断。归一化仅使用训练集；附件2 test 未用于本实验。输入为50个对齐位置的文本768维、语音74维、视觉35维特征。文本使用固定BERT重新编码，缺失内容替换为token 100并保留原位置，不删除token；语音/视觉全零行作为不可用观测。", "",
        "数据审计发现训练集110条、验证集15条样本的视觉模态原本完全不可用。按样本ID中视频前缀分组，训练集包含1528个来源、验证集239个来源，两个划分没有相同来源前缀。训练阶段保留原有缺失样本；人工遮蔽不可行时该增强视图保持原样，实际增强比例见mask_protocol.json。", "",
        "人工遮蔽为单个连续区间，要求每个目标模态至少移除一个原观测且至少保留一个原观测；不可行条件剔除该样本—条件并单独统计，不缩短区间。原数据的天然整模态缺失不冒充人工局部缺失。时长以内容对齐位置数和比例度量，数据没有秒级时间戳。前/中/后按缺失区间中心所属三分之一区域定义。联合缺失采用同一区间同步遮蔽。", "",
        "## 模型结构与目标函数", "",
        "每个模态以观测位置的均值、标准差及前中后三段均值组成5d维输入，投影为64维。可靠性门根据三个模态表示和可用比例生成权重；无门控对照按可用比例归一化。加权表示拼接并与加权和、覆盖率输入128维融合层，分别输出三分类logits和经3·tanh约束的强度。文本基线将其他模态输入及覆盖率置零。", "",
        "损失 L=SmoothL1(y_hat,y; beta=0.5)+0.7·WeightedCE(c_hat,c)，类别权重由训练集类别频次平方根倒数确定。AdamW学习率0.001、权重衰减0.0001、batch128，最多30轮、耐心值6。所有组每轮使用1个原始视图和3个视图槽；无增强组重复原样本，匹配增强组的每轮更新次数。5个种子为2026至2030，相同种子共享缺失视图、顺序和同形状模块初始权重。", "",
        "文本人工缺失在冻结BERT编码之前施加，完整与受损输入采用同一编码流程。原训练/验证文本没有[UNK]，因此将原生[UNK]视为可用或不可用在这两个划分上结果相同；附件3中[UNK]代表缺失仍为接口假设，未依据附件3预测分布调整规则。", "",
        "语音和视觉按训练集观测行拟合逐维均值与标准差，标准差下限0.001，标准化值裁剪至[-10,10]。模态投影为Linear→LayerNorm→GELU→Dropout(0.15)，融合层采用同类结构、Dropout(0.20)。门控logit为线性函数加log(coverage)，无观测模态屏蔽后softmax；固定对照使用coverage/sum(coverage)。训练集增强从7种非空模态组合等概率抽取，连续长度比例均匀抽取10%–70%，每种子预先生成3个视图并跨模型复用。", "",
        "## 选择与输出规则", "",
        "按固定验证掩码选择：S=MAE_clean+mean(MAE_grid)+0.6[2−F1_clean−mean(F1_grid)]。每个候选检查点先从0至0.30、步长0.05中选择中性阈值τ，再计算同口径S。强度绝对值≤τ时输出Neutral及0，否则按强度正负号输出极性；三分类头仍参与辅助训练。统一解码的代价与双头原始/校准输出比较见decoding_comparison.csv。", "",
        f"最终多模态配置按5个种子的平均选择分数确定为 **{chosen}**，固定使用种子2026的早停检查点，τ={frozen['tau']:.2f}。未挑选5个种子中最优者。", "",
        "## 基础性能与鲁棒性", "",
        "T0/T1为文本无/有增强；M0/M1为无门控多模态无/有增强；M2/M3为有门控多模态无/有增强。clean表示无额外人工缺失，仍允许原数据已有缺失。stress为63条件×5套掩码的等权平均。表中为5训练种子的均值±样本标准差。", "",
        "| 模型 | 条件 | Accuracy | Macro-F1 | MAE | Pearson r |", "|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append("| "+row["variant"]+" | "+row["condition"]+" | "+" | ".join(f"{row[k+'_mean']:.4f} ± {row[k+'_sd']:.4f}" for k in v2.METRICS)+" |")
    lines += ["", "![主实验](figures/main_comparison.png)", "", "### 消融及配对区间", "",
        "采用同一视频来源组的2000次配对bootstrap，重复遮蔽不作为独立样本。区间反映验证来源抽样不确定性，训练种子波动另以上表标准差描述。验证集曾参与选择，区间不能消除选择偏差。", "",
        "| 比较（左−右） | 指标 | 差值 | 95%区间 |", "|---|---|---:|---|"]
    for row in comparisons:
        if row["metric"] in ("mae","macro_f1"):
            lines.append(f"| {row['left']}−{row['right']} | {row['metric']} | {row['difference']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |")
    best_text=min((r for r in summary if r["variant"].startswith("T") and r["condition"]=="stress"),key=lambda r:r["mae_mean"])
    chosenrow=next(r for r in summary if r["variant"]==chosen and r["condition"]=="stress")
    lines += ["", f"所选多模态{chosen}的压力MAE为{chosenrow['mae_mean']:.4f}，文本基线中较低者{best_text['variant']}为{best_text['mae_mean']:.4f}；差值为{chosenrow['mae_mean']-best_text['mae_mean']:+.4f}。多模态优势不能从网络结构直接推定，应结合四项指标和配对比较解释。", "",
        "### 缺失类型、位置及长度", "",
        f"全网格包含7种非空模态组合、前中后3位置、15%/35%/55%三长度，每格5套固定掩码。为避免样本组成变化，以下规律使用所有诊断条件都可行的共同子集（n={error_summary['common_feasible_n']}）；全部可行样本结果另见missingness_grid_v2.csv。该共同子集偏向观测较完整、能满足区间约束的样本，不能代表全部验证样本。", ""]
    for field,values in (("模态",v2.COMBINATIONS),("位置",v2.POSITIONS),("长度",(.15,.35,.55))):
        lines += [f"按{field}分组，对其余因素等权平均：", "", "| 水平 | MAE | Macro-F1 |", "|---|---:|---:|"]
        for value in values:
            index=[i for i,(m,p,r) in enumerate(v2.GRID) if (m if field=="模态" else p if field=="位置" else r)==value]
            mean=grid[index].mean(0)
            name="+".join(v2.MODES[j] for j in value) if field=="模态" else str(value)
            lines.append(f"| {name} | {mean[2]:.4f} | {mean[1]:.4f} |")
        lines.append("")
    lines += ["上述为固定协议下的描述性关系，不把组合缺失差异单独归因于某一个模态，不保证每个条件都单调。", "",
        "![缺失长度](figures/missing_length_effect.png)", "", "![缺失网格](figures/missing_grid_common.png)", "",
        "## 验证可视化及错误分析", "", "![验证](figures/validation_overview.png)", "", "![误差](figures/error_distributions.png)", "",
        "| 真值类别 | 样本数 | clean召回率 | clean MAE |", "|---|---:|---:|---:|"]
    for name,row in error_summary["by_class"].items():
        lines.append(f"| {name} | {row['n']} | {row['recall']:.4f} | {row['clean_mae']:.4f} |")
    lines += ["", "遮蔽前后类别转移（统计单位为样本—条件实现，重复样本不独立）："+json.dumps(error_summary["mask_outcomes"],ensure_ascii=False), "",
        "按固定规则抽取6个不同样本的最大误差增量案例和6个固定随机案例，详见error_cases.csv。以下是可观测的错误变化，不能仅由变化认定语义或音视频层面的因果机制：", ""]
    for row in errors:
        lines += [f"- `{row['sample_id']}`：{row['missing_mode']}，{row['position']}，区间比例{row['span_fraction']:.0%}；真值{row['true_intensity']:.3f}，clean {row['clean_intensity']:.3f}，遮蔽后{row['masked_intensity']:.3f}，绝对误差变化{row['error_increase']:+.3f}。文本：{row['raw_text']}"]
    lines += ["", "## 附件3全量预测", "", "模型、阈值和预处理冻结后对30条无标签样本推理，CSV为attachment3_predictions_v2.csv；预测类别分布不是分类准确率。", "",
        "![附件3](figures/attachment3_overview.png)", "", "| 样本 | 极性 | 强度 |", "|---|---|---:|"]
    with (output/"attachment3_predictions_v2.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):lines.append(f"| {row['sample_id']} | {row['pred_polarity']} | {float(row['pred_intensity']):.6f} |")
    lines += ["", "## 局限与复现", "",
        "模型使用分段统计而非完整时序编码，不能据此宣称已精确定位情感转折；缺失时间为对齐步数代理；全零行和UNK的缺失解释有接口假设；只评估同步组合缺失；验证集兼用于选择，缺乏独立有标签专项测试。增强失败或模态收益不稳定的结果同样保留。", "",
        "复现需固定BERT版本、环境、掩码和配置，详见config_v2.json、environment_server.txt、environment_packages.txt、data_audit.json、frozen_model.json以及代码README。BERT权重和原始数据不放入50MB提交附件；输出文件均为派生结果，原附件保持只读。", ""]
    (output/"Q2论文结果与方法.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__":
    main()
