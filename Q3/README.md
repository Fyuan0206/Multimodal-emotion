# Q3：可解释性多模态情感预测

当前状态：已完成第一轮Q3训练、解释验证、20条附件4预测与自动证据定位。详见[实际结果](RESULTS.md)和[分析](ANALYSIS.md)。视频词时间是自动候选，人工边界验证仍待补。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `Q3/*.py`、`job_q3*.sbatch` | 当前可运行代码和服务器作业脚本 |
| [`DESIGN.md`](DESIGN.md) | 题目映射、候选模型、解释与验证协议 |
| [`audit/initial/`](audit/initial/audit.json) | 输入审计 |
| [`outputs/run_1530329/`](outputs/run_1530329/) | 正式训练、解释、验证明细和检查点 |
| [`figures/`](figures/README.md) | 论文用图、附件4预测表和示例解释卡 |
| `logs/` | 后续 Slurm 日志写入处 |
| `outputs/Q3_1530329_轻量提交包.zip` | 竞赛精简包 |

仓库外的日期快照：`../实验结果/Q3_1530329_20260925/`（压缩包与解压查阅副本）。其中完整快照与整理前的本目录文件一致；`ANALYSIS.md` 在快照之后写入，只在本仓库。

- [20条预测与解释汇总](figures/03_附件4预测与解释/attachment4_predictions_explanations.csv)
- [116条证据定位](figures/03_附件4预测与解释/attachment4_evidence.csv)
- [EPS等图表](figures/README.md)
- [Q3轻量提交包](outputs/Q3_1530329_轻量提交包.zip)，约2.89MiB；与Q1/Q2材料合并时仍须检查竞赛总附件大小。

## 重现初步审计

Python≥3.10、NumPy≥2.0。NumPy 1.x环境可能无法读取提供的NumPy 2序列化数据；请使用独立兼容环境，保留原PKL。脚本只读取用户提供的可信PKL。

在仓库根目录执行（PowerShell或Bash都可使用这一行）：

```text
python Q3/audit_q3.py --attachment2 "../E题数据/附件2-数据集特征文件/aligned_50.pkl" --attachment4-aligned "../E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本" --output Q3/audit/recheck
```

已运行环境：Python和NumPy精确版本见audit.json；本次NumPy为2.3.5。预期结果：无形状、有限值、标签范围/符号错误；train/valid来源组无重叠；2种规范化文本交叉；附件4 ID=13内容位置视觉全零。20个特征与视频按文件名一一对应。

此脚本未训练模型、未使用附件2 test标签、未预测附件4，也未验证视频能否解码、语义配对或秒级时间映射。全零特征指示不等于已确认模态缺失。

## 复现实验

服务器项目目录 `~/xbmu-CCQ`，Q1已提供的Python环境和离线预训练模型位于 `Q1/.venv`、`Q1/models`；输入路径见各作业脚本。Slurm分区`scx6706_test`、1张GPU。正式作业依次执行：

```bash
sbatch Q3/job_q3.sbatch
# 获得正式作业ID，当前归档为1530329
sbatch Q3/job_q3_explain.sbatch
sbatch Q3/job_q3_align.sbatch
sbatch Q3/job_q3_detail.sbatch
sbatch Q3/job_q3_sensitivity.sbatch
sbatch Q3/job_q3_text_mask.sbatch
sbatch Q3/job_q3_recheck.sbatch
```

后六个作业脚本引用本次归档`run_1530329`；重跑时应先将脚本中的运行目录改为新训练作业ID。服务器训练与解释输出已复制到 `Q3/outputs/run_1530329/`，环境包版本在 `environment_packages.txt`，模型选择和归一化配置在 `selection.json`、`protocol.json`。后续作业的 Slurm 标准输出写入 `Q3/logs/`；本次已完成作业的日志也保存在该运行目录内。

原始数据须保持于同一对齐版本，附件4只作冻结后的推理。`run_q3.py`的 `--smoke` 单批检查不产生附件4预测。独立检查点推理可执行：

```bash
python Q3/predict_q3.py --aligned "data/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本" --run Q3/outputs/run_1530329 --output Q3/outputs/run_1530329/attachment4_predictions_recheck.csv --device cuda --compare Q3/outputs/run_1530329/attachment4_predictions.csv
```

论文图在有Matplotlib的本地环境执行 `python Q3/report_q3.py --run Q3/outputs/run_1530329`，自动导出PNG/PDF/EPS。完整性检查需NumPy 2兼容环境：

```bash
python Q3/verify_q3.py --run Q3/outputs/run_1530329 --aligned "../E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"
```

Q1的预训练BERT/CTC模型文件没有收入小型结果包；如需重做原文MASK或视频词时间，需要按Q1环境说明准备模型。脚本直接读取可信的竞赛PKL。
