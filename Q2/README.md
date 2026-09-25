# Q2：局部模态缺失下的情感预测

**状态：2026-09-25 已完成一次服务器训练与附件3推理。** 实测结果见下方运行记录；验证集指标不是独立测试集指标，附件3没有真实标签。

第二轮实验设计见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)：修正缺失协议和对照条件，安排6组模型×5个种子的主实验、缺失网格及最终交付。实现与复现方法见下一节，第一轮记录保留在后续章节。

## 第二轮实现与复现（2026-09-25）

第二轮独立入口为 `run_q2_v2.py`，统计和中文结果稿入口为 `report_q2_v2.py`。第二轮使用同一Q1服务器环境，6组模型×5训练种子，每个检查点评估63条件×5诊断掩码，附件3仅在选择冻结后推理。

本机先检查协议（无需BERT权重或GPU）：

```powershell
python -m unittest Q2/test_q2_v2.py
python Q2/run_q2_v2.py --data-root ../E题数据 --output ../实验结果/Q2_v2_local_audit --audit-only
```

服务器从项目根目录提交（脚本自动先检查协议，再运行独立冒烟实验、正式训练及统计）：

```bash
cd ~/xbmu-CCQ
sbatch Q2/job_q2_v2.sbatch
```

每次作业输出独立保存于 `Q2/outputs/v2_<JOB_ID>/`，日志为 `Q2/slurm-v2-<JOB_ID>.out`。`exit.status` 为进程退出码；`TRAINING_AND_INFERENCE_COMPLETE.json` 仅表示训练与最终推理完成，`REPORT_COMPLETE.json` 才表示统计报告也已完成。冒烟结果保存在独立 `smoke/` 目录，不混入正式指标。首次启动记录环境时发现Q1环境没有pip，已改为通过标准库 `importlib.metadata` 记录安装版本；不修改Q1环境。

运行入口支持相同代码、相同配置、相同输出目录下复用已完成的缓存和检查点。中断后，在已获GPU资源且加载相同模块和libgomp设置的环境中执行：

```bash
Q1/.venv/bin/python Q2/run_q2_v2.py --data-root data --output Q2/outputs/v2_<JOB_ID> --device cuda
Q1/.venv/bin/python Q2/report_q2_v2.py --output Q2/outputs/v2_<JOB_ID>
```

`<JOB_ID>`替换为待恢复目录的编号。未完成单模型训练会从头训练该单元，已完成单元不会重复。代码/配置哈希不一致时拒绝复用；需新建输出目录。统计脚本仅读取已保存预测，不再拟合模型，也不依据诊断网格调参。单独运行统计可以用CPU，不需要占用GPU，但仍应按服务器资源使用规范执行。

主结果见 `main_summary.csv`，逐种子结果见 `main_results.csv`，配对统计见 `paired_comparisons.csv`。`missingness_grid_v2.csv` 使用各条件全部可行样本，`missingness_common_samples.csv` 使用跨所有条件的共同可行子集；两者不能混为同一评估人群。`decoding_comparison.csv` 比较原始双头、验证偏置校准双头、统一解码。`Q2论文结果与方法.md` 为中文方法与结果稿；正式附件3预测使用 `attachment3_predictions_v2.csv`。

缓存文件位于 `cache/`，包括派生池化特征和掩码清单；不随提交包发布。全量内部归档保留30个检查点、逐样本预测、掩码和日志。轻量提交包只保留必要源代码、固定配置、最终检查点、表格、图和正文材料，并检查小于50MB；不打包原始数据、BERT大权重、密钥或账户信息。

## 数据边界

- Q2 使用附件2 `aligned_50.pkl` 的 `train`（3395条）学习参数，`valid`（728条）选择模型、早停轮次和中性类别偏置。附件2的 `test` 划分不参与本脚本训练与模型选择。
- 附件3对齐版本的30个无标签PKL只在模型选择后推理。每个文件包含 `text_bert`（1×3×50）、`audio`（1×50×74）、`vision`（1×50×35），没有样本ID；CSV以原文件名去掉扩展名为 `sample_id`。
- Q1针对附件1提取的100条特征采用另一套声学和视觉维度，不进入Q2模型。附件3未对齐版也不与此对齐版模型混用。
- 原始附件2、附件3文件均只读；输出直接写到本 `Q2` 目录。

## 模型与训练方案

输入文本使用附件2预计算的768维 BERT 输出。附件3未提供该字段，脚本使用固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594` 的 `google-bert/bert-base-uncased` 对其 `text_bert` 重新编码。本机对附件2前三条重算的平均绝对差约 `5–6×10⁻⁷`，确认编码器接口一致；正式复现仍需下载同一权重。

模型将各模态的非缺失位置分别汇总为全局均值、标准差、前/中/后三段均值，再经独立的64维投影。一个可靠性门根据三模态表示和实际可用比例计算融合权重，最后通过共享128维层输出：三分类情感极性（Negative、Neutral、Positive）和限定在[-3,3]的连续情感强度。若验证集选择“无可靠性门”变体，则使用可用比例归一化权重，仍保留三个模态。

文本位置的 `0` 为填充、`101/102` 为特殊标记；`100`（`[UNK]`）在附件3按不可用文本位置处理。语音或视觉某一位置的整行特征全零时视为不可用。先只用训练集观测到的语音/视觉行拟合逐维均值和标准差，随后将标准化结果限幅到[-10,10]；零值位置由掩码排除，不混入池化统计。视觉自然失检也会被视为不可用，属于模型接口假设。

训练目标为 `SmoothL1(强度, β=0.5) + 0.7 × 加权交叉熵(极性)`，类别权重按训练集频次平方根倒数确定；AdamW学习率 `1e-3`、权重衰减 `1e-4`、批量约128、最多30轮、6轮早停、种子2026。训练集除原样外，为每条样本生成3种随机连续片段遮蔽，随机组合缺失模态，长度为内容位置的10%–70%，且至少保留一个位置。文本增强以冻结BERT特征置零近似；附件3的 `[UNK]` 会先经BERT重新编码，两者并非完全同分布，这是方法限制。

选择准则只看验证集的完整输入与固定缺失压力样本：`MAE_clean + MAE_stress + 0.6×(2−MacroF1_clean−MacroF1_stress)`。极性预测的中性类别偏置也在验证集上选择。对照实验为：无连续遮蔽增强、无可学习可靠性门、仅文本。最终模型只从两个保留三模态且进行遮蔽增强的变体中选择。报告的 Accuracy 和 Macro-F1 为三分类口径，回归报告 MAE 和 Pearson r。

缺失扫描采用“7种单模态/组合类型 × 前/中/后位置 × 15%/35%/55%连续长度”的63格协议，并记录实际平均移除位置数。附件2/3没有每个对齐位置的秒级时间戳，因此“时长”只能量化为对齐位置数和比例，不能写成秒数。验证集同时用于选择和诊断，这些扫描结果是描述性分析，不能宣称独立测试泛化结论。

## 环境与命令

本机核对环境为 Windows、Python 3.12、CPU版 PyTorch 2.8.0；依赖版本见 `requirements.txt`，固定实验设置见 `config.json`。配置中的方法参数与代码常量会相互校验；最大轮次可通过 `--epochs` 覆盖，`--device` 可选 `auto`、`cpu` 或 `cuda`。在本仓库根目录运行本机 CPU 版：

```powershell
python -m pip install -r Q2/requirements.txt
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('google-bert/bert-base-uncased', 'config.json', revision='86b5e0934494bd15c9632b12f734a8a67f723594'); hf_hub_download('google-bert/bert-base-uncased', 'model.safetensors', revision='86b5e0934494bd15c9632b12f734a8a67f723594')"
python -m unittest Q2/test_q2.py
python Q2/run_q2.py --data-root ../E题数据 --output Q2 --epochs 30 --device cpu
```

服务器已有 `~/xbmu-CCQ/Q1/.venv`，可复用它的 Python 3.10、PyTorch 2.5.1+cu121 和其余依赖；服务器环境版本与上面的 Windows 锁定版本不同，无需在 Q1 环境中运行 `pip install -r Q2/requirements.txt`。从登录节点提交独立的 GPU 训练作业，最长占用 10 小时，程序结束会自动释放资源：

```bash
cd ~/xbmu-CCQ
sbatch Q2/job_q2.sbatch
```

服务器的 ARM 环境需要先加载 scikit-learn 自带的 `libgomp`，否则导入时可能报 `cannot allocate memory in static TLS block`；`run_server.sh` 会为训练自动设置该变量并加载 CUDA 模块。`launch_server.sh` 将训练输出写入唯一的 `Q2/train_<作业ID>_<时间>.log`，退出码写入同名 `.status` 文件，`0` 表示程序正常结束。Slurm 自身输出在 `Q2/slurm-<作业ID>.out`。`--device cuda` 在 GPU 不可用时会直接报错，避免占用 GPU 作业却在 CPU 上训练。附件3推理所需的固定 revision BERT 权重必须事先放入登录节点可见的 Hugging Face 缓存；计算节点通常无法联网。

训练完成后，只用保存的模型参数重新执行附件3推理：

```powershell
python Q2/run_q2.py --data-root ../E题数据 --output Q2 --checkpoint Q2/robust_fusion.pt --device cpu
```

检查程序应生成 `metrics.json`（配置、数据哈希和验证指标）、`robust_fusion.pt`（可学习参数、归一化参数、BERT权重标识）、`missingness_grid.csv`、`validation_top_errors.csv`、`validation_error_analysis.json`、三张PNG验证图，以及 `attachment3_predictions.csv`、`attachment3_coverage.csv`、`attachment3_summary.json` 和附件3预测展示图。正式提交附件3结果使用 `attachment3_predictions.csv`，列为 `sample_id,pred_polarity,pred_intensity`；覆盖率文件是诊断材料，不是标签或得分。模型参数文件仅加载自己训练生成的可信文件。

## 已完成的服务器运行：Slurm 作业 1529871

从 `~/xbmu-CCQ` 执行 `sbatch Q2/job_q2.sbatch`，在 NVIDIA A100-PCIE-40GB 上运行，PyTorch `2.5.1+cu121`，`device=cuda`。Slurm 状态 `COMPLETED`、退出码 `0:0`、耗时 1 分 14 秒。训练集 3395 条、验证集 728 条；保存的模型从两个带遮蔽增强的三模态变体中选择 `robust_gate`，最佳轮次为 13。验证集实测如下：

| 验证条件 | Accuracy | Macro-F1 | MAE | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| 完整输入 | 0.5975 | 0.5839 | 0.6139 | 0.6143 |
| 固定缺失压力 | 0.5975 | 0.5786 | 0.6448 | 0.5670 |

服务器输出保存在 `~/xbmu-CCQ/Q2`，本机快照保存在仓库相邻的 `../实验结果/Q2_1529871`。已核对 `attachment3_predictions.csv` 有 30 条不重复预测，`missingness_grid.csv` 有 63 行实验结果，模型文件和图表均存在。详细指标、四个模型变体、数据与权重哈希以该快照的 `metrics.json` 和 `train_1529871_20260925_155602.log` 为准。

论文正文可依据实测文件填写验证集完整与缺失条件的指标、混淆矩阵和分组错误、63格缺失影响、三个消融对照、30条附件3预测汇总及样例。附件3不能当作有标签评测集，验证集诊断也不能宣称独立测试泛化结论。
