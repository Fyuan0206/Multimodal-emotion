# Q2：局部模态缺失下的情感预测

**状态：代码和输入接口已准备；按当前要求暂停正式训练。** 本目录不含正式验证指标、模型参数或附件3预测文件。需要运行下方命令后，才能把实测数字写入论文或提交 CSV。

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

本机核对环境为 Windows、Python 3.12、CPU版 PyTorch 2.8.0；依赖版本见 `requirements.txt`，固定实验设置见 `config.json`。配置中的方法参数与代码常量会相互校验；当前只有最大轮次允许通过 `--epochs` 覆盖。在本仓库根目录运行：

```powershell
python -m pip install -r Q2/requirements.txt
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('google-bert/bert-base-uncased', 'config.json', revision='86b5e0934494bd15c9632b12f734a8a67f723594'); hf_hub_download('google-bert/bert-base-uncased', 'model.safetensors', revision='86b5e0934494bd15c9632b12f734a8a67f723594')"
python -m unittest Q2/test_q2.py
python Q2/run_q2.py --data-root ../E题数据 --output Q2 --epochs 30
```

训练完成后，只用保存的模型参数重新执行附件3推理：

```powershell
python Q2/run_q2.py --data-root ../E题数据 --output Q2 --checkpoint Q2/robust_fusion.pt
```

检查程序应生成 `metrics.json`（配置、数据哈希和验证指标）、`robust_fusion.pt`（可学习参数、归一化参数、BERT权重标识）、`missingness_grid.csv`、`validation_top_errors.csv`、`validation_error_analysis.json`、三张PNG验证图，以及 `attachment3_predictions.csv`、`attachment3_coverage.csv`、`attachment3_summary.json` 和附件3预测展示图。正式提交附件3结果使用 `attachment3_predictions.csv`，列为 `sample_id,pred_polarity,pred_intensity`；覆盖率文件是诊断材料，不是标签或得分。模型参数文件仅加载自己训练生成的可信文件。

论文正文需在实际运行后填入：验证集完整与缺失条件的四项指标、三类混淆矩阵和分组错误、63格缺失影响、三个消融对照、30条附件3预测汇总及样例。不能把附件3当作有标签评测集，也不能把当前暂停状态写成已完成实验。
