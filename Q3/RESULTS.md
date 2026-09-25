# Q3 第一轮结果：附件4可解释情感预测

数据与运行日期：2026-09-25。输入使用附件2/4的**对齐版本**。GPU正式训练作业1530329、解释作业1530333、自动视频词对齐作业1530339、逐样本验证导出作业1530349、敏感性作业1530358、原词元MASK作业1530359、独立推理作业1530363均已结束；逐样本导出记录为14560条预测和1455条归因。输入与结果复核见[verification.json](outputs/run_1530329/verification.json)。

## 1. 数据与模型

附件2 train 3395条用于学习参数，valid 728条按来源视频组固定分为**选择437条、诊断291条**；诊断集有96个来源组。附件2 test标签与附件4标签未用于训练或选模。附件4是20条无标签专项样本，不能计算其Accuracy等性能指标。附件2输入SHA256为`66e867aa74bc70a844e806e5571e371c9abb4a35f9e2887ce9b4d97ff2cb8fcd`。

四种候选各训练5个种子：C0仅文本时序注意力；C1三模态统计池化；C2三模态时序编码+均值池化；C3三模态时序编码+注意力池化。均使用回归强度与辅助三分类联合训练；最终类别由回归值及验证选择的中性阈值统一解码。按选择集五种子平均目标值，在三模态候选中选择**C2**，固定种子2026，epoch3、阈值τ=0.25。权重SHA256为`b7039976cb346ba4c6638ad9673856db86d2ed312d06ecacf33b12e51503b924`。

四模型的选择目标均值：C0 0.7990、C1 0.8544、**C2 0.7949**、C3 0.8076。C2与C0在该目标上的差距很小；C0仍作为文本控制组单独报告。

## 2. 验证基础性能

下表为同一291条诊断样本上**5个训练种子的指标均值**，不是最终单检查点：

| 架构 | Accuracy↑ | Macro-F1↑ | MAE↓ | Pearson r↑ |
|---|---:|---:|---:|---:|
| C0 文本 | 0.5766 | 0.5537 | 0.6315 | 0.6597 |
| C1 多模态统计池化 | 0.5739 | 0.5366 | 0.6716 | 0.6103 |
| C2 多模态时序均值 | 0.5794 | 0.5520 | 0.6312 | 0.6508 |
| C3 多模态时序注意力 | 0.5725 | 0.5365 | **0.6243** | **0.6728** |

来源：[results.json](outputs/run_1530329/results.json)与[逐样本明细](outputs/run_1530329/validation_detail/predictions.csv)。C2没有在全部指标上超过C0，C3的回归指标均值更好；选模依据是预定的MAE+0.6×(1−Macro-F1)，只在三模态候选中比较。

**最终C2种子2026**在诊断集上：Accuracy **0.6117**、Macro-F1 **0.5853**、MAE **0.6165**、Pearson **0.6650**。混淆矩阵真实类×预测类（负/中/正）为`[[56,17,13],[13,29,29],[15,26,93]]`。中性类召回29/71=0.408，明显低于负向56/86=0.651和正向93/134=0.694。负/中/正类别MAE分别0.7672、0.3036、0.6856。

固定种子2026的C2−C0诊断集差值为：Accuracy +0.0309（95%来源组重采样区间[-0.0176,0.0823]）、Macro-F1 +0.0330（[-0.0231,0.0898]）、MAE −0.0146（[-0.0506,0.0211]）、Pearson +0.0075（[-0.0325,0.0434]）。**四项区间均跨0，当前证据不足以断定多模态模型相较文本控制组有稳定性能优势。**[计算结果](outputs/run_1530329/report/report_metrics.json)。

典型高误差包括`266791$_$23`，真实−2.667但预测+0.368；`102858$_$6`，真实+2.667但预测−0.272；短文本`247318$_$3`中出现“B here standing for bad”，真实−2.333但预测中性。它们证明存在极性反转或低估，不能仅凭文本断定原因是语音/视觉干扰。来源：[逐样本预测](outputs/run_1530329/validation_detail/predictions.csv)与原始raw_text。

## 3. 解释验证与限制

主解释目标是**原预测类别的决策余量**。每样本在文本、语音、视觉三模态的8种保留组合上精确计算Shapley分解，输出有符号贡献及绝对贡献占比。解释的是指定train均值参考表示下的**模型特征作用**，不是人类情感成因。

诊断集291条的主要模态为：文本209、语音44、视觉38。对于同一模型、同一样本，删除所选主要证据相较等长度随机删除，原类别余量平均额外下降0.1493；96个来源组做2000次配对bootstrap的95%区间为[0.1248,0.1756]，291条中93.5%为正。**这个对照与证据排名都来自同一模型的遮蔽响应，因此它更接近解释程序的一致性检查，不能单独证明原始词语的因果作用或外部忠实性。**

五种子C2的主要模态完全一致率只有**37.8%**；限制在五种子预测类别一致的147条，完全一致率为**55.1%**。该不稳定性必须在论文中报告；不宜对单个样本的“唯一主要模态”作过强结论。

对291条诊断样本，采用16个train样本的已观测行向量作配对参考，主要模态与均值参考一致的比例平均**64.4%**，各参考为45.4%–90.0%。随机化回归头、再随机化融合层、再随机化编码器后，主要模态一致率依次为59.8%、20.3%、23.0%。这说明解释随参数改变而改变，但也暴露了**参考选择敏感**；这16个参考是行向量，不是完整来源视频序列。[原始结果](outputs/run_1530329/explanation/sensitivity.json)。

对209条“文本为主要模态”的诊断样本，将选中词元替换为BERT `[MASK]` 并重新编码，保持位置与A/V输入不变。原输入重编码后，活动词元平均绝对表示差为6.49×10⁻⁷，最大差2.88×10⁻⁴；模型回归输出最大差3.10×10⁻⁶。`[MASK]`后原类别余量在**89.0%**样本下降，下降均值0.4668，与特征行替换下降量的Pearson相关0.7457。原方案设想的逐元素最大差1e−4未满足，但模型输出复核差异很小；该实验应按输出复核结果和实际误差报告。[逐样本结果](outputs/run_1530329/explanation/text_mask_check/sample_results.csv)、[汇总](outputs/run_1530329/explanation/text_mask_check/summary.json)。`[MASK]`改变BERT上下文表示，所以它仍是模型输入扰动证据，并非对人类情感成因的实验。

保持排名固定，在291条诊断样本上分别请求10%/20%/30%的证据预算，主要证据删除后的原类别余量平均下降为**0.1704/0.2634/0.3318**；等长度随机对照为**0.0576/0.1159/0.1636**。实际平均覆盖2.70/4.90/6.82个特征位置；短样本或最多3个区间会使实际覆盖少于请求上限。此实验支持该排名规则下的预算趋势，仍受“用同一模型遮蔽响应排序和评价”的限制。[完整记录](outputs/run_1530329/explanation/diagnostic/budget_curves.csv)与[汇总](outputs/run_1530329/explanation/diagnostic/budget_summary.json)。

已保存[每样本Shapley分数](outputs/run_1530329/explanation/diagnostic/shapley.jsonl)、[每区间候选分数](outputs/run_1530329/explanation/diagnostic/local_candidate_scores.csv)、[诊断摘要](outputs/run_1530329/explanation/summary.json)和[种子归因明细](outputs/run_1530329/validation_detail/c2_seed_attributions.csv)。内部注意力不是提交贡献度；最终C2使用均值池化。

## 4. 附件4全量预测与证据

20条预测的极性分布：负向7、中性5、正向8。主要模态：文本15、语音3、视觉2。13号样本视觉有效行全部为零，作为质量异常保留，不当成已经确证的模态缺失。

独立检查点脚本重跑附件4，**20条极性全部一致、20条未阈值化强度最大差值为0**；[复核预测](outputs/run_1530329/attachment4_predictions_recheck.csv)。

- [20条预测与解释合并表](outputs/run_1530329/explanation/attachment4/attachment4_predictions_explanations.csv)
- [116条局部证据表](outputs/run_1530329/explanation/attachment4/attachment4_evidence.csv)
- [解释卡示例01：文本主要](outputs/run_1530329/explanation/attachment4/cards/01.md)、[05：语音主要](outputs/run_1530329/explanation/attachment4/cards/05.md)、[14：视觉主要](outputs/run_1530329/explanation/attachment4/cards/14.md)

文本词元重新编码与附件4的`text_bert`全部20条匹配。07和18号达到50位上限，末位是SEP，已按截断处理。所有WordPiece连续子词上的A/V特征行与前一个子词相同，支持按当前对齐位置寻找原文。Q1 CTC对20段原视频生成候选词时间；**116条证据全部有自动时间候选**，35张视觉关键帧图已导出。02号两条证据包含没有发音的连字符`-`，其时间由前后已对齐词的起止点包络，证据表显式记录跳过的标点数。候选时间未经人工真值校验，音视频特征如何由原素材聚合的原始元数据也未提供，因此物理时间只能标作自动候选，不能作为已验证的精确边界。

## 5. 论文图与文件

写作时优先使用[figures 目录](figures/README.md)中的中文文件名。下列路径为原始导出位置，均提供PNG、PDF、**EPS**：

| 图 | EPS |
|---|---|
| 五种子基础性能 | [validation_five_seed_metrics.eps](outputs/run_1530329/report/validation_five_seed_metrics.eps) |
| 最终诊断混淆矩阵 | [validation_confusion.eps](outputs/run_1530329/report/validation_confusion.eps) |
| 强度真实值与预测值 | [validation_intensity_scatter.eps](outputs/run_1530329/report/validation_intensity_scatter.eps) |
| 三模态贡献占比 | [modality_contribution.eps](outputs/run_1530329/explanation/diagnostic/modality_contribution.eps) |
| 等预算删除检验 | [deletion_validation.eps](outputs/run_1530329/explanation/diagnostic/deletion_validation.eps) |
| 10%/20%/30%证据预算曲线 | [evidence_budget_curve.eps](outputs/run_1530329/report/evidence_budget_curve.eps) |
| 文本主要样本局部影响 | [local_importance_01.eps](outputs/run_1530329/report/local_importance_01.eps) |
| 语音主要样本局部影响 | [local_importance_05.eps](outputs/run_1530329/report/local_importance_05.eps) |
| 视觉主要样本局部影响 | [local_importance_14.eps](outputs/run_1530329/report/local_importance_14.eps) |

## 6. 当前待补项目

1. Q1自动词时间的人工参考边界与02号包含连字符的候选片段听看复核。
2. 对原始单词直接删除后重新分词的效果、视频与给定转写的一致性人工复核尚无结果。现有词元MASK实验保持序列位置，不能替代所有原文编辑场景。
3. [Q3轻量提交包](outputs/Q3_1530329_轻量提交包.zip)约2.89MiB，包含核心代码、选定权重、20条结果、解释卡与EPS图；不含原视频和全部关键帧图。[包清单与SHA256](outputs/Q3_1530329_轻量提交包.json)。Q1特征约9.81MiB、现有Q2包约2.93MiB，加Q3约15.63MiB，仍需将其他Q1代码与正文附件纳入最终统一体积与匿名核查。

此页区分已运行的结果与待补实验；附件4无标签预测不能当作测试集精度证据。
