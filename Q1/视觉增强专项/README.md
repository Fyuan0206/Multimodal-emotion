> 静音修正后的追溯说明：本目录实验使用的旧Q1特征已归档到 `../archive/pre_silence_fix/outputs_server_a100/`。原生视觉与文本是否保持一致见当前输出的 `silence_release_audit.json`；实验本身不使用词级池化视觉。当前主结果与写作边界以 `../FINAL_STATUS.md` 为准。

# 视觉增强专项：无人值守运行

使用OpenCV Zoo的Progressive Teacher / MobileFaceNet预训练七类表情模型。权重及官方人脸对齐代码固定至 `model_sources.json` 中的提交，运行前逐文件SHA256校验；许可证见vendor/LICENSE。没有使用生成图像或随机数据替代视频。模型来源及预处理说明见vendor/README.md和来源清单。

本次仅在独立目录增加候选视觉表示，不覆盖原15维几何、词边界、论文图片或正文。模型类别顺序：angry、disgust、fearful、happy、neutral、sad、surprised。保存原始模型分数，不宣称为校准概率或本数据的真实情绪标签。

## 后台执行内容

1. 固定使用原A100特征中保存的视频帧索引、时间和有效掩码；校验原视频哈希及帧时间。
2. 恢复同一320×320帧的5个关键点，按官方实现对齐为112×112人脸；RGB、归一化与官方模型一致。
3. 在全部有效帧进行七类表情模型推理，未检出帧保留零向量与false掩码。每条结果原子保存并记录来源哈希。
4. 对100条样本按原视频分组做外层5折、内层4折岭回归评价；标准化和参数选择仅用训练折。与原几何、质量对照及文本对照比较，保存配对描述性组自助区间。
5. 自动检查100条完整性、分组隔离、有限数值、掩码和指标。全部成功后才写results/SUCCESS.txt；失败记录错误并退出，不会把部分完成报告为成功。

原始七维分数汇总为均值/标准差，加3个质量变量得到17维；几何+表情为47维，文本+表情为785维。评分目标仍是附件1连续情感倾向标签，没有本数据集七类表情人工真值；不能把本实验写成七类情绪准确率。此为观察原几何结果后的探索性增强，同一100条数据不构成独立最终测试集，不能保证效果改善。

使用现有OpenCV DNN CPU后端，无新Python依赖。服务器分区要求申请GPU资源，但这里不虚称GPU推理；小型ONNX网络用CPU即可。脚本只在Slurm计算节点运行，SSH断开不影响任务。

## 查看结果

服务器根目录：`/home/bingxing2/home/scx6706/xbmu-CCQ/Q1/视觉增强专项/`

- `jobs.json`：实际任务号、定时检查任务号。
- `results/progress.json`：最近完成样本、计数、更新时间及错误。
- `results/features/`：100个候选特征及逐样本来源JSON。
- `results/evaluation/metrics.csv`：各模型MAE、R²、Spearman。
- `results/evaluation/paired_comparisons.csv`：配对比较与描述性区间。
- `results/validation.json`、`results/SUCCESS.txt`：校验及成功标记。
- `results/结果摘要.md`：自动生成的结果摘要。
- `results/status_1230.json`：北京时间2026-09-25 12:30起由调度器运行检查后生成，若资源排队可能延迟。检查不以提取任务成功为前提，失败也会记录。
- `../logs/expression-任务号.log`、`../logs/expression-check-任务号.log`：计算/定时检查日志。

当前会话没有定时推送聊天消息的接口；定时任务会写检查结果，不会自动发送聊天通知。本机结果也不会自动回传；之后连接服务器即可拉回。

## 失败续跑

排除日志记录的原因后，在服务器Q1根目录重新提交同一脚本：

```bash
sbatch -p scx6706_test --gres=gpu:1 -N1 --cpus-per-task=4 \
  --time=01:30:00 -J q1-expression --output=logs/expression-%j.log \
  视觉增强专项/run_server.sh
```

来源、模型和提取代码哈希一致的已完成样本会复用；其余重新计算。不要同时提交两个提取任务写同一目录。运行中不要改代码、权重或输入。若重新提交，需要更新jobs.json供以后状态检查使用。
