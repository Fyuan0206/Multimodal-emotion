# Q2 第二轮复现与提交说明

results/attachment3_predictions_v2.csv 为30条专项测试预测；模型参数为results/robust_fusion_v2.pt。
结果正文与图表见 results/Q2论文结果与方法.md。该包仅为Q2组件，合并其他问题附件后仍须检查全部附件总计不超过50MB。

## 环境与外部依赖

训练环境精确包版本见results/environment_packages.txt；Python、PyTorch与CUDA信息见results/environment_server.txt。
主要依赖：numpy、scipy、scikit-learn、matplotlib、torch、transformers、huggingface-hub。
原始附件与BERT权重需另外准备，不包含在此包中。固定模型为google-bert/bert-base-uncased，revision为86b5e0934494bd15c9632b12f734a8a67f723594；权重哈希见results/frozen_model.json。
可在有网络的准备环境通过huggingface_hub.hf_hub_download下载该revision的config.json和model.safetensors，推理环境仅使用本地缓存。
请使用自己生成或可信来源的checkpoint和PKL；完整实验使用训练/验证数据，推理命令只读取附件3。

## 仅复现最终预测

在本包根目录运行（data-root指向包含原始中文附件子目录的父目录）：

```bash
python code/predict_q2_v2.py --checkpoint results/robust_fusion_v2.pt --data-root /path/to/data --output reproduced_predictions.csv --device cpu
```

GPU可用时将cpu改为cuda。输出极性与强度按冻结阈值统一解码，不能依据附件3重新调阈值。

## 完整实验

```bash
python code/test_q2_v2.py
python code/run_q2_v2.py --data-root /path/to/data --output /path/to/new_output --device cuda --smoke
python code/run_q2_v2.py --data-root /path/to/data --output /path/to/formal_output --device cuda
python code/report_q2_v2.py --output /path/to/formal_output
python code/verify_q2_v2.py --output /path/to/formal_output
```

冒烟与正式输出必须分目录。正式运行包含6模型×5种子、固定缺失评估和附件3推理；统计脚本包含2000次来源分组配对bootstrap。
完整内部归档中的缓存、其他检查点及全部逐样本诊断预测没有打包；通过完整运行可重建。F1为三分类Macro-F1，验证集参与选择，不能把验证指标写成独立测试指标。无秒级时间戳，缺失时长使用对齐位置及比例。
