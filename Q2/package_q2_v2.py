"""Build a small anonymous Q2 submission component without data, cache or secrets."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if not (args.output / "REPORT_COMPLETE.json").exists():
        raise ValueError("Report incomplete")
    files = []
    for pattern in ("*.csv", "*.json", "*.md", "environment*.txt", "figures/*.png", "figures/*.pdf", "figures/*.svg"):
        files.extend((p, "results/" + p.relative_to(args.output).as_posix()) for p in args.output.glob(pattern)
                     if p.name != "reproduced_predictions.csv")
    files.append((args.output / "robust_fusion_v2.pt", "results/robust_fusion_v2.pt"))
    source = Path(__file__).parent
    files.append((args.output / "bert_vocab.txt", "results/bert_vocab.txt"))
    for name in ("run_q2.py", "run_q2_v2.py", "report_q2_v2.py", "predict_q2_v2.py", "test_q2_v2.py", "verify_q2_v2.py", "explain_q2_errors.py"):
        files.append((source / name, "code/" + name))
    # No host/account-specific README, scheduler script, credentials or full cached datasets.
    banned = (b"scx6706", b"C:\\Users\\24019", b"C:/Users/24019", b"Fyuan0206", b"PRIVATE KEY", b"ssh.paracloud")
    manifest = []
    for path, relative in files:
        data = path.read_bytes()
        if any(word in data for word in banned):
            raise ValueError(f"Identity/private material detected: {relative}")
        manifest.append(dict(path=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    readme = """# Q2 第二轮复现与提交说明

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
"""
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, relative in files:
            archive.write(path, relative)
        archive.writestr("README.md", readme)
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    size = args.destination.stat().st_size
    if size >= 50_000_000:
        raise ValueError(f"Package exceeds 50 MB: {size}")
    print(json.dumps(dict(package=str(args.destination), bytes=size, files=len(files)+2, anonymous_scan="PASS"), ensure_ascii=False))


if __name__ == "__main__":
    main()
