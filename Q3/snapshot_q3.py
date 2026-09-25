"""Save and verify a dated copy of the current Q3 code and full local results."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return sha256(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q3", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = args.q3.resolve()
    destination = args.destination.resolve()
    if not source.is_dir() or destination.is_relative_to(source):
        raise ValueError("Invalid source or destination")
    if destination.exists():
        raise FileExistsError(f"Snapshot destination already exists: {destination}")
    expected = source / "outputs" / "run_1530329" / "verification.json"
    verification = json.loads(expected.read_text(encoding="utf-8"))
    if not verification.get("passed") or verification["prediction_rows"] != 20:
        raise ValueError("Q3 run has not passed verification")
    destination.mkdir(parents=True)
    full = destination / "Q3_完整快照.zip"
    files = sorted(path for path in source.rglob("*") if path.is_file()
                   and "__pycache__" not in path.parts and path.suffix != ".pyc")
    manifest = {"schema": "q3-local-snapshot-v1", "run_id": 1530329,
                "source": "Multimodal-emotion/Q3", "entry_count": len(files),
                "files": []}
    with ZipFile(full, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            name = Path("Q3") / path.relative_to(source)
            archive.write(path, name.as_posix())
            manifest["files"].append({"path": name.as_posix(), "bytes": path.stat().st_size,
                                      "sha256": file_hash(path)})
    with ZipFile(full) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files):
            raise ValueError("Full snapshot ZIP failed integrity test")
        for entry in manifest["files"]:
            with archive.open(entry["path"]) as stream:
                if sha256(stream) != entry["sha256"]:
                    raise ValueError(f"ZIP content mismatch: {entry['path']}")
    lightweight_source = source / "outputs" / "Q3_1530329_轻量提交包.zip"
    lightweight = destination / lightweight_source.name
    shutil.copy2(lightweight_source, lightweight)
    if file_hash(lightweight) != file_hash(lightweight_source):
        raise ValueError("Lightweight package copy failed")
    manifest["archives"] = {
        full.name: {"bytes": full.stat().st_size, "sha256": file_hash(full)},
        lightweight.name: {"bytes": lightweight.stat().st_size,
                           "sha256": file_hash(lightweight)}}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                      indent=2) + "\n", encoding="utf-8")
    (destination / "SHA256SUMS.txt").write_text("".join(
        f"{values['sha256']}  {name}\n" for name, values in manifest["archives"].items()),
        encoding="utf-8")
    (destination / "README.md").write_text(
        "# Q3 当前结果快照\n\n"
        "本目录是 Q3 作业 1530329 的本地保存快照。`Q3_完整快照.zip` 包含当前 Q3 "
        "代码、审计、全部本地运行结果、关键帧及 EPS 图片；`Q3_1530329_轻量提交包.zip` "
        "为竞赛附件准备的精简版本。\n\n"
        "`manifest.json` 列出完整快照中每个文件的大小和 SHA256；`SHA256SUMS.txt` "
        "记录两个压缩包的校验值。生成后已逐项解压重算校验值。\n\n"
        "主要报告：完整快照内 `Q3/RESULTS.md`；正式预测和解释："
        "`Q3/outputs/run_1530329/explanation/attachment4/`。\n\n"
        "视频原始附件及预训练 BERT/CTC 权重没有复制到此快照；结果中的自动词时间尚无人工参考边界。\n",
        encoding="utf-8")
    print(json.dumps({"destination": str(destination),
                      "entry_count": len(files), "archives": manifest["archives"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
