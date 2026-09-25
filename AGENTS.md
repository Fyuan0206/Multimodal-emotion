# Repository Guidelines

## Project Structure & Module Organization

This repository is the code version for the multimodal emotion project: https://github.com/Fyuan0206/Multimodal-emotion. The current tree has `Q1/`, `Q2/`, and `Q3/` work directories. `Q2/` contains missing-modality training code; `Q3/` contains explainable prediction code, tests of explanation procedures, and server job scripts. First-round Q2 snapshots live in `Q2/outputs/1529871/`; second-round Q2 results live in `Q2/outputs/v2_1529979/`. Q3 first-round results live in `Q3/outputs/run_1530329/`. Keep modeling code and code documentation in this repository. The neighboring `E题数据/` folder in the parent workspace contains supplied datasets and should remain outside this checkout. Server run outputs are also saved separately under `../实验结果/` on this machine.

As the project grows, group implementation by responsibility (for example, `src/` for reusable code, `tests/` for automated checks, and `docs/` for methods and reproduction notes). Keep raw inputs separate from generated outputs and document any required external data paths.

## Build, Test, and Development Commands

There is no repository-wide build command. Before contributing, check the current branch and working tree with `git status --short --branch`. Q2 local checks run with `python -m unittest Q2/test_q2.py` from the repository root; dependencies are in `Q2/requirements.txt`. On the configured server, submit the Q2 GPU run from `~/xbmu-CCQ` with `sbatch Q2/job_q2.sbatch`; its setup and output files are documented in `Q2/README.md`. Keep run and test commands in the relevant README when adding code.

Q2 second-round protocol checks run with `python -m unittest Q2/test_q2_v2.py`. Submit its full smoke/train/evaluate/report workflow with `sbatch Q2/job_q2_v2.sbatch`. Q3 server jobs start with `sbatch Q3/job_q3.sbatch`; reproduction commands are in `Q3/README.md`. Preserve completed run directories such as `Q3/outputs/run_1530329/`.

## Coding Style & Naming Conventions

Use names that make the modality, task, and output clear. Preserve the README's Chinese project description. For Python code, use four-space indentation, `snake_case` for functions and variables, and explicit configuration for data paths and random seeds. Avoid machine-specific absolute paths and keep preprocessing choices documented near the relevant code.

## Testing Guidelines

No tests or coverage target exist yet. Add focused tests with new analysis or model code, using the selected framework's standard test discovery and naming conventions. Cover input validation, expected shapes and labels, and reproducibility where practical. Do not claim results without recording the command and data/configuration used.

## Commit & Pull Request Guidelines

The available history contains only the `Initial commit`, so no established message convention can be inferred. Use short, imperative commit subjects (for example, `Add feature validation`) and keep each commit focused. Pull requests should summarize the modeling change, list affected files, include reproduction steps and test results, and attach plots or tables when they support the claims.

## Data and Configuration

Keep credentials and machine-local settings out of version control. Do not commit raw video or large supplied feature datasets; refer to the provided `E题数据/` workspace inputs and document any transformations that produce derived files.

## Q3 Audit and Design

Q3's staged modeling and explanation protocol is in `Q3/DESIGN.md`. Its initial read-only audit requires Python >=3.10 and NumPy >=2.0. From the repository root run `python Q3/audit_q3.py --attachment2 "../E题数据/附件2-数据集特征文件/aligned_50.pkl" --attachment4-aligned "../E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本" --output Q3/audit/recheck`. Expected output: 3395 train, 728 valid and 20 Attachment 4 rows; 4143 inventory rows total. This checks feature structure, labels, source-group overlap and file matching; it does not train, predict, decode videos or validate physical timestamps. Keep Q3 outputs separate from source data and preserve UNK as an unknown token unless a verified input convention specifies otherwise.

Q3's first completed run is documented in `Q3/RESULTS.md`, with scripts and reproduction commands in `Q3/README.md`. `Q3/job_q3.sbatch` performs a GPU smoke check and trains four architectures; subsequent named Q3 job scripts generate explanations, automatic media alignment, per-sample validation details and sensitivity checks. `python Q3/report_q3.py --run Q3/outputs/run_1530329` exports PNG/PDF/EPS figures, and `python Q3/verify_q3.py --run Q3/outputs/run_1530329 --aligned "../E题数据/附件4-可解释专项视频样本与特征文件/附件4-可解释专项视频样本与特征文件/对齐版本"` checks output integrity. Verify after generating or transferring a run; do not claim automatic CTC times are human-confirmed.
