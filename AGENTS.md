# Repository Guidelines

## Project Structure & Module Organization

This repository is the code version for the multimodal emotion project: https://github.com/Fyuan0206/Multimodal-emotion. The current tree has `Q1/`, `Q2/`, and `Q3/` work directories; `Q2/` contains the missing-modality training code, tests, configuration, and server job scripts. Keep modeling code and code documentation in this repository. The neighboring `E题数据/` folder in the parent workspace contains supplied datasets and should remain outside this checkout. Server run outputs are saved separately under `../实验结果/` on this machine.

As the project grows, group implementation by responsibility (for example, `src/` for reusable code, `tests/` for automated checks, and `docs/` for methods and reproduction notes). Keep raw inputs separate from generated outputs and document any required external data paths.

## Build, Test, and Development Commands

There is no repository-wide build command. Before contributing, check the current branch and working tree with `git status --short --branch`. Q2 local checks run with `python -m unittest Q2/test_q2.py` from the repository root; dependencies are in `Q2/requirements.txt`. On the configured server, submit the Q2 GPU run from `~/xbmu-CCQ` with `sbatch Q2/job_q2.sbatch`; its setup and output files are documented in `Q2/README.md`. Keep run and test commands in the relevant README when adding code.

Q2 second-round protocol checks run with `python -m unittest Q2/test_q2_v2.py`. Submit its full smoke/train/evaluate/report workflow with `sbatch Q2/job_q2_v2.sbatch`. The separate `Q2/run_q2_v2.py` and `Q2/report_q2_v2.py` entries, resume behavior, and output meanings are documented in the Q2 README; preserve first-round snapshots.

## Coding Style & Naming Conventions

Use names that make the modality, task, and output clear. Preserve the README's Chinese project description. For Python code, use four-space indentation, `snake_case` for functions and variables, and explicit configuration for data paths and random seeds. Avoid machine-specific absolute paths and keep preprocessing choices documented near the relevant code.

## Testing Guidelines

No tests or coverage target exist yet. Add focused tests with new analysis or model code, using the selected framework's standard test discovery and naming conventions. Cover input validation, expected shapes and labels, and reproducibility where practical. Do not claim results without recording the command and data/configuration used.

## Commit & Pull Request Guidelines

The available history contains only the `Initial commit`, so no established message convention can be inferred. Use short, imperative commit subjects (for example, `Add feature validation`) and keep each commit focused. Pull requests should summarize the modeling change, list affected files, include reproduction steps and test results, and attach plots or tables when they support the claims.

## Data and Configuration

Keep credentials and machine-local settings out of version control. Do not commit raw video or large supplied feature datasets; refer to the provided `E题数据/` workspace inputs and document any transformations that produce derived files.
