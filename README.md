This project is intended to run directly from the repository root.

Usage

1. Activate the target conda environment.
2. Change directory to the repository root.
3. Run scripts from this directory, for example:

python exp/train.py --help
python exp/test.py --help

Notes

- Do not run `pip install --user .` for this repository.
- Do not add `sys.path` patches in individual scripts.
- The `WFlib/` directory is imported directly because the repository root is on Python's module search path when you launch commands here.