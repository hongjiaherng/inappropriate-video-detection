## Pre-requisites

- Install poetry in base python environment
- Install jupyter notebook in base python environment

## Dependencies Installation & Activation

```bash
poetry install --no-root
poetry shell
```

## IPython Kernel Initialization

```bash
jupyter kernelspec list # check if kernel is already installed, it should not be
python -m ipykernel install --name=<venv-name> --display-name <optional-display-name>
jupyter kernelspec list # check if kernel is installed
```