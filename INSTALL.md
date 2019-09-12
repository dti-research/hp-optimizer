# Installing HP-Optimizer

## From source

We recommend that you either use virtualenv or Docker.

```bash
# Install HP-Optimizer
git clone https://github.com/dti-research/hp-optimizer
cd hp-optimizer
pip install -e .
```

### For contributors

```bash
# Install developer dependencies
pip install -r requirements-dev.txt
```

Make sure that you'll install the pre-commit hooks

```bash
cd hp-optimizer
pre-commit install
```
