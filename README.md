# Burgers PINNs: Standard and Artificial Viscosity

**Yanghao Chen · Tongji University**

[Install](#install) · [Quick execution check](#quick-execution-check) · [Train and evaluate](#train-and-evaluate) · [Methods](docs/methods.md) · [Results](docs/results.md) · [Reproducibility](docs/reproducibility.md) · [Research homepage](https://cyhcyh070126-bot.github.io/)

A PyTorch study of shock formation and propagation in a one-dimensional Burgers
Riemann problem. Two implementations share the same coordinate network and
training data: a standard strong-form PINN and a PINN with constant global
artificial viscosity.

**First run:** after [installing](#install), run this from the repository root:

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/first-run --device cpu --smoke-test
```

This checks training and saves a checkpoint using four optimizer updates.
It needs no downloaded dataset, pretrained model, GPU, or plotting fonts.
Look for `Completed` in the terminal and `completed: true` in
`outputs/first-run/result.json`. This small execution check does not reproduce
the research results below. Use a new output-directory name when running again.

## Prediction examples

| Standard PINN | Artificial-viscosity PINN, $\nu=0.001$ |
| :---: | :---: |
| ![Standard PINN prediction](assets/gifs/standard-pinn.gif) | ![Artificial-viscosity PINN prediction](assets/gifs/artificial-viscosity-pinn.gif) |

These preserved research animations compare predictions with the **inviscid
entropy solution**. Artificial viscosity changes the PDE, so its agreement with
that reference is a regularization diagnostic. The GIFs and results below are
from archived experiments, not the quick-start execution check.

## Problem and training data

The dimensionless domain is $x\in[-1,1]$, $t\in[0,1]$. The initial state is 1
to the left of zero and 0 to the right. Boundary values remain 1 at the left
boundary and 0 at the right. The inviscid entropy shock moves along $x=t/2$.

No downloaded dataset is required. The code samples fixed uniform pools of
4,096 initial-condition points, 4,096 boundary-condition points, and 16,384
interior collocation points. Each epoch shuffles and sweeps these pools.
The analytical shock solution is used only for diagnostics and plots.

![Standard PINN training-point layout](assets/figures/standard-training-points.png)

[Original sampling-layout PDF](assets/pdf/standard-training-points.pdf). The
red shock path is a reference overlay, not an interior training constraint.

## Network and loss

The network maps $(x,t)$ to a scalar $u_\theta(x,t)$ through **eight hidden
layers of width 64**, Tanh activations, and a linear output. It has **29,377
parameters**, initialized with Xavier-normal weights and zero biases.

| Method | Residual used in training |
| --- | --- |
| Standard PINN | $r=u_t+u u_x$ |
| Global artificial viscosity | $r=u_t+u u_x-\nu u_{xx}$, default $\nu=10^{-3}$ |

Automatic differentiation computes the derivatives. Both methods minimize

```math
\mathcal{L}=\mathrm{MSE}_{\mathrm{IC}}+\mathrm{MSE}_{\mathrm{BC}}+\mathrm{mean}(r^2)
```

All three terms have weight 1. Viscosity is fixed rather than learned or
adapted. The analytical solution is excluded from optimizer losses and
checkpoint selection. See [the method description](docs/methods.md) for the
full equations and the limitations of a smooth strong-form network at a shock.

## Training strategy

| Setting | Shared default |
| --- | --- |
| Seed | 1234 |
| Batch | 128 IC + 128 BC + 512 PDE points |
| Epochs / updates | 1,000 / 32,000 |
| Optimizer | Adam |
| Learning rate | Per-update cosine decay from `1e-3` to `1e-5` |
| Training-history diagnostic | Relative L2 on a 401 × 81 grid |
| Final evaluation | 1,001 × 101 grid |
| Saved model | Final update, without analytical-error selection |

![Archived standard PINN training history](assets/figures/standard-training-history.png)

[Original training-history PDF](assets/pdf/standard-training-history.pdf).

## Recorded results

| Diagnostic against the inviscid reference | Standard PINN | Artificial viscosity, $\nu=10^{-3}$ |
| --- | ---: | ---: |
| Relative L2 error | 0.158199 | 0.0248065 |
| Mean absolute error | 0.0594167 | 0.00194412 |
| Maximum absolute error | 0.521592 | 0.554839 |
| Mean absolute shock-position error | 0.00112313 | 0.000225788 |
| Mean shock thickness | 0.410108 | 0.00892251 |

These are archived fixed-grid diagnostics, not new measurements or a ranking
of solvers for the same PDE. The local archive does not contain the original
trained checkpoints. [Results and provenance](docs/results.md) explain the
metric definitions, source reports, and available figures.

![Archived standard PINN profiles](assets/figures/standard-solution-comparison.png)

[Original profile-comparison PDF](assets/pdf/standard-solution-comparison.pdf).

## Install

The verified installation uses **Python 3.12** and **PyTorch 2.10.0** in a new
virtual environment. The commands below select the CPU build so the first run
does not depend on CUDA. Clone the repository and run all commands from its root:

```bash
git clone https://github.com/cyhcyh070126-bot/burgers-pinn.git
cd burgers-pinn
```

On **Windows PowerShell**:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Activate with `.\.venv\Scripts\Activate.ps1` to use the shorter `python`
commands below. If PowerShell blocks activation, replace `python` in those
commands with `.\.venv\Scripts\python.exe`; no system policy changes are needed.
For example:

```powershell
.\.venv\Scripts\python.exe -m burgers_pinn.train --method standard --output-dir outputs/first-run --device cpu --smoke-test
```

On **Linux**:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

For CUDA training or another platform, install the compatible PyTorch build
using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
before installing `requirements.txt`. The independent installation audit was
performed on Windows CPU; other platforms have not been tested in this release.

The original plot style requires locally installed **Times New Roman** faces
and uses STIX math text; font files are not bundled. Use `--skip-plots` when
you only need numerical training/evaluation without those fonts.

## Quick execution check

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/smoke-standard --device cpu --smoke-test
python -m burgers_pinn.train --method viscosity --output-dir outputs/smoke-viscosity --device cpu --smoke-test
python -m burgers_pinn.evaluate --checkpoint outputs/smoke-standard/checkpoint_final.pt --output-dir outputs/smoke-standard-evaluation --device cpu
python -m burgers_pinn.evaluate --checkpoint outputs/smoke-viscosity/checkpoint_final.pt --output-dir outputs/smoke-viscosity-evaluation --device cpu
```

Each training command writes `checkpoint_final.pt`, `training_history.csv`,
`result.json`, and `evaluation_fields.npz`. Each evaluation command reloads that
checkpoint and writes a new `result.json` and `evaluation_fields.npz`.
Successful runs record `completed: true`; smoke runs and evaluations of smoke
checkpoints also record `smoke_test: true`. They intentionally produce no PDF
or GIF, even when `--skip-plots` is omitted during evaluation.

| | Execution check | Formal defaults |
| --- | --- | --- |
| Network | 2 hidden layers, width 8 | 8 hidden layers, width 64 |
| Training | 2 epochs, 4 updates | 1,000 epochs, 32,000 updates |
| Training pools | Reduced | 24,576 fixed points |
| Purpose | Verify installation and the train/save/evaluate path | Run a new research experiment |

The smoke results are not accuracy benchmarks or reproductions of the archived
figures. Every train or evaluation command requires a **new output directory**;
on a second attempt, choose another name rather than mixing it with an old run.

## Train and evaluate

These commands use the formal model and full training budget, and work without
Times New Roman because `--skip-plots` keeps only numerical outputs:

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/standard --device auto --skip-plots
python -m burgers_pinn.train --method viscosity --viscosity 0.001 --output-dir outputs/viscosity --device auto --skip-plots
python -m burgers_pinn.evaluate --checkpoint outputs/viscosity/checkpoint_final.pt --output-dir outputs/viscosity-evaluation --device auto --skip-plots
```

`--device auto` selects CUDA when available and CPU otherwise. Use `--device cpu`
to force CPU execution. Full training takes substantially more work than the
smoke check; the terminal reports progress at the diagnostic epochs.

The default budget is 1,000 epochs. `--epochs` and `--seed` allow explicit
overrides; changing the budget does not reproduce the archived run. Seeds must
be integers from 0 to 4,294,967,295. `configs/formal_defaults.json` is a record
for inspection, not a CLI configuration file; editing it does not change a run.

| Output | Contents |
| --- | --- |
| `checkpoint_final.pt` | Final weights, full configuration, viscosity, sampled points, and smoke flag; training only |
| `training_history.csv` | Per-epoch losses, learning rate, timing, and diagnostic relative L2; training only |
| `result.json` | Completion status, configuration, device, and numerical metrics |
| `evaluation_fields.npz` | `x`, `t`, `prediction`, `truth` (inviscid reference), and `absolute_error` arrays |

To generate the original-style PDFs and GIF, install Times New Roman and omit
`--skip-plots`. Training then also writes three PDFs (history, profiles, and
point layout) and a prediction GIF. Standalone evaluation writes profiles,
point layout when available, and a GIF; it does not reconstruct training history.

No historical pretrained research checkpoint is bundled. Evaluation commands
must point to a checkpoint created by an earlier training command. A checkpoint
reloads its saved configuration and viscosity, but **does not contain optimizer
state or support exact training resumption**. Every training command starts a
new model; standalone evaluation performs no optimization.

## Repository layout

```text
burgers_pinn/   Model, sampling, residuals, training, evaluation, original plotting
configs/       Recorded formal defaults
tests/         Numerical and execution-contract checks
assets/        Preserved research GIFs/PDFs and PDF previews for this README
docs/          Methods, results, provenance, and reproducibility notes
```

Run checks with `python -m unittest discover -s tests -v`. Full 1,000-epoch
experiments were not rerun during packaging. Attribution and licensing scope
are described in [NOTICE.md](NOTICE.md).
