# Burgers PINNs: Standard and Artificial Viscosity

**Yanghao Chen · Tongji University**

[Research homepage](https://cyhcyh070126-bot.github.io/) · [Methods](docs/methods.md) · [Results and figures](docs/results.md) · [Reproducibility](docs/reproducibility.md)

A PyTorch study of shock formation and propagation in a one-dimensional Burgers
Riemann problem. Two implementations share the same coordinate network and
training data: a standard strong-form PINN and a PINN with constant global
artificial viscosity.

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
| Standard PINN | $r=u_t+u\,u_x$ |
| Global artificial viscosity | $r=u_t+u\,u_x-\nu u_{xx}$, default $\nu=10^{-3}$ |

Automatic differentiation computes the derivatives. Both methods minimize

$$\mathcal L=\operatorname{MSE}_{\mathrm{IC}}+
\operatorname{MSE}_{\mathrm{BC}}+\operatorname{mean}(r^2).$$

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

Use Python 3.10 or newer; preparation checks used Python 3.12 and PyTorch 2.10.

```bash
git clone https://github.com/cyhcyh070126-bot/burgers-pinn.git
cd burgers-pinn
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for your hardware.
The original plot style requires locally installed **Times New Roman** faces
and uses STIX math text; font files are not bundled. Use `--skip-plots` when
you only need numerical training/evaluation without those fonts.

## Quick execution check

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/smoke-standard --device cpu --smoke-test
python -m burgers_pinn.train --method viscosity --output-dir outputs/smoke-viscosity --device cpu --smoke-test
python -m burgers_pinn.evaluate --checkpoint outputs/smoke-viscosity/checkpoint_final.pt --output-dir outputs/smoke-evaluation --device cpu
```

Smoke checks use a reduced network, reduced pools, and four optimizer updates.
Their metadata records `smoke_test: true`, and research plots are disabled.
They verify execution and checkpoint loading, not the archived performance.
Every run requires a **new output directory**.

## Train and evaluate

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/standard --device auto
python -m burgers_pinn.train --method viscosity --viscosity 0.001 --output-dir outputs/viscosity --device auto
python -m burgers_pinn.evaluate --checkpoint outputs/viscosity/checkpoint_final.pt --output-dir outputs/viscosity-evaluation --device auto
```

The default budget is 1,000 epochs. `--epochs` and `--seed` allow explicit
overrides; changing the budget does not reproduce the archived run.

Training writes a checkpoint, CSV history, `result.json`, and
`evaluation_fields.npz` containing coordinates, prediction, inviscid reference,
and absolute error. With plots enabled, it also writes three PDFs (training
history, profiles, and point layout) and a prediction GIF. Standalone evaluation
reloads the stored configuration and viscosity and writes metrics, arrays,
profiles, point layout when available, and a GIF.

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
