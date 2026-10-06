# Burgers PINNs: Standard and Artificial Viscosity

**Yanghao Chen · Tongji University**

This study compares a standard physics-informed neural network (PINN) with a
PINN using constant artificial viscosity for a one-dimensional Burgers Riemann
problem. Both PyTorch implementations use the same network architecture,
sampling configuration, and training budget. We evaluate shock position,
transition width, and error relative to the analytical inviscid entropy solution.

[Experiment](#what-one-experiment-contains) · [Problem and data](#problem-and-training-data) · [Network and loss](#network-and-loss) · [Training](#training-strategy) · [Results](#research-results) · [Run the workflow](#install) · [Documentation](#documentation)

## What one experiment contains

Each run generates fixed training-point pools, fits a coordinate network
$(x,t)\mapsto u_\theta(x,t)$, and saves the final weights and predictions.

| Component | Contents |
| :--- | :--- |
| Physical problem | Initial jump from 1 to 0 on $x\in[-1,1]$, with $t\in[0,1]$ |
| Training data | Fixed pools of initial, boundary, and interior coordinates |
| Learned field | Coordinate network $(x,t)\mapsto u_\theta(x,t)$ |
| Numerical outputs | Checkpoint, loss-history CSV, metrics JSON, and field arrays |
| Visual outputs | Training-history, sampling, and profile PDFs; prediction GIF |

The [experiment guide](docs/experiment-guide.md) gives a worked training batch
and an example of extracting a profile from the saved field arrays.

## Problem and training data

The dimensionless domain is $x\in[-1,1]$, $t\in[0,1]$. The initial state is 1
to the left of zero and 0 to the right. Boundary values remain 1 at the left
boundary and 0 at the right. The inviscid entropy shock moves along $x=t/2$.

### Analytical reference animations

| Analytical solution profile $u(x,t)$ | Analytical shock trajectory in $(x,t)$ |
| :---: | :---: |
| ![Exact entropy-solution profile moving from x=0 to x=0.5](assets/gifs/shock-motion.gif) | ![Exact shock trajectory x_s(t)=0.5t in the space-time plane](assets/gifs/shock-trajectory.gif) |
| Horizontal axis: position $x$; vertical axis: state $u$. The blue step is the exact inviscid solution; the red dashed vertical line marks the shock position at the time shown above the plot. | Horizontal axis: position $x$; vertical axis: physical time $t$. The teal dashed line is the full shock path; the red segment shows the path already traversed, the red dot marks the current position, and the orange horizontal line marks the current time. |

The analytical shock advances from $x=0$ to $x=0.5$ as physical time increases
from 0 to 1, retaining states 1 and 0 on either side. Each animation contains
61 time slices. Network predictions are shown separately in
[Prediction examples](#prediction-examples).

[Static reference at $t=0.5$](assets/figures/inviscid-shock-reference.png) ·
[PDF](assets/pdf/inviscid-shock-reference.pdf). The shock position is $x=0.25$;
the static trajectory uses gray dashes.

### Training-point generation

Training points are sampled uniformly within each prescribed set once, then
shuffled each epoch. IC and BC pools are balanced across their two sides.

| Point set | Pool size | Per update | Training target |
| :--- | ---: | ---: | :--- |
| Initial condition (IC) | 4,096 | 128 | Prescribed step at $t=0$ |
| Boundary condition (BC) | 4,096 | 128 | Prescribed values at $x=-1$ and $x=1$ |
| Interior (PDE) | 16,384 | 512 | Zero PDE residual |
| **Total** | **24,576** | **768** | |

Interior coordinates enter through the PDE residual, evaluated by automatic
differentiation. The analytical field is used only for diagnostics and plots.
The [batch definition](docs/methods.md#what-goes-into-a-training-batch) specifies
the coordinates and prescribed IC/BC targets.

![Standard PINN training-point layout](assets/figures/standard-training-points.png)

[Original sampling-layout PDF](assets/pdf/standard-training-points.pdf).
Orange squares mark initial points at $t=0$, teal triangles mark boundary
points at $x=\pm1$, and blue dots mark interior collocation points. The figure
shows a readable subset of the fixed pools. The dashed red line marks the
analytical shock path $x=t/2$ as a visual reference.

## Network and loss

![Burgers problem, characteristic lines, coordinate network, automatic differentiation, and three PINN loss terms](assets/figures/pinn-problem-framework.svg)

[Vector SVG](assets/figures/pinn-problem-framework.svg) ·
[PDF](assets/pdf/pinn-problem-framework.pdf) ·
[PNG](assets/figures/pinn-problem-framework.png)

The diagram connects the prescribed problem to the coordinate network,
automatic differentiation, and three loss terms. Its network is schematic;
the implemented dimensions are listed below. The viscosity variant adds
$-\nu u_{xx}$ to the displayed inviscid residual.

| Network component | Configuration |
| :--- | :--- |
| Mapping | $(x,t)\mapsto u_\theta(x,t)$ |
| Hidden layers | 8 layers × 64 units, Tanh activation |
| Output | One scalar, linear activation |
| Initialization | Xavier-normal weights, zero biases |
| Trainable parameters | 29,377 |

| Method | Residual used in training |
| --- | --- |
| Standard PINN | $r=u_t+u u_x$ |
| Artificial-viscosity PINN | $r=u_t+u u_x-\nu u_{xx}$, default $\nu=10^{-3}$ |

Automatic differentiation computes the derivatives. Both methods minimize

```math
\mathcal{L}=\mathrm{MSE}_{\mathrm{IC}}+\mathrm{MSE}_{\mathrm{BC}}+\mathrm{mean}(r^2)
```

The three loss weights are 1. Viscosity is a prescribed global coefficient,
constant in space and time. [Methods](docs/methods.md) gives the equations and
diagnostic definitions.

## Training strategy

| Setting | Shared default |
| :--- | :--- |
| Seed | 1234 |
| Batch | 128 IC + 128 BC + 512 PDE points |
| Epochs / updates | 1,000 / 32,000 |
| Optimizer | Adam |
| Learning rate | Per-update cosine decay from `1e-3` to `1e-5` |
| Training-history diagnostic | Relative L2 on a 401 × 81 grid |
| Final evaluation | 1,001 × 101 grid |
| Saved model | Weights at the final update |

![Archived standard PINN training history](assets/figures/standard-training-history.png)

[Original training-history PDF](assets/pdf/standard-training-history.pdf).
Panels show **(a)** total training loss, **(b)** its initial, boundary, and
PDE-residual components, **(c)** relative $L^2$ against the analytical reference,
and **(d)** learning rate. All vertical axes use a logarithmic scale. Panel (a)
is the sum of the three training terms in (b); panel (c) is an evaluation
diagnostic.

## Research results

The archived runs were evaluated on a **1,001 × 101 space–time grid** against
the inviscid entropy solution. The table reproduces their recorded diagnostics.

| Diagnostic | Standard PINN | Artificial viscosity, $\nu=10^{-3}$ |
| :--- | ---: | ---: |
| Relative $L^2$ error | 0.158199 | 0.0248065 |
| Mean absolute error | 0.0594167 | 0.00194412 |
| Maximum absolute error | 0.521592 | 0.554839 |
| Mean absolute shock-position error | 0.00112313 | 0.000225788 |
| Mean shock thickness | 0.410108 | 0.00892251 |

For these runs, artificial viscosity reduces relative $L^2$ and mean absolute
error and gives a narrower transition, but increases the maximum pointwise
error. Shock thickness is the distance between the predicted 0.9 and 0.1
crossings. These are results for one configuration and seed, rather than
statistics over repeated trials. Because viscosity changes the PDE, comparison
with the inviscid reference includes the effect of regularization.
[Results and provenance](docs/results.md) gives the metric definitions and reports.

![Archived standard PINN profiles](assets/figures/standard-solution-comparison.png)

[Original profile-comparison PDF](assets/pdf/standard-solution-comparison.pdf).
The five panels show the standard PINN at $t=0,0.25,0.5,0.75,1$.
Blue: inviscid reference. Red: prediction from the same trained network.
The profiles show propagation of the transition and its finite width.

### Prediction examples

| Standard PINN | Artificial-viscosity PINN, $\nu=0.001$ |
| :---: | :---: |
| ![Standard PINN prediction](assets/gifs/standard-pinn.gif) | ![Artificial-viscosity PINN prediction](assets/gifs/artificial-viscosity-pinn.gif) |
| Standard PINN: blue is the analytical inviscid step; red is this trained network's predicted profile. | Artificial-viscosity PINN with $\nu=0.001$: blue is the same inviscid reference; red is this separately trained network's predicted profile. |

Each animation evaluates a fixed trained model over physical time, from 0 to 1.
The inviscid solution is the common reference in both panels.
The archived checkpoints are not included. The commands below train new models
with the recorded configuration; they do not reload the models behind these figures.

## Install

The tested environment uses **Python 3.12** and **PyTorch 2.10.0**. Start with
the CPU installation below; training points are generated locally, so no
dataset download is required. Run all commands from the repository root:

```bash
git clone https://github.com/cyhcyh070126-bot/burgers-pinn.git
cd burgers-pinn
```

<details open>
<summary><strong>Windows PowerShell</strong></summary>

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

</details>

<details>
<summary><strong>Linux</strong></summary>

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

</details>

For CUDA training or another platform, install the compatible PyTorch build
using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/)
before installing `requirements.txt`. The installation and execution checks
used a fresh Windows CPU environment.

PDF and GIF generation uses **Times New Roman** and STIX math text.
Use `--skip-plots` for numerical training and evaluation without the plot fonts.

## Quick execution check

After installation, run the commands below to check both methods and reload their saved checkpoints:

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
checkpoints also record `smoke_test: true`. Smoke checks use numerical outputs;
the research configuration also supports PDF and GIF generation as described below.

| Configuration | Execution check | Research run |
| :--- | :--- | :--- |
| Hidden layers × width | 2 × 8 | 8 × 64 |
| Epochs / optimizer updates | 2 / 4 | 1,000 / 32,000 |
| Training-point pools | Reduced | 24,576 points |
| Purpose | Verify train, save, and evaluate | Train the full model |

Every train or evaluation command requires a **new output directory** to keep
each experiment's configuration and results together. Choose a new name when
running again.

## Train and evaluate

These commands use the full model and 1,000-epoch training budget.
`--skip-plots` produces numerical outputs without requiring Times New Roman:

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/standard --device auto --skip-plots
python -m burgers_pinn.train --method viscosity --viscosity 0.001 --output-dir outputs/viscosity --device auto --skip-plots
python -m burgers_pinn.evaluate --checkpoint outputs/viscosity/checkpoint_final.pt --output-dir outputs/viscosity-evaluation --device auto --skip-plots
```

`--device auto` selects CUDA when available and CPU otherwise. Use `--device cpu`
to force CPU execution. Full training takes substantially more work than the
smoke check; the terminal reports progress at the diagnostic epochs.

The default budget is 1,000 epochs. Use `--epochs` and `--seed` to configure a
new experiment. Seeds must be integers from 0 to 4,294,967,295.
`configs/formal_defaults.json` documents the default settings; configure runs
through the CLI options, which are listed by `python -m burgers_pinn.train --help`.

| Output | Contents |
| --- | --- |
| `checkpoint_final.pt` | Final model, configuration, viscosity, and sampled points; saved by training |
| `training_history.csv` | Per-epoch losses, learning rate, timing, and relative $L^2$; saved by training |
| `result.json` | Run status, configuration, device, and metrics |
| `evaluation_fields.npz` | Coordinates, predicted field, inviscid reference, and absolute error |

The NPZ file contains `x`, `t`, `prediction`, `truth`, and `absolute_error`.
Field arrays are indexed by time, then space.

See the [illustrated output-file guide](docs/experiment-guide.md#5-what-appears-in-the-experiment-folder)
for exact filenames and a runnable array-reading example, and
[reading a saved evaluation](docs/results.md#reading-a-saved-evaluation)
for array shapes and the connection between field values and profile plots.

To generate the original-style PDFs and GIF, install Times New Roman and omit
`--skip-plots`. Training then also writes three PDFs (history, profiles, and
point layout) and a prediction GIF. Standalone evaluation writes profiles,
point layout when available, and a GIF. Training-history plots use the history
recorded during training.

Training starts a new experiment; evaluation reloads its final checkpoint,
including the saved configuration, viscosity, and sampled points.

## Documentation

The [source audit](docs/source-audit.md) records the four supplied Burgers scripts,
their roles, hashes, and execution fixes. `source_archive/` preserves the original
files; `research_scripts/` contains the reviewed historical versions. The
commands above use the packaged scalar implementation in `burgers_pinn/`.

| Guide | What it covers |
| :--- | :--- |
| [Illustrated experiment guide](docs/experiment-guide.md) | One experiment from problem setup to saved predictions |
| [Methods](docs/methods.md) | Equations, sampling, losses, and shock diagnostics |
| [Results](docs/results.md) | Archived metrics, figures, and evaluation examples |
| [Reproducibility](docs/reproducibility.md) | Environment and execution checks |
| [Research homepage](https://cyhcyh070126-bot.github.io/) | Project context and related research |

## Repository layout

```text
burgers_pinn/   Model, sampling, residuals, training, evaluation, original plotting
source_archive/ Exact original source snapshots
research_scripts/ Reviewed original research and reference scripts
configs/       Recorded formal defaults
tests/         Numerical and execution-contract checks
assets/        Preserved research GIFs/PDFs and PDF previews for this README
docs/          Illustrated experiment guide, methods, results, and reproducibility
```

Run the numerical and train/save/evaluate checks with
`python -m unittest discover -s tests -v`. See
[Reproducibility](docs/reproducibility.md) for the tested environment and
verification details, and [NOTICE.md](NOTICE.md) for attribution and licensing.
