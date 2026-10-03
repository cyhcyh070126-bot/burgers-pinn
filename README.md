# Burgers PINNs: Standard and Artificial Viscosity

**Yanghao Chen · Tongji University**

[Illustrated experiment guide](docs/experiment-guide.md) · [Install](#install) · [Quick execution check](#quick-execution-check) · [Train and evaluate](#train-and-evaluate) · [Methods](docs/methods.md) · [Results](docs/results.md) · [Reproducibility](docs/reproducibility.md) · [Research homepage](https://cyhcyh070126-bot.github.io/)

A PyTorch study of shock propagation in a one-dimensional Burgers
Riemann problem. Two implementations share the same coordinate network and
training data: a standard strong-form PINN and a PINN with constant global
artificial viscosity.

**First run:** after [installing](#install), run this from the repository root:

```bash
python -m burgers_pinn.train --method standard --output-dir outputs/first-run --device cpu --smoke-test
```

This checks training and saves a checkpoint using four optimizer updates.
The example runs on CPU with generated training points and numerical outputs.
Look for `Completed` in the terminal and `completed: true` in
`outputs/first-run/result.json`. For the 1,000-epoch research configuration,
see [Train and evaluate](#train-and-evaluate). Use a new output-directory name
when running again.

## Prediction examples

| Standard PINN | Artificial-viscosity PINN, $\nu=0.001$ |
| :---: | :---: |
| ![Standard PINN prediction](assets/gifs/standard-pinn.gif) | ![Artificial-viscosity PINN prediction](assets/gifs/artificial-viscosity-pinn.gif) |

These research animations and the results below come from archived experiments.
Each frame shows the predicted spatial profile at the physical time printed
above the axes; the animation advances through physical time, not training
epochs. **Blue is the inviscid entropy solution; red is the network prediction.**
The left animation uses the standard residual, and the right adds constant
artificial viscosity. For the latter method, comparison with the inviscid
reference serves as a regularization diagnostic.

## Analytical reference animations

| Burgers shock motion | Burgers shock trajectory |
| :---: | :---: |
| ![Exact entropy-solution profile moving from x=0 to x=0.5](assets/gifs/shock-motion.gif) | ![Exact shock trajectory x_s(t)=0.5t in the space-time plane](assets/gifs/shock-trajectory.gif) |

These two animations from the research homepage show the analytical inviscid
reference used to interpret the predictions above. On the left, the blue step
is the solution profile and the dashed red line marks its moving discontinuity.
On the right, the teal dashed line is the full path $x_s(t)=0.5t$, the red
segment and dot track its progress, and the orange line marks the current time.
Both advance from $t=0$ to $t=1$ in 61 frames.

## What one experiment contains

One experiment learns a continuous field from a prescribed Riemann problem.
The program generates the training points, optimizes a coordinate network,
and saves its weights, training history, and space-time predictions.

| Component | Default contents |
| --- | --- |
| Physical problem | $x\in[-1,1]$, $t\in[0,1]$; initial jump from 1 to 0; boundary values 1 and 0 |
| Training points | 4,096 initial + 4,096 boundary + 16,384 interior points, generated once per run |
| One optimizer batch | 128 initial + 128 boundary + 512 interior points |
| Coordinate network | $(x,t)\mapsto u_\theta(x,t)$; eight hidden layers of width 64 |
| Saved numerical files | Final checkpoint, per-epoch history CSV, result JSON, and evaluation NPZ |
| Figures when enabled | Training-history PDF, point-layout PDF, profile-comparison PDF, and prediction GIF |

The **[complete illustrated experiment guide](docs/experiment-guide.md)**
follows a coordinate sample into a training batch, explains the two residuals,
lists all output files, and shows how to read a saved prediction at $t=0.5$.

## Problem and training data

The dimensionless domain is $x\in[-1,1]$, $t\in[0,1]$. The initial state is 1
to the left of zero and 0 to the right. Boundary values remain 1 at the left
boundary and 0 at the right. The inviscid entropy shock moves along $x=t/2$.

![Analytical inviscid shock profiles and trajectory](assets/figures/inviscid-shock-reference.png)

This original analytical reference figure shows the moving step at $t=0.5$
and its path through space and time; the shock is at $x=0.25$. It defines the
reference used in the prediction plots.
[Open the original PDF](assets/pdf/inviscid-shock-reference.pdf).

The code generates its training data as fixed uniform pools of
4,096 initial-condition points, 4,096 boundary-condition points, and 16,384
interior collocation points. Each epoch shuffles and sweeps these pools.
The analytical shock solution is used only for diagnostics and plots.

An individual training item is a coordinate pair $(x,t)$. Initial and boundary
points also have a prescribed value of 0 or 1; interior points contribute through
the PDE residual computed by automatic differentiation. One optimizer update
uses **128 initial + 128 boundary + 512 interior points**. See
[what goes into a training batch](docs/methods.md#what-goes-into-a-training-batch)
for the coordinates, targets, and loss terms.

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

The existing research-homepage diagram summarizes the standard PINN:
coordinates enter a network, automatic differentiation forms the PDE residual,
and initial, boundary, and residual losses train the same field. The network
drawing is schematic; the implemented architecture is specified below.
The three loss weights shown in the diagram are all set to 1 in this implementation.
The artificial-viscosity variant adds $-\nu u_{xx}$ to the displayed residual.

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

All three terms have weight 1, and viscosity is a fixed global coefficient.
Training uses the prescribed initial and boundary values together with the PDE
residual; the analytical solution is reserved for evaluation. See
[the method description](docs/methods.md) for the full equations and shock diagnostics.

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
| Saved model | Weights at the final update |

![Archived standard PINN training history](assets/figures/standard-training-history.png)

[Original training-history PDF](assets/pdf/standard-training-history.pdf).
Panels show **(a)** total training loss, **(b)** its initial, boundary, and
PDE-residual components, **(c)** relative $L^2$ against the analytical reference,
and **(d)** learning rate. All vertical axes use a logarithmic scale. Panel (a)
is the sum of the three training terms in (b); panel (c) is an evaluation
diagnostic.

## Research results

| Diagnostic against the inviscid reference | Standard PINN | Artificial viscosity, $\nu=10^{-3}$ |
| --- | ---: | ---: |
| Relative L2 error | 0.158199 | 0.0248065 |
| Mean absolute error | 0.0594167 | 0.00194412 |
| Maximum absolute error | 0.521592 | 0.554839 |
| Mean absolute shock-position error | 0.00112313 | 0.000225788 |
| Mean shock thickness | 0.410108 | 0.00892251 |

These fixed-grid diagnostics are recorded in the archived experiment reports.
[Results and provenance](docs/results.md) describe the metric definitions,
reference solution, source reports, and research figures.

![Archived standard PINN profiles](assets/figures/standard-solution-comparison.png)

[Original profile-comparison PDF](assets/pdf/standard-solution-comparison.pdf).
The five panels show the same trained standard PINN at
$t=0,0.25,0.5,0.75,1$: the blue step is the inviscid reference, and the red
curve is the prediction. Read the horizontal shift as shock propagation and
the transition width as predicted shock thickness. The
[illustrated results guide](docs/results.md#reading-the-research-figures)
connects each figure to the reported diagnostics.

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
before installing `requirements.txt`. The installation and execution checks
used a fresh Windows CPU environment.

The original plot style uses locally installed **Times New Roman** faces and
STIX math text. Use `--skip-plots` for numerical training and evaluation.

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
checkpoints also record `smoke_test: true`. Smoke checks use numerical outputs;
the research configuration also supports PDF and GIF generation as described below.

| | Execution check | Formal defaults |
| --- | --- | --- |
| Network | 2 hidden layers, width 8 | 8 hidden layers, width 64 |
| Training | 2 epochs, 4 updates | 1,000 epochs, 32,000 updates |
| Training pools | Reduced | 24,576 fixed points |
| Purpose | Verify installation and the train/save/evaluate path | Run a new research experiment |

Every train or evaluation command requires a **new output directory** to keep
each experiment's configuration and results together. Choose a new name when
running again.

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

The default budget is 1,000 epochs. Use `--epochs` and `--seed` to configure a
new experiment. Seeds must be integers from 0 to 4,294,967,295.
`configs/formal_defaults.json` documents the default settings; configure runs
through the CLI options, which are listed by `python -m burgers_pinn.train --help`.

| Output | Contents |
| --- | --- |
| `checkpoint_final.pt` | Final weights, full configuration, viscosity, sampled points, and smoke flag; training only |
| `training_history.csv` | Per-epoch losses, learning rate, timing, and diagnostic relative L2; training only |
| `result.json` | Completion status, configuration, device, and numerical metrics |
| `evaluation_fields.npz` | Coordinate vectors `x`, `t`, plus `prediction`, `truth` (inviscid reference), and `absolute_error` arrays indexed by time, then space |

See the [illustrated output-file guide](docs/experiment-guide.md#5-what-appears-in-the-experiment-folder)
for exact filenames and a runnable array-reading example, and
[reading a saved evaluation](docs/results.md#reading-a-saved-evaluation)
for array shapes and the connection between field values and profile plots.

To generate the original-style PDFs and GIF, install Times New Roman and omit
`--skip-plots`. Training then also writes three PDFs (history, profiles, and
point layout) and a prediction GIF. Standalone evaluation writes profiles,
point layout when available, and a GIF. Training-history plots use the history
recorded during training.

Training creates `checkpoint_final.pt`; pass that file to the evaluation command
to reload the model with its saved configuration, viscosity, and sampled points.
Each training command initializes a new experiment. The evaluation command
uses a saved model to generate predictions and metrics.

## Repository layout

```text
burgers_pinn/   Model, sampling, residuals, training, evaluation, original plotting
configs/       Recorded formal defaults
tests/         Numerical and execution-contract checks
assets/        Preserved research GIFs/PDFs and PDF previews for this README
docs/          Illustrated experiment guide, methods, results, and reproducibility
```

Run the numerical and train/save/evaluate checks with
`python -m unittest discover -s tests -v`. See
[Reproducibility](docs/reproducibility.md) for the tested environment and
verification details, and [NOTICE.md](NOTICE.md) for attribution and licensing.
