# Research results and visual provenance

The figures and numbers on this page come from archived 1,000-epoch experiments.
The repository includes their prediction animations, original PDF figures,
and recorded diagnostics. For a new experiment, use the full training commands
in the [README](../README.md#train-and-evaluate).

## Problem and comparison

Both runs use the Riemann initial condition $u(x,0)=1$ for $x<0$ and $0$
otherwise, with $u(-1,t)=1$ and $u(1,t)=0$ on
$x\in[-1,1]$, $t\in[0,1]$.

- **Standard PINN:** strong residual $u_t+u u_x=0$.
- **Artificial-viscosity PINN:** strong residual
  $u_t+u u_x-\nu u_{xx}=0$, with constant $\nu=10^{-3}$.

The reference is the **inviscid entropy solution**, whose discontinuity travels
along $x_s(t)=0.5t$. The reference is reserved for evaluation and visual
comparison. For the artificial-viscosity PDE, the distance to this inviscid
reference is a diagnostic of regularization. The GIF label “Exact solution”
denotes this inviscid reference in both panels.

## Preserved prediction animations

| Standard PINN | Artificial-viscosity PINN, $\nu=10^{-3}$ |
| --- | --- |
| ![Archived Standard PINN prediction and inviscid entropy reference](../assets/gifs/standard-pinn.gif) | ![Archived artificial-viscosity PINN prediction and inviscid entropy reference](../assets/gifs/artificial-viscosity-pinn.gif) |

These animations are preserved from the author's personal homepage. Each has
61 frames, a 960 by 900 pixel canvas, and a 100 ms frame duration. Their method
labels and equations correspond to the archived result reports and experiment index.

## Recorded diagnostics

The archived reports evaluate a fixed grid of **1001 spatial points by 101 time
points**. The table reproduces the diagnostics recorded in those reports.

| Diagnostic against the inviscid entropy reference | Standard PINN | Artificial viscosity, $\nu=10^{-3}$ |
| --- | ---: | ---: |
| Relative $L^2$ error | 0.158199 | 0.0248065 |
| Mean absolute error | 0.0594167 | 0.00194412 |
| Maximum absolute error | 0.521592 | 0.554839 |
| Mean absolute shock-location error | 0.00112313 | 0.000225788 |
| Maximum absolute shock-location error | 0.00194988 | 0.000481546 |
| Mean predicted shock thickness | 0.410108 | 0.00892251 |
| Maximum predicted shock thickness | 0.718692 | 0.00899506 |

Shock location is the leftmost downward crossing of the predicted value 0.5,
with a nearest-value fallback if no crossing exists. Shock thickness is the
distance between the predicted 0.9 and 0.1 crossings. These are dimensionless
quantities. Maximum pointwise error is sensitive to the discontinuity and should
be read alongside the integrated and shock diagnostics.

Both reports specify seed 1234, eight hidden layers of width 64 with Tanh
activations, 29,377 trainable parameters, and 1000 training epochs. Fixed pools
contain 4096 initial-condition, 4096 boundary-condition, and 16,384 PDE points.
The batch sizes are 64 per initial-condition side, 64 per boundary-condition
side, and 512 PDE points. The recorded learning-rate range is $10^{-3}$ to
$10^{-5}$; evaluation occurs every five epochs.

## Original PDF figures

The following original figures document the **Standard PINN** experiment:

- [Training history](../assets/pdf/standard-training-history.pdf): total loss,
  IC/BC/PDE MSE, relative $L^2$ diagnostic, and learning rate.
- [Solution comparison](../assets/pdf/standard-solution-comparison.pdf): profiles
  at $t=0,0.25,0.5,0.75,1$.
- [Training-point layout](../assets/pdf/standard-training-points.pdf): a displayed
  subset of the fixed point pools, with the analytical shock path shown as a
  dashed reference line.

The PDFs are unmodified, single-page originals with selectable text and their
original layout and typography.

The [provenance manifest](assets-provenance.json) records SHA-256 hashes for every
copied asset and original result report, the selected configuration, and the
reported metrics. Together, these records link the displayed results to the
standard and constant-global-viscosity experiments.
