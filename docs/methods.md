# Problem and methods

## One Riemann problem

The space-time domain is `x in [-1, 1]`, `t in [0, 1]`. Coordinates and
solution values in this benchmark are dimensionless. The prescribed data are

```math
u(x,0)=\begin{cases}1 & x<0,\\0 & x\geq0,\end{cases}
\qquad u(-1,t)=1,\quad u(1,t)=0
```

The inviscid entropy solution is a step moving along `x = t/2`:
`u(x,t) = 1` for `x < t/2`, and `0` otherwise. The value on the discontinuity
is fixed by this convention for grid diagnostics.

## Standard PINN

A fully connected network takes `(x,t)` and returns a scalar `u`.
It has eight hidden layers of width 64, Tanh activations, Xavier-normal
weights, zero biases, and 29,377 trainable parameters.

Automatic differentiation supplies the strong-form residual

```math
r_0=u_t+u u_x
```

The objective is the equally weighted sum

```math
\mathcal{L}=\mathcal{L}_{\mathrm{IC}}+
\mathcal{L}_{\mathrm{BC}}+\mathcal{L}_{\mathrm{PDE}},
\qquad\mathcal{L}_{\mathrm{PDE}}=\mathrm{mean}(r_0^2)
```

The IC and BC terms are MSEs against the prescribed initial and boundary
values. Interior points contribute through the PDE residual. The moving
analytical shock is reserved for evaluation and illustration, providing a
reference for the learned profile and its propagation.

## Global artificial-viscosity PINN

This variant uses the same network, data pools, and loss weights, with

```math
r_\nu=u_t+u u_x-\nu u_{xx},\qquad\nu>0
```

The default is the constant `nu = 0.001`. The coefficient is fixed throughout
space and time. This global artificial viscosity regularizes the PDE and
smooths the shock. Comparison to the inviscid entropy solution is a
regularization diagnostic: it measures the combined effect of the learned
approximation and the added viscosity relative to that inviscid reference.

## Shared training configuration

| Setting | Default |
| --- | --- |
| Seed | 1234 |
| Initial pool | 2,048 points per side of the initial discontinuity |
| Boundary pool | 2,048 time samples at each boundary |
| Interior pool | 16,384 uniform space-time points |
| Batch | 128 IC + 128 BC + 512 PDE points |
| Epochs | 1,000 |
| Updates | 32 per epoch, 32,000 total |
| Optimizer | Adam |
| Learning rate | Per-update cosine schedule, `1e-3` to `1e-5` |
| Training-history diagnostic grid | 401 spatial by 81 temporal points |
| Final evaluation grid | 1,001 spatial by 101 temporal points |

Training pools are sampled once and shuffled each epoch. Every epoch sweeps
the full pools. Training saves the model at the final update and records the
analytical-reference diagnostics along the way.

## Reading the diagnostics

Relative L2 and MAE summarize the prediction over the stated fixed grid, while
maximum absolute error captures the largest local discrepancy, including at
the discontinuity. Shock position is estimated from the leftmost downward crossing
of `u = 0.5`; thickness is the distance between the `u = 0.9` and `u = 0.1`
crossings. The estimator falls back to the nearest value when no crossing
exists. Reading these quantities alongside the profile plots shows both the
shock trajectory and its spatial sharpness.

The two residual choices provide a controlled comparison of coordinate-network
solutions for the same prescribed Riemann initial and boundary conditions.
