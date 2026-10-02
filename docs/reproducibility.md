# Reproducibility and preparation checks

## Selected source

The release uses the archived `PDF_PLOTTING_MIGRATION` source copies:

- `code/pinn_burgers_common_plotting.py`: despite its archive name, this is
  the complete baseline module, including the model, configuration, point
  pools, diagnostics, and plotting functions.
- `code/pinn_burgers_formal_rebuild.py`: supplies the controlled scalar
  residual and training loop for zero or fixed positive viscosity.

Their original SHA-256 hashes are in [source-manifest.json](source-manifest.json).
The original source archive is preserved. The migration archive did not record
the runtime source hash of every historical plot, so source compatibility is
not a claim of bitwise reproduction of historical figures.

## Packaging changes

- The selected model, point sampling, residual, optimizer, and diagnostic
  functions were separated into an importable Python package.
- The entry points select only the standard or global artificial-viscosity
  method. Other experiments from the larger archive are not packaged.
- Local CPU/CUDA selection replaces the original cluster-only device guard.
  CUDA synchronization is conditional on the selected device.
- New output paths prevent mixing or overwriting runs. Configuration, viscosity,
  the sampled training points, and a smoke-run flag accompany saved weights.
- Checkpoints load strictly with `weights_only=True`. The version-1 format
  requires the complete saved configuration and a boolean smoke flag; missing
  settings are not silently replaced with current defaults. No original
  trained research checkpoint is included in this repository.
- The original eight plotting functions retain the same parsed function
  bodies. Method labels and equations are passed through their existing
  arguments. Times New Roman and STIX typography are preserved, and missing
  required fonts are reported before full training starts.
- Tiny smoke checks use reduced model/pool sizes and are explicitly marked.
  They skip research plots to avoid reusing historical fixed-pool annotations
  for a different sampling budget.
- Machine-specific font paths are recorded as font filenames in new reports.
- Seed and grid settings are validated before a new run is created. Training
  prints progress at the existing diagnostic epochs; this does not alter the
  optimizer or its schedule.

`configs/formal_defaults.json` records the formal defaults for inspection;
the executable defaults are `TrainingConfig` in `burgers_pinn/core.py`.
The configuration file is not a separate CLI input.

Saved checkpoints support prediction and evaluation with the recorded network,
configuration, viscosity, and training points. They do not save optimizer state,
RNG state, or a resumable optimizer-step counter. There is no exact-resume CLI:
each training invocation initializes a new model, and evaluation does not train.

## Numerical interpretation

The input is a pair of dimensionless coordinates and the output is a scalar
state. Training uses prescribed IC/BC values and PDE residuals. The analytical
inviscid entropy solution appears only in evaluation and figures. It does not
enter the optimizer loss or select a checkpoint.

The training-history L2 diagnostic uses a 401 by 81 grid. The final metric
report uses 1001 by 101 points by default. These grids differ, so the last
history value need not equal the final-report value.

The viscosity variant changes the PDE. Its discrepancy from the inviscid
reference includes the effect of regularization. Fixed-grid errors and the
level-crossing shock estimates are diagnostics for this particular Riemann
problem, not evidence of generalization to new initial conditions.

## Checks completed on 2026-10-02

- Nine tests passed for residual derivatives (including the second derivative
  and viscosity sign), prescribed data and fixed-pool batching, model shape,
  checkpoint round-trip and metadata validation, early invalid-seed rejection,
  and output protection. End-to-end tests train, save, and evaluate both methods;
  they verify exact CPU array/metric agreement and smoke-flag propagation while
  the plotting module is unavailable.
- Both methods completed CPU smoke training and independent checkpoint
  evaluation. Smoke runs contain four optimizer updates each.
- The packaged network and the selected original network, with a common
  state dictionary and input, produced exactly equal output (`torch.equal`).
- Parsed function bodies of eight original plotting functions matched the
  packaged functions.
- An additional viscosity check used the original 8 by 64 network and full
  point pools for one CUDA epoch (32 updates). It generated three PDFs and a
  61-frame GIF. These are execution checks stored in ignored outputs, not
  results displayed in the README.
- The three preserved PDF assets are single-page originals with selectable
  text and vector drawing objects. README PNGs are direct page renderings,
  with no changes to plot data or layout.
- Copied research assets were checked against SHA-256 hashes; see
  [visual provenance](assets-provenance.json).

## Independent CPU installation audit on 2026-10-02

A new GitHub clone was used to create a fresh virtual environment without
`--system-site-packages`. The CPU PyTorch wheel and `requirements.txt` were
installed there, and `python -m pip check` passed. That independent interpreter
then ran the updated source and all nine tests successfully. The tested versions
were Python 3.12.12, PyTorch 2.10.0+cpu, NumPy 2.5.3, Matplotlib 3.11.2,
Pillow 12.3.0, and PyMuPDF 1.28.2 on Windows.

The public standard and viscosity CLI commands each completed CPU smoke
training, saved a final checkpoint, and completed a separate CPU evaluation
in a fresh output directory. Both paths used four optimizer updates and
preserved `smoke_test: true`. All four reports recorded `completed: true`;
no research plots were generated. These outputs remain local execution evidence
and are not presented as research results.

A separate CUDA smoke check also completed for the viscosity method. Loading
that checkpoint for CPU evaluation preserved the method and configuration;
the maximum CPU/GPU prediction difference was approximately `2.16e-7`.
That cross-device check used the existing CUDA environment, not the independent
CPU installation. It does not establish identical numerical results across all
hardware or PyTorch versions.

Checks used Python 3.12 and PyTorch 2.10. The full 1000-epoch runs were not
repeated. The original historical checkpoints and numerical training-history
files are not present in the local migration archive. Archived metrics are
reported as such and have not been independently recomputed in this release.
