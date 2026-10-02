# Reproducibility and verification

## Selected source

The release uses the archived `PDF_PLOTTING_MIGRATION` source copies:

- `code/pinn_burgers_common_plotting.py`: the complete baseline module,
  including the model, configuration, point pools, diagnostics, and plotting
  functions.
- `code/pinn_burgers_formal_rebuild.py`: supplies the controlled scalar
  residual and training loop for zero or fixed positive viscosity.

Their original SHA-256 hashes are in [source-manifest.json](source-manifest.json).
The source archive is preserved, and the package retains the selected model,
sampling, residuals, and research plotting functions.

## Package design

- The selected model, point sampling, residual, optimizer, and diagnostic
  functions were separated into an importable Python package.
- The entry points provide the standard and global artificial-viscosity
  methods under a shared training and evaluation interface.
- Local CPU/CUDA selection replaces the original cluster-only device guard.
  CUDA synchronization is conditional on the selected device.
- New output paths prevent mixing or overwriting runs. Configuration, viscosity,
  the sampled training points, and a smoke-run flag accompany saved weights.
- Checkpoints load strictly with `weights_only=True`. The version-1 format
  records the complete saved configuration and a boolean smoke flag so that
  evaluation uses the settings of the corresponding training run.
- The original eight plotting functions retain the same parsed function
  bodies. Method labels and equations are passed through their existing
  arguments. Times New Roman and STIX typography are preserved, with font
  availability checked before full training starts.
- Smoke checks use reduced model and pool sizes, record `smoke_test: true`,
  and produce numerical outputs for a quick installation check.
- Machine-specific font paths are recorded as font filenames in new reports.
- Seed and grid settings are validated before a new run is created. Training
  prints progress at the diagnostic epochs.

`configs/formal_defaults.json` documents the formal defaults. The executable
defaults are `TrainingConfig` in `burgers_pinn/core.py`; CLI options configure
each new run.

Each training invocation initializes a new model and writes a final checkpoint.
The evaluation command loads that checkpoint for prediction with the recorded
network, configuration, viscosity, and training points.

## Numerical interpretation

The input is a pair of dimensionless coordinates and the output is a scalar
state. Training uses prescribed IC/BC values and PDE residuals. The analytical
inviscid entropy solution supplies the reference for evaluation and figures.
Training saves the weights at its final update.

The training-history L2 diagnostic uses a 401 by 81 grid. The final metric
report uses 1001 by 101 points by default, providing a denser evaluation of
the final model.

The viscosity variant changes the PDE. Its discrepancy from the inviscid
reference includes the effect of regularization. Fixed-grid errors and
level-crossing shock estimates characterize the learned solution for the
prescribed Riemann problem.

## Checks completed on 2026-10-02

- Nine tests passed for residual derivatives (including the second derivative
  and viscosity sign), prescribed data and fixed-pool batching, model shape,
  checkpoint round-trip and metadata validation, early invalid-seed rejection,
  and output protection. End-to-end tests train, save, and evaluate both methods;
  they verify exact CPU array/metric agreement and smoke-flag propagation
  independently of the plotting module.
- Both methods completed CPU smoke training and independent checkpoint
  evaluation. Smoke runs contain four optimizer updates each.
- The packaged network and the selected original network, with a common
  state dictionary and input, produced exactly equal output (`torch.equal`).
- Parsed function bodies of eight original plotting functions matched the
  packaged functions.
- An additional viscosity check used the original 8 by 64 network and full
  point pools for one CUDA epoch (32 updates), verifying generation of three
  PDFs and a 61-frame GIF.
- The three preserved PDF assets are single-page originals with selectable
  text and vector drawing objects. README PNGs are direct page renderings,
  with no changes to plot data or layout.
- Copied research assets were checked against SHA-256 hashes; see
  [visual provenance](assets-provenance.json).

## Fresh CPU installation on 2026-10-02

A new GitHub clone was used to create an isolated virtual environment. The
CPU PyTorch wheel and `requirements.txt` were installed there, and
`python -m pip check` passed. That independent interpreter
then ran the updated source and all nine tests successfully. The tested versions
were Python 3.12.12, PyTorch 2.10.0+cpu, NumPy 2.5.3, Matplotlib 3.11.2,
Pillow 12.3.0, and PyMuPDF 1.28.2 on Windows.

The public standard and viscosity CLI commands each completed CPU smoke
training, saved a final checkpoint, and completed a separate CPU evaluation
in a fresh output directory. Both paths used four optimizer updates and
preserved `smoke_test: true`. All four reports recorded `completed: true`,
verifying the complete numerical train/save/evaluate path.

A separate CUDA smoke check also completed for the viscosity method. Loading
that checkpoint for CPU evaluation preserved the method and configuration;
the maximum CPU/GPU prediction difference was approximately `2.16e-7`.
The CUDA training environment and the fresh CPU installation were checked
separately; this additional check verified loading a CUDA-trained smoke
checkpoint for CPU evaluation.

The [research results](results.md) are sourced from the archived 1,000-epoch
experiment reports. The checks above document installation, numerical
implementation, checkpoint handling, and figure generation.
