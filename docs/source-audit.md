# Original-source audit (2026-10-06)

This audit compares each in-scope source file with the author’s supplied local originals. It does not mean every historical version is used by the current quick start.

Exact source bytes are retained in `source_archive/`. Separately reviewed copies are in `research_scripts/`; each change has a [unified diff](source-patches/). The supported portable entry points remain the ones in the repository README.

[File hashes and coverage](complete-source-manifest.json) · [Definition-level review and settings](file-review.json)

## Coverage and current entry points

All four original `pinn_burgers_*.py` files from the supplied `AA_画图PDF` migration code folder are preserved and reviewed. The earlier selected-source manifest contained two. Unrelated FEM, composite and neural-operator projects in that archive are not Burgers PINN sources.

The supported `burgers_pinn.train` and `burgers_pinn.evaluate` entry points remain the standard and artificial-viscosity scalar workflows. The original prescribed-interface scripts are retained separately: they use the supplied analytical shock trajectory and are not an unknown-interface solver. Original GPU-only research-script requirements and plotting settings are retained.

## Verified comparisons

- 14 Python unit tests passed in a separate CPU environment, including the train/save/evaluate CLI checks.
- Original configuration, seeded network parameters, fixed point pools and all 32 shuffled batches matched the packaged scalar workflow.
- Residuals at viscosity 0, 0.001, 0.003 and 0.01, parameter gradients and one Adam update matched the original exactly.
- Original prescribed-interface point pools match the reviewed script; partial pool sweeps now raise an error instead of dropping points.
- Saved run metadata with extra viscosity fields can be reconstructed; the animation label reflects the actual saved viscosity. These are rendering/configuration checks, not a newly completed full research training.

The test fixture preloads Adam’s lazy backend before an existing mocked module-cache test. This prevents test-order-dependent operator-registration errors; it changes no training algorithm. Full 1,000-epoch research reruns and every historical PDF/GIF export were not rerun during this audit.

## File-by-file review

### Source root

| Original file | Reviewed responsibility | Fixes |
| :--- | :--- | :--- |
| [pinn_burgers_common_plotting.py](../source_archive/pinn_burgers_common_plotting.py) | Standard Burgers PINN configuration, network, fixed sampling pools, residual, optimizer and original plots/GIFs. Filter run metadata when rebuilding config and label actual viscosity. | Filter run-specific viscosity metadata when reconstructing TrainingConfig; label the actual saved viscosity, preserving layout. |
| [pinn_burgers_dd_plotting.py](../source_archive/pinn_burgers_dd_plotting.py) | Historical prescribed-interface domain-decomposition PINN. Interface x=0.5t is supplied analytically; no automatic shock-discovery claim. Reject partial point-pool sweeps. | Resolve original module imports to their actual archived filenames.; Reject partial XPINN pool sweeps; original 32-batch defaults unchanged. |
| [pinn_burgers_formal_rebuild.py](../source_archive/pinn_burgers_formal_rebuild.py) | Controlled standard/artificial-viscosity reruns and prescribed-interface check. Repair imports to the actual original filenames; residual and optimizer parity verified. | Resolve original module imports to their actual archived filenames. |
| [pinn_burgers_shock_reference.py](../source_archive/pinn_burgers_shock_reference.py) | Exact inviscid entropy-solution reference rendering and PDF/GIF checks; analytical visualization, not a trained prediction. | None; reviewed copy retains original bytes. |
