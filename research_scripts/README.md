# Reviewed original Burgers research scripts

Four author-provided scripts, with explicit import/configuration/label fixes. Their numerical formulas, seeds, plotting constants and GPU-only research requirements are retained. See the [file-by-file audit](../docs/source-audit.md) and [diffs](../docs/source-patches/).

For portable standard/artificial-viscosity training and evaluation, use `burgers_pinn.train` and `burgers_pinn.evaluate` from the main README. The domain-decomposition originals prescribe the analytical interface; they are separate historical checks. Inspect output paths before running a research script. The exact-reference script renders the analytical solution, not learned results.
