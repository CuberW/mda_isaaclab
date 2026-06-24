# Shared configuration

This directory is the cross-task configuration seam for parameters that should
not be buried in task episode code.

The existing `configs/` directory remains the authoritative per-task
configuration location. This `config/` directory holds shared tunables and
architecture-level defaults used across perception, planning, and control.

