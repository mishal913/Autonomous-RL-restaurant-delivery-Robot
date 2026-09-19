# Large Training Artifacts

The original uploaded project archive contains **116 binary artifacts totaling approximately 203.8 MB**.

They include:

- Stable-Baselines3 SAC model ZIP files
- intermediate SAC checkpoints
- replay-buffer PKL files
- compressed per-episode NPZ experience files
- TensorBoard event files
- a smoke-test archive

The exact expected path, byte size and SHA-256 for every binary artifact is stored in:

```text
artifacts_manifest.csv
```

## Why Git LFS is used

These files are machine-learning artifacts rather than source code. Storing them as ordinary Git blobs unnecessarily bloats repository history.

The repository therefore contains:

```text
.gitattributes
```

with Git LFS rules for:

```text
*.zip
*.pkl
*.npz
*.tfevents.*
```

## Restore the full uploaded archive

The helper script in:

```text
scripts/import_large_artifacts.ps1
```

can import the binary artifacts from the original `robot(1).zip`, verify them against `artifacts_manifest.csv`, stage them through Git LFS, and optionally commit/push them.

This keeps the source repository readable while still allowing the complete training state to be versioned with the correct Git mechanism.
