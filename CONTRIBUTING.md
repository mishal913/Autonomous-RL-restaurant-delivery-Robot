# Contributing

This repository contains reinforcement-learning and motion-planning experiments for an autonomous delivery robot.

## Before changing the controller

- keep evaluation seeds/results separate from training
- report success and collision metrics together
- do not claim safety from reward alone
- preserve the distinction between static-obstacle navigation and dynamic-human avoidance
- document observation-space changes because old SAC checkpoints may become incompatible
- avoid committing Python caches or local simulator files

## Validation

At minimum:

```bash
python -m compileall -q robot
python -m compileall -q restaurant
```

For controller changes, also run the relevant seeded evaluation script and commit the resulting summary CSV.

## Large artifacts

Model checkpoints, replay buffers, NumPy episode archives and TensorBoard logs are configured for Git LFS. Install Git LFS before adding those artifacts:

```bash
git lfs install
```
