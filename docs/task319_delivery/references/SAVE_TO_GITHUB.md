# Save Task319 Code

This directory is saved as an independent Git repository for the Task319 code and configuration.

Local identity:

```bash
git config user.name "CuberW"
git config user.email "773329413@qq.com"
```

Create or connect a GitHub repository, then push:

```bash
cd mda_isaaclab/task_319_garbage_sort
git remote add origin git@github.com:CuberW/task319-garbage-sort.git
git branch -M main
git push -u origin main
```

If using HTTPS instead of SSH:

```bash
git remote add origin https://github.com/CuberW/task319-garbage-sort.git
git branch -M main
git push -u origin main
```

Notes:
- `output/`, `models/`, `third_party/`, Python caches, and model weights are ignored.
- The GLM API key must stay in environment variables and must not be committed.
- Large downloaded model files should be recorded in documentation or handled by a prepare script, not committed to GitHub.
