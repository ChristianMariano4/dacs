# Contributing to DACS

Thanks for contributing to DACS.

## Contribution Scope

Useful contributions include:
- bug fixes and reliability improvements
- planning or memory logic improvements
- perception pipeline and graph reasoning improvements
- documentation and reproducibility updates

## Workflow

1. Fork the repository and create a branch from `main`.
2. Use a focused branch name such as `fix/tello-timeout` or `feat/context-graph-cache`.
3. Keep each pull request scoped to one coherent change.
4. Update documentation when user-facing behavior or setup changes.

## Local Validation

Run the minimum checks relevant to your change before opening a PR.

Core smoke test:
```bash
make typefly-v
```

If your change touches protobuf contracts:
```bash
cd proto && bash generate.sh
```

If your change touches YOLO service or router containers:
```bash
make SERVICE=yolo build
make SERVICE=router build
```

## Pull Request Best Practices

- Use a precise title that describes behavior change, not only the implementation detail.
- In the description, include:
  - problem statement
  - approach and tradeoffs
  - validation steps you executed locally
  - expected impact on runtime behavior
- Link related issues/discussions when available.
- Keep commits atomic and readable; avoid mixing refactor-only commits with behavior changes.
- Prefer opening a draft PR early for design feedback on larger changes.
- For UI/perception/planning behavior changes, attach screenshots or short videos.
- For hardware-dependent changes, state explicitly how the change was validated (real drone, virtual mode, or both).

## PR Checklist

Before requesting review, confirm:
- [ ] I rebased or merged latest `main` and resolved conflicts.
- [ ] The PR is focused and does not include unrelated cleanup.
- [ ] I ran relevant local validation commands.
- [ ] I updated docs/config comments if behavior changed.
- [ ] I added evidence (logs, screenshots, or videos) when useful.
- [ ] I described limitations and follow-ups clearly.

