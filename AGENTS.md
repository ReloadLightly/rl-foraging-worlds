# Project working agreements

This small Python, NumPy, and Matplotlib project learns reinforcement learning
through Sutton and Barto and original, reproducible scientific experiments.

- Prioritize readable RL code, running experiments, and interpreting measured
  results. Keep scientific checks brief; avoid extended test suites, frameworks,
  CI, dashboards, and infrastructure projects unless explicitly requested.
- Explain algorithms for someone learning RL. Separate supplied-model planning
  from learning through experience; distinguish observations from hypotheses.
- Preserve completed results. Save compact numerical outcomes, configuration,
  seeds, runtime, and figures. Update the README with measured results,
  uncertainty, limitations, chapter connections, and reproduction commands.
- Keep environments fixed during an experiment; report mixed and negative
  findings. Use local CPU computation; no paid API calls are needed.
- The user authorizes automatically committing and pushing completed project
  changes. Before each commit or push, verify the working directory is this
  repository and its remote is ReloadLightly/rl-foraging-worlds. Verify the
  published commit and report any concrete authentication blocker.
- Keep the preceding rl-changing-worlds repository intact. Preserve attribution
  and license notices if code from another source is reused.
