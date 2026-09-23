# Doom with local Decider on Apple Silicon

This fork connects Dasein Labs' terminal Doom harness to a local Decider HTTP
server using precompiled ExecuTorch MLX models. It defaults to the 2B model and
the `defend_the_center` scenario. No API key, model export or fine-tuning is needed.

## Run

Install [uv](https://docs.astral.sh/uv/) once with `brew install uv`, then:

```sh
git clone https://github.com/edbordin-linktree/open-jev.git && cd open-jev && bash run-decider.sh
```

The launcher uses a dedicated uv project and committed lockfile in `demo/doom`.
It installs the runtime, starts the Decider server on `127.0.0.1:8000`, waits for
readiness, and launches Doom in your terminal. Quitting stops the server it
started. It refuses an occupied port or a second launcher instance.

The first run downloads Python dependencies and the public 3.51 GiB 2B model.
Models are cached by Hugging Face; later runs reuse them. The first decision
loads the model into memory and is slower than subsequent decisions. Download
and server progress are in `~/.cache/decider-doom/server.log` (or under
`$XDG_CACHE_HOME` if set). Use `tail -f` in another terminal while waiting.

Run one model at a time: quit other inference or export processes first. The
launcher lock covers this demo, not unrelated applications. Tested hardware:
M4 Pro, 48 GB RAM. Lower-memory Macs and clean Macs without full Xcode are not
yet validated. The models are precompiled, but that is not proof that every
runtime dependency works without Xcode.

Keys: **q** quit, **p** pause/resume, **space** single-step while paused,
**m** toggle manual control. Use a large terminal with true-color support.

```sh
bash run-decider.sh --model 0.8b       # 1.41 GiB; poorer decision quality in our trials
bash run-decider.sh --seed 47 --port 8001
bash run-decider.sh --record run.jsonl
```

## What changed

The model sees the same structured state from ViZDoom's labels and depth buffer,
not raw pixels. We replaced the original long instructions/examples with a plain
question and described options, matching Decider's typed-question interface:

> Which action best matches the current visible monsters? Use their positions, not last_action.

- `attack`: A visible monster is dead center, in the crosshair. Shoot it.
- `turn left`: The closest visible monster is left of center. Turn left to face it.
- `turn right`: No monsters are visible, or the closest visible monster is right of center. Turn right to search or face it.

The model selects the action; these descriptions are not a scripted controller.
No game-specific training or answer caching is used. The game pauses while each
HTTP decision is made. The panel's latency includes the HTTP round trip, unlike
earlier direct-call measurements. Quality varies by seed; this is a demonstration,
not a benchmark or a claim of parity with Jev.

The compact exports support at most 575 rendered tokens per row, with separate
512-token prefix and 63-token suffix bounds. They do not include a large-context
fallback. The adjusted prompt is scoped to `defend_the_center`; other scenarios
may exceed those bounds and are not covered by this launcher.

## Separate server and client

From the repository root, in two terminals:

```sh
uv run --project demo/doom --locked --no-dev python -m decider.serve --backend mlx \
  --model edbordin-linktree/decider-2b-executorch-mlx \
  --revision 4641daf648b8577b3f7d16c77581e6483289a303
```

```sh
uv run --project demo/doom --locked --no-dev python demo/doom/play.py \
  --api systemone --prompt criteria --seed 37
```

These manual commands do not use the launcher's single-instance lock or cleanup.
The default upstream `play.py` behavior remains available with `--prompt original`.

## Credits and licenses

- [Dasein Labs/open-jev](https://github.com/daseinlabs/open-jev): original terminal
  harness, ViZDoom integration, observation descriptions, renderer and HTTP client.
  Original MIT license and upstream documentation are retained.
- [Mark Marosi / Mapika, Decider](https://github.com/Mapika/decider): model training,
  checkpoints, calibration and typed-decision API, based on Qwen3.5. Apache-2.0.
- [Decider MLX fork](https://github.com/edbordin-linktree/decider): optional
  ExecuTorch MLX backend, compact FP16 exports, fused operations, padding removal,
  parallel suffix scoring, automatic Hub loading and local server support.
- [PyTorch ExecuTorch](https://github.com/pytorch/executorch),
  [Apple MLX](https://github.com/ml-explore/mlx),
  [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) and
  [Freedoom](https://freedoom.github.io/): runtime and game components, under their
  respective licenses. Game assets come from ViZDoom, not this fork.
- [TypeSafe / Jev](https://docs.typesafe.ai): inspiration for the decision API and
  demo concept. This is an independent reproduction, with no claimed endorsement.

This fork adds the local launcher and Decider-specific question wording. No model
weights or proprietary Doom assets are included in Git.

## Development checks

```sh
uv run --project demo/doom --locked python -m pytest demo/doom/test_local_client.py demo/doom/test_launch_decider.py
```

These tests cover prompts, HTTP transport and launcher cleanup without loading a model.
