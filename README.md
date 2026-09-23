# Decider Jev Doom demo

[Decider MLX fork](https://github.com/edbordin-linktree/decider) ·
[Hugging Face: 2B model card (default)](https://huggingface.co/edbordin-linktree/decider-2b-executorch-mlx) ·
[Hugging Face: 0.8B model card](https://huggingface.co/edbordin-linktree/decider-0.8b-executorch-mlx)

Watch Decider 2B play Doom in your terminal, running locally on an Apple Silicon
Mac through the ExecuTorch MLX backend. No API key, model export or fine-tuning
is needed.

## Quick start

Install [uv](https://docs.astral.sh/uv/) once:

```sh
brew install uv
```

Then download and run the demo:

```sh
git clone https://github.com/edbordin-linktree/decider-jev-doom-demo.git && cd decider-jev-doom-demo && bash run-decider.sh
```

The launcher installs dependencies from a locked uv project, downloads the public
2B model, starts the local Decider HTTP server, and opens the terminal demo.
Press **q** to quit; the launcher also stops its server.

The first run downloads Python dependencies and a **3.51 GiB model**. Later runs
reuse the Hugging Face cache. Download progress appears directly in the terminal,
followed by separate server-starting and game-launching messages. Ctrl+C during
the download cancels it before any server is started.
The first decision is slower while the model loads.

Use a large, true-color terminal. Quit other model or export processes first:
the launcher prevents duplicate instances of this demo, but cannot prevent other
applications from consuming GPU memory.

Tested on an M4 Pro with 48 GB RAM. Smaller-memory Macs and clean Macs without
full Xcode have not yet been validated. Intel Macs and Linux are not supported
by this launcher.

## Run it again

From the repository directory:

```sh
bash run-decider.sh
```

Optional settings:

```sh
bash run-decider.sh --model 0.8b          # Smaller 1.41 GiB model; poorer play in our trials
bash run-decider.sh --seed 47            # Reproducible episode
bash run-decider.sh --port 8001          # If port 8000 is occupied
bash run-decider.sh --record run.jsonl   # Save decisions
```

The default is Decider 2B with a random seed in the `defend_the_center` arena.
The launcher prints the chosen seed; pass it with `--seed` to replay that setup.

## Controls

- **q**: quit.
- **p**: pause or resume.
- **space**: single-step while paused.
- **m**: toggle manual control.
- In manual mode: **a/d** turn, **f** fires. The model still scores each step.

The side panel shows action probabilities, the selected action, HTTP decision
latency, and the state sent to the model.

## If startup takes a while

The first download can take several minutes; watch its progress in the launching
terminal. Once the download finishes, the server uses the cached model offline.
If startup stalls at the server stage, check its log in another terminal:

```sh
tail -f ~/.cache/decider-doom/server.log
```

If you set `XDG_CACHE_HOME`, the log is under `$XDG_CACHE_HOME/decider-doom/`
instead. Dependency installation progress appears in the launching terminal.

An occupied port is an error; stop the existing server or choose `--port 8001`.
If uv reports that it is too old, run `brew upgrade uv`.

## What this demonstrates

This combines two changes:

- **Decider optimized for a MacBook:** precompiled ExecuTorch MLX exports with
  compact FP16 scoring, fused operations, padding removal and parallel suffix scoring.
- **Questions adapted to Decider:** a plain question with described action options
  replaces the harness's original long instructions and examples.

The model receives structured observations from ViZDoom's labels and depth buffer,
not raw images. It chooses each action; there is no scripted action selector or
game-specific fine-tuning. The game pauses during inference. Quality varies by
seed, and this is not a claim of parity with Jev.

See [technical details and separate server/client commands](demo/doom/DECIDER.md)
for the exact question, export limits and development tests.

## Credits

- [Dasein Labs/open-jev](https://github.com/daseinlabs/open-jev): original Doom
  harness, observation descriptions, terminal renderer and HTTP client. This
  repository is a fork; the [MIT license](LICENSE) is retained.
- [Mark Marosi / Mapika, Decider](https://github.com/Mapika/decider): model training,
  checkpoints, calibration and typed-decision API, based on Qwen3.5. Apache-2.0.
- [Decider MLX fork](https://github.com/edbordin-linktree/decider): Apple Silicon
  runtime/export optimizations, Hub model loading and local server support.
- [PyTorch ExecuTorch](https://github.com/pytorch/executorch),
  [Apple MLX](https://github.com/ml-explore/mlx),
  [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) and
  [Freedoom](https://freedoom.github.io/): runtime and game components.
- [TypeSafe / Jev](https://docs.typesafe.ai): inspiration for the decision API and
  demo concept. This is an independent project; no endorsement is implied.

This demo fork adds the launcher and Decider-specific question wording.
Model weights are downloaded separately; proprietary Doom assets are not included.

The original Gemma backend and its documentation remain available in
[README.upstream.md](README.upstream.md). They are not needed to run this demo.
