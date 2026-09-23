"""Start one local Decider server, run the terminal demo, then stop the server."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
MODELS = {
    "2b": ("edbordin-linktree/decider-2b-executorch-mlx", "4641daf648b8577b3f7d16c77581e6483289a303"),
    "0.8b": ("edbordin-linktree/decider-0.8b-executorch-mlx", "d1325c1ba36409f6e9efbc431854cb83982582da"),
}


@contextmanager
def single_instance(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Decider demo launcher is running. Quit it first.") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_ready(process, url, model, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Decider server exited; see the server log.")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                health = json.load(response)
            if health.get("ok") and health.get("model") == model:
                return
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        time.sleep(0.5)
    raise RuntimeError("Server startup timed out; see the log for download progress.")


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(args):
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "decider-doom"
    with single_instance(cache / "launcher.lock"):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", args.port))
            except OSError:
                raise RuntimeError(f"Port {args.port} is occupied. Stop that server or select --port.") from None
        model, revision = MODELS[args.model]
        seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
        url = f"http://127.0.0.1:{args.port}"
        log_path = cache / "server.log"
        print(f"Loading {model}. First run downloads the model; later runs use the Hub cache.", flush=True)
        print(f"Server log: {log_path}\nQuit other model processes before continuing.", flush=True)
        print(f"Episode seed: {seed} (replay with --seed {seed})", flush=True)
        with log_path.open("w") as log:
            server = subprocess.Popen([
                sys.executable, "-m", "decider.serve", "--backend", "mlx",
                "--model", model, "--revision", revision,
                "--host", "127.0.0.1", "--port", str(args.port),
            ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                wait_ready(server, url, model)
                command = [sys.executable, str(HERE / "play.py"), "--url", url,
                           "--api", "systemone", "--prompt", "criteria", "--seed", str(seed)]
                if args.record:
                    command += ["--record", args.record]
                game = subprocess.Popen(command)
                try:
                    return game.wait()
                finally:
                    stop(game)
            finally:
                stop(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="2b")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None, help="Episode seed (default: random)")
    parser.add_argument("--record", help="Optional JSONL decision log")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        parser.error("Apple Silicon macOS is required")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        parser.error("Run this command in an interactive terminal")
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
