"""Launcher lifecycle tests; no model downloads or inference."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import launch_decider as launcher


def test_lock_prevents_second_model(tmp_path):
    path = tmp_path / "launcher.lock"
    with launcher.single_instance(path):
        with pytest.raises(RuntimeError, match="Another"):
            with launcher.single_instance(path):
                pytest.fail("Second launcher entered")
    with launcher.single_instance(path):
        pass


def test_ready_checks_model_identity():
    process = Mock()
    process.poll.return_value = None
    responses = [io.BytesIO(json.dumps(h).encode()) for h in (
        {"ok": True, "model": "wrong"}, {"ok": True, "model": "expected"})]
    with patch.object(launcher.urllib.request, "urlopen", side_effect=responses) as request, \
         patch.object(launcher.time, "sleep"):
        launcher.wait_ready(process, "http://localhost:8000", "expected")
    assert request.call_count == 2


def test_dead_server_fails_promptly():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match="exited"):
        launcher.wait_ready(process, "http://localhost:8000", "expected")


def test_stop_escalates_only_owned_process():
    process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [launcher.subprocess.TimeoutExpired("server", 15), 0]
    launcher.stop(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


@pytest.mark.parametrize("startup_error", [False, True])
@pytest.mark.parametrize("seed", [None, 0, 37])
def test_run_cleans_up_server(tmp_path, monkeypatch, startup_error, seed):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    server, game = Mock(), Mock()
    server.poll.return_value = None
    game.poll.return_value = 0
    game.wait.return_value = 0
    args = SimpleNamespace(model="2b", port=8000, seed=seed, record=None)
    random_seed = Mock(return_value=12345)
    monkeypatch.setattr(launcher.secrets, "randbelow", random_seed)
    with patch.object(launcher.socket, "socket"), \
         patch.object(launcher.subprocess, "Popen", side_effect=[server, game]) as popen, \
         patch.object(launcher, "wait_ready", side_effect=RuntimeError("startup") if startup_error else None):
        if startup_error:
            with pytest.raises(RuntimeError, match="startup"):
                launcher.run(args)
            assert popen.call_count == 1
        else:
            assert launcher.run(args) == 0
            server_command = popen.call_args_list[0].args[0]
            assert launcher.MODELS["2b"][0] in server_command
            assert launcher.MODELS["2b"][1] in server_command
            game_command = popen.call_args_list[1].args[0]
            assert game_command[game_command.index("--prompt") + 1] == "criteria"
            assert game_command[game_command.index("--seed") + 1] == str(12345 if seed is None else seed)
        server.terminate.assert_called_once()
        if seed is None:
            random_seed.assert_called_once_with(2**31)
        else:
            random_seed.assert_not_called()


def test_http_contract_round_trip():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    from client import Client
    from decider_prompt import apply_prompt

    bodies = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/systemone"
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            result = {"answers": {"action": {"choice": "turn right", "confidence": 0.8,
                      "probabilities": {"attack": 0.1, "turn left": 0.1, "turn right": 0.8}}}, "usage": {}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        state = {"monsters": [], "health": 100}
        body = apply_prompt({"state": state, "questions": {"action": {}}}, "criteria")
        question = body["questions"]["action"]
        client = Client(f"http://127.0.0.1:{server.server_port}", api="systemone")
        result = client.decide_systemone(state, question["instructions"], question["criteria"])
        assert result.best == "turn right"
        assert bodies[0]["state"] == state
        assert bodies[0]["questions"]["action"]["criteria"] == question["criteria"]
        # Exercise real ViZDoom observations/actions, with HTTP inference stubbed.
        from game import Doom
        from play import decide
        doom = Doom("defend_the_center", seed=37)
        try:
            doom.new_episode()
            last = None
            for _ in range(3):
                decision = decide(client, "systemone", doom.snapshot(), doom, last, "criteria")
                assert decision.best in doom.actions
                doom.step(decision.best, 5)
                last = decision.best
            assert len(bodies) == 4
            assert "monsters" in bodies[-1]["state"]
        finally:
            doom.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
