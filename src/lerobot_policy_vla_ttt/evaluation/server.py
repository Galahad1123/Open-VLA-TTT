"""Single-session server inspired by StarVLA's model server architecture."""

import argparse
import logging
import sys
from pathlib import Path
from threading import Lock

import numpy as np
from vla_ttt_eval.wire import MAX_BYTES, PROTOCOL, pack, unpack
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve


class Session:
    def __init__(self, policy):
        self.policy = policy
        self.next_id = 0
        self.episode_id = None

    def handle(self, request):
        if not isinstance(request, dict):
            raise ValueError("Expected request mapping")
        if type(request.get("request_id")) is not int or request["request_id"] != self.next_id:
            raise ValueError("Unexpected request_id; requests cannot be replayed")
        if request.get("type") == "reset":
            episode = request.get("episode_id")
            seed = request.get("seed")
            if not isinstance(episode, str) or not episode or type(seed) is not int or seed < 0:
                raise ValueError("reset requires episode_id and nonnegative integer seed")
            self.policy.reset(seed)
            self.episode_id = episode
            result = {}
        elif request.get("type") == "infer":
            if self.episode_id is None:
                raise ValueError("reset required before infer")
            if request.get("episode_id") != self.episode_id:
                raise ValueError("Wrong episode_id")
            action = np.asarray(self.policy.infer(request["payload"]), dtype=np.float32)
            if (
                action.shape != (self.policy.metadata["action_dim"],)
                or not np.isfinite(action).all()
            ):
                raise ValueError("Invalid policy action shape or nonfinite values")
            result = {"action": action}
        else:
            raise ValueError("Unknown request type")
        self.next_id += 1
        return result


def make_server(policy, host="127.0.0.1", port=8000):
    lock = Lock()

    def handle(connection):
        if not lock.acquire(blocking=False):
            connection.close(code=1013, reason="Policy already has an active session")
            return
        try:
            connection.send(pack({**policy.metadata, "protocol": PROTOCOL}))
            session = Session(policy)
            for frame in connection:
                request_id = None
                try:
                    request = unpack(frame)
                    request_id = request.get("request_id") if isinstance(request, dict) else None
                    data = session.handle(request)
                    connection.send(pack({"ok": True, "request_id": request_id, "data": data}))
                except Exception as exc:
                    logging.exception("Policy session failed")
                    connection.send(
                        pack({"ok": False, "request_id": request_id, "error": str(exc)})
                    )
                    connection.close(code=1011, reason="Reset required in a new connection")
                    break
        except ConnectionClosed:
            pass
        finally:
            lock.release()

    return serve(handle, host, port, compression=None, max_size=MAX_BYTES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--machine", type=Path, help="Machine TOML (VLM path and HF cache)")
    parser.add_argument("--vlm-path", type=Path, help="Override the checkpoint's VLM location")
    parser.add_argument(
        "--implementation",
        choices=["native", "local", "evo1_groot", "robottt", "mac", "mag", "mal", "aqmem", "t2mem"],
        required=True,
    )
    parser.add_argument("--benchmark", choices=["libero", "robomme"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-action-steps", type=int, default=10)
    args = parser.parse_args()
    from ..portability import configure_cache, read_machine

    try:
        machine = read_machine(args.machine)
        configure_cache(machine, "lerobot_policy_vla_ttt.evaluation.server", sys.argv[1:])
        vlm_path = args.vlm_path or machine.get("assets", {}).get("vlm")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    from .policy import Evo1Inference

    policy = Evo1Inference(
        args.checkpoint,
        args.implementation,
        args.benchmark,
        args.device,
        args.n_action_steps,
        vlm_path=vlm_path,
    )
    print(f"Serving {policy.metadata} on {args.host}:{args.port}", flush=True)
    with make_server(policy, args.host, args.port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
