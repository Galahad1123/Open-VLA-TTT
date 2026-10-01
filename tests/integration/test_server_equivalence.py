from threading import Thread

import numpy as np
import pytest
from vla_ttt_eval.client import PolicyClient

from lerobot_policy_vla_ttt.evaluation.policy import Evo1Inference
from lerobot_policy_vla_ttt.evaluation.server import make_server


@pytest.mark.integration
def test_real_checkpoint_network_actions_match_direct_and_local(evo1_checkpoint):
    checkpoint, _ = evo1_checkpoint
    observation = {
        "images": {key: np.zeros((256, 256, 3), np.uint8) for key in ("image", "image2")},
        "state": np.zeros(8, np.float32),
        "task": "pick up the black bowl",
    }
    native = Evo1Inference(checkpoint, "native", "libero", n_action_steps=2)
    native.reset(123)
    direct = [native.infer(observation) for _ in range(3)]
    with make_server(native, "127.0.0.1", 0) as service:
        thread = Thread(target=service.serve_forever, daemon=True)
        thread.start()
        try:
            with PolicyClient(
                f"ws://127.0.0.1:{service.socket.getsockname()[1]}", benchmark="libero"
            ) as client:
                client.reset("a", 123)
                for expected in direct:
                    np.testing.assert_allclose(
                        client.infer(observation), expected, atol=5e-3, rtol=5e-3
                    )
                client.reset("b", 123)
                np.testing.assert_allclose(
                    client.infer(observation), direct[0], atol=5e-3, rtol=5e-3
                )
        finally:
            service.shutdown()
            thread.join(timeout=5)
    del native
    local = Evo1Inference(checkpoint, "local", "libero", n_action_steps=2)
    local.reset(123)
    for expected in direct:
        np.testing.assert_allclose(local.infer(observation), expected, atol=5e-3, rtol=5e-3)
