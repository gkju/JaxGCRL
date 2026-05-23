import importlib

import jax
import jax.numpy as jnp
import pytest

import jaxgcrl
from jaxgcrl.utils.env import MetricsRecorder, create_env


def test_top_level_import_is_lazy():
    assert jaxgcrl.envs is importlib.import_module("jaxgcrl.envs")


def test_agent_exports_import_with_current_brax():
    from jaxgcrl.agents import CRL, PPO, SAC, TD3

    assert CRL.__name__ == "CRL"
    assert PPO.__name__ == "PPO"
    assert SAC.__name__ == "SAC"
    assert TD3.__name__ == "TD3"


def test_train_extra_error_is_actionable():
    recorder = MetricsRecorder(
        total_env_steps=1,
        metrics_to_collect=[],
        exp_dir=".",
        exp_name="test",
        mode="online",
    )

    with pytest.raises(ImportError, match=r"pip install -e \.\[train\]"):
        recorder.log_wandb()


@pytest.mark.parametrize("env_name", ["reacher", "ant_push", "arm_reach", "arm_push_easy"])
def test_short_jitted_reset_step_smoke(env_name):
    env = create_env(env_name=env_name)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    state.obs.block_until_ready()

    action = jnp.zeros(env.action_size)
    next_state = jax.jit(env.step)(state, action)
    next_state.obs.block_until_ready()

    assert next_state.obs.shape == state.obs.shape
    assert not jnp.isnan(next_state.obs).any()
