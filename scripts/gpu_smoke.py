import importlib.metadata as metadata
import sys

import jax
import jax.numpy as jnp
import jaxlib

from jaxgcrl.envs.ant_push import AntPush


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"jax: {jax.__version__}")
    print(f"jaxlib: {jaxlib.__version__}")
    print(f"jax-cuda13-plugin: {_version('jax-cuda13-plugin')}")
    print(f"jax-cuda13-pjrt: {_version('jax-cuda13-pjrt')}")

    devices = jax.devices()
    print("devices:")
    for device in devices:
        print(f"  - {device} platform={device.platform}")

    backend = jax.default_backend()
    print(f"default_backend: {backend}")

    gpu_devices = [device for device in devices if device.platform in {"gpu", "cuda"}]
    if not gpu_devices:
        print("ERROR: no GPU device visible to JAX")
        return 2

    print("initializing AntPush(mjx)")
    env = AntPush(backend="mjx")
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    print("compiling reset")
    state = reset(jax.random.PRNGKey(0))
    state.pipeline_state.q.block_until_ready()
    print(f"reset obs shape: {state.obs.shape}")

    print("compiling step")
    next_state = step(state, jnp.zeros(env.action_size))
    next_state.pipeline_state.q.block_until_ready()
    print(f"step obs shape: {next_state.obs.shape}")

    if jnp.isnan(next_state.obs).any() or jnp.isinf(next_state.obs).any():
        print("ERROR: invalid values in next observation")
        return 3

    print("GPU smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
