from envs.bimanual_dex_env import BimanualDexConfig, BimanualDexEnv

try:
    import gymnasium as gym

    gym.register(
        id="BimanualDex-v0",
        entry_point="envs.bimanual_dex_env:BimanualDexEnv",
    )
except ImportError:
    pass

__all__ = ["BimanualDexEnv", "BimanualDexConfig"]
