"""多 ckpt 动作 ensemble（M4 n8）。

动机（PROGRESS 问题 44）：EGL 渲染在同状态下偶发 ~1px 边缘像素差，单策略经视觉网络
→ 毫米级接触混沌放大成成败翻转，单轮 12 变体评测抖动 ±2~4/12。
不同训练时刻的 ckpt 对同一像素扰动的响应不相关，逐帧平均动作可压制该方差，
同时不改变与环境/RL 的接口（.reset()/.action(obs) 与 ACTBase 完全一致）。
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ACTEnsemble:
    """K 个 ACT ckpt 逐帧动作等权平均；各成员保留自己的 temporal ensemble 队列。"""

    def __init__(self, ckpts, ensemble, device):
        from policy.residual_td3_transfer import ACTBase
        self.members = [ACTBase(Path(c) if not isinstance(c, Path) else c,
                                ensemble, device)
                        for c in ckpts]
        self.k = len(self.members)

    def reset(self):
        for m in self.members:
            m.reset()

    def action(self, obs):
        acc = None
        for m in self.members:
            a = m.action(obs)
            acc = a if acc is None else acc + a
        return acc / self.k
