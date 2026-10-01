"""M4 n4：transfer 示教数据集录制（LeRobot v3，断点续录）。

episode = 标称 reset+pick（不录）→ settle 后起录 handoff/catch/release/hold
（proto_transfer run_phases 段，STEP_HOOK 单点注入），只收 hold 成功条。
成功池来自 run_shallow.py 的 transfer_success_pool.json（互异组合；
同组合重复 = 逐比特相同，池内每组合恰录 1 条）。录数 argv 与池验证逐字一致。

跑法（必须 lerobot-env）:
  MUJOCO_GL=egl <lerobot-env>/bin/python scripts/record_transfer.py \
      --cfg configs/record_transfer_marker.yaml [--total N] [--qc-only]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # 本地数据集，resume 勿查 HF hub

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import proto_transfer as PT  # noqa: E402
import record_dataset as RD  # noqa: E402  复用 build_features
from run_shallow import PARAM_SETS  # noqa: E402
from envs import BimanualDexConfig  # noqa: E402
from envs.bimanual_dex_env import BimanualDexEnv  # noqa: E402


def combo_key(c):
    return (f"{c['point'][0]:+.4f},{c['point'][1]:+.4f},"
            f"ps{c['plan_seed']},{c['param_set']}")


def probe_cams():
    env = BimanualDexEnv(config=BimanualDexConfig(max_episode_seconds=20),
                         render_mode="rgb_array")
    cams, size = list(env.cfg.cameras), int(env.cfg.image_size)
    env.close()
    return cams, size


def make_datasets(rc, size):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    names = {"train": rc["train_name"], "val": rc["val_name"]}
    out_root = ROOT / rc["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    ds = {}
    for split, nm in names.items():
        root = out_root / nm
        has_meta = (root / "meta").exists()
        has_eps = (root / "meta" / "tasks.parquet").exists()
        if has_meta and not has_eps:
            # 建库后零条落盘（如 val 空壳）：meta 不全，resume 会 FileNotFoundError
            import shutil
            shutil.rmtree(root)
            has_meta = False
        if has_meta:
            d = LeRobotDataset.resume(f"local/{nm}", root=root)  # 续录必须 resume
        else:
            d = LeRobotDataset.create(f"local/{nm}", fps=int(rc["fps"]),
                                      features=RD.build_features(size), root=root,
                                      robot_type="bimanual_dex")
        ds[split] = d
    return ds


def record_one(c, debug_dir, cams):
    """跑 1 条 proto_transfer（argv 与池验证一致），返回 (success, frames, result)。"""
    x, y = c["point"]
    pargs = dict(PARAM_SETS)[c["param_set"]]
    debug_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    def hook(obs, action):
        frames.append({
            "images": {cam: obs["images"][cam] for cam in cams},
            "state": np.asarray(obs["state"], np.float32),
            "object": np.asarray(obs["object"], np.float32),
            "action": np.asarray(action, np.float32),
        })

    argv = sys.argv
    sys.argv = ["proto_transfer.py", "--x", str(x), "--y", str(y),
                "--seed", "1002", "--plan-seed", str(c["plan_seed"]),
                "--out", str(debug_dir.relative_to(ROOT))] + pargs
    PT.STEP_HOOK = hook
    try:
        rc = PT.main()
    finally:
        PT.STEP_HOOK = None
        sys.argv = argv
    res_file = debug_dir / "result.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    return rc == 0, frames, res


def qc_figure(rc, manifest):
    """数据集体检图：split/帧数/点位覆盖 + 3 条 episode 阶段仿真截图带。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    eps = manifest["episodes"]
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(2, 4, 1)
    n_tr = sum(1 for e in eps if e["split"] == "train")
    ax.bar(["train", "val"], [n_tr, len(eps) - n_tr], color=["#4472a8", "#c08a3e"])
    ax.set_title(f"episode 数（共 {len(eps)}）")
    ax.grid(axis="y", alpha=0.3)

    ax = fig.add_subplot(2, 4, 2)
    ax.hist([e["frames"] for e in eps], bins=15, color="#4472a8")
    ax.set_xlabel("帧数")
    ax.set_title("episode 帧数分布")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 4, 3)
    for pname, mk in (("def", "o"), ("dz05", "s"), ("cs05", "^")):
        xs = [e["point"][0] * 1000 for e in eps if e["param_set"] == pname]
        ys = [e["point"][1] * 1000 for e in eps if e["param_set"] == pname]
        ax.scatter(xs, ys, marker=mk, s=70, label=pname, alpha=0.85)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("点位 × 参数集覆盖")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # 第二行：3 条 episode 的 hold.png + 首条的 catch.png（录制时 proto 按阶段存档）
    show = [eps[i] for i in np.linspace(0, len(eps) - 1, min(3, len(eps))).astype(int)] if eps else []
    for i, e in enumerate(show[:3]):
        png = ROOT / e["debug"] / "hold.png"
        if not png.exists():
            continue
        axi = fig.add_subplot(2, 4, 5 + i)
        axi.imshow(np.asarray(Image.open(png)))
        axi.set_title(f"ep{i} ({e['point'][0]*1000:+.0f},{e['point'][1]*1000:+.0f})"
                      f"ps{e['plan_seed']} {e['param_set']} hold", fontsize=9)
        axi.axis("off")
    if show:
        png = ROOT / show[0]["debug"] / "catch.png"
        if png.exists():
            axi = fig.add_subplot(2, 4, 8)
            axi.imshow(np.asarray(Image.open(png)))
            axi.set_title(f"ep0 {show[0]['param_set']} catch", fontsize=9)
            axi.axis("off")

    fig.suptitle(f"M4 n4 transfer 数据集体检：{len(eps)} 条（{rc['task']}，{rc['fps']}Hz）")
    fig.tight_layout()
    out = ROOT / rc["out_root"] / "transfer_dataset_qc.png"
    fig.savefig(out, dpi=140)
    print(f"[qc] 图: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/record_transfer_marker.yaml")
    ap.add_argument("--total", type=int, default=None)
    ap.add_argument("--qc-only", action="store_true")
    args = ap.parse_args()

    rc = yaml.safe_load((ROOT / args.cfg).read_text())
    if args.total is not None:
        rc["total"] = args.total
    pool = json.loads((ROOT / rc["pool"]).read_text())
    total = min(int(rc["total"]), len(pool))

    mpath = ROOT / rc["manifest"]
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"episodes": []}
    done_keys = {e["key"] for e in manifest["episodes"]}
    fail_keys = {e["key"] for e in manifest.get("failed", [])}
    print(f"[rec] 池 {len(pool)} 组合，目标 {total} 条，已录 {len(done_keys)}，"
          f"已标记失败 {len(fail_keys)}")

    if args.qc_only:
        qc_figure(rc, manifest)
        return

    cams, size = probe_cams()
    ds = make_datasets(rc, size)
    t0 = time.time()
    g = len(manifest["episodes"])
    for c in pool:
        k = combo_key(c)
        if k in done_keys or k in fail_keys:
            continue
        if g >= total:
            break
        split = "val" if (g % int(rc["val_every"])) == int(rc["val_every"]) - 1 else "train"
        debug = ROOT / rc["debug_root"] / f"ep{g:03d}_{k.replace(',', '_')}"
        print(f"[rec] ep{g} {k} -> {split} ...", flush=True)
        try:
            ok, frames, res = record_one(c, debug, cams)
        except Exception as e:  # 单条异常不中断整批
            print(f"[rec] ep{g} 异常: {e}", flush=True)
            manifest.setdefault("failed", []).append({"key": k, "err": str(e)[:200]})
            mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            continue
        if not ok or not frames:
            print(f"[rec] ep{g} hold 未达标（池外漂移），标记失败", flush=True)
            manifest.setdefault("failed", []).append({"key": k, "err": "hold_fail_at_record"})
            mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
            continue
        d = ds[split]
        for fr in frames:
            frame = {"task": rc["task"]}
            for cam in cams:
                frame[f"observation.images.{cam}"] = fr["images"][cam]
            frame["observation.state"] = fr["state"]
            frame["observation.object"] = fr["object"]
            frame["action"] = fr["action"]
            d.add_frame(frame)
        d.save_episode()
        manifest["episodes"].append({
            "key": k, "split": split, "point": c["point"],
            "plan_seed": c["plan_seed"], "param_set": c["param_set"],
            "angle_deg": c.get("angle_deg"), "frames": len(frames),
            "debug": str(debug.relative_to(ROOT)),
            "hold": res.get("phases", [{}])[-1].get("hold")})
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        g += 1
        print(f"[rec] ep{g - 1} 落盘 {split} frames={len(frames)} "
              f"累计 {g}/{total} elapsed={time.time() - t0:.0f}s", flush=True)

    print(f"[rec] DONE 共录 {len(manifest['episodes'])} 条 "
          f"(train={sum(1 for e in manifest['episodes'] if e['split'] == 'train')})")
    qc_figure(rc, manifest)


if __name__ == "__main__":
    main()
