"""Quick visual check: render scene from several viewpoints + FK info."""
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(ROOT, "assets/bimanual_scene.xml")
OUT = os.path.join(ROOT, "verify_out")
os.makedirs(OUT, exist_ok=True)

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, 480, 640)

# add temporary overview cameras
def add_cam(xml_pos_quat):
    import re
    return None

views = {
    "tune_head": ("head",),
    "tune_top": None,
    "tune_front": None,
    "tune_side": None,
}
extra_cams = {
    "tune_top": {"pos": "0 0 2.2", "xyaxes": "1 0 0 0 1 0", "fovy": "50"},
    "tune_front": {"pos": "0 -2.0 1.05", "xyaxes": "1 0 0 0 0 1", "fovy": "50"},
    "tune_side": {"pos": "-2.0 0 1.05", "xyaxes": "0 1 0 0 0 -1", "fovy": "50"},
    "tune_iso": {"pos": "-1.2 -1.2 1.1", "target": "0 0 0.45", "fovy": "50"},
}

# hack: temporarily attach extra cameras via spec
spec = mujoco.MjSpec.from_file(SCENE)
worldbody = spec.worldbody

def xyaxes_to_quat(x, y):
    x = np.array(x, float); x /= np.linalg.norm(x)
    y = np.array(y, float); y -= x * np.dot(x, y); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R = np.stack([x, y, z], axis=1)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [(R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(1 + R[i,i] - R[j,j] - R[k,k]) * 2
        q = np.zeros(4)
        q[i] = s/4; q[3] = (R[k,j]-R[j,k])/s
        q[j] = (R[j,i]+R[i,j])/s; q[k] = (R[k,i]+R[i,k])/s
    qx, qy, qz, qw = q
    return [qw, qx, qy, qz]  # mujoco order: w x y z

for name, attrs in extra_cams.items():
    cam = worldbody.add_camera(name=name)
    cam.pos = [float(x) for x in attrs["pos"].split()]
    if "target" in attrs:
        pos = np.array(attrs["pos"].split(), float)
        tgt = np.array(attrs["target"].split(), float)
        zax = pos - tgt
        xax = np.cross([0, 0, 1], zax)
        cam.quat = xyaxes_to_quat(xax, np.cross(zax, xax))
    else:
        cam.quat = xyaxes_to_quat(attrs["xyaxes"].split()[:3], attrs["xyaxes"].split()[3:])
    cam.fovy = float(attrs["fovy"])
spec.compile()
model2 = spec.compile()
data2 = mujoco.MjData(model2)
k2 = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model2, data2, k2)
mujoco.mj_forward(model2, data2)

r2 = mujoco.Renderer(model2, 480, 640)
for cam in ["tune_head", "tune_top", "tune_front", "tune_side", "tune_iso",
            "left_wrist", "right_wrist"]:
    cid = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    r2.update_scene(data2, camera=cid)
    img = r2.render()
    from PIL import Image
    Image.fromarray(img).save(os.path.join(OUT, f"tune_{cam}.png"))
    print("saved", cam)

# FK info at home
for name in ["l_attachment_site", "r_attachment_site", "lh_palm", "rh_palm",
             "lh_forearm", "rh_forearm", "pen"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE if "site" in name else
                            mujoco.mjtObj.mjOBJ_BODY, name)
    xpos = data.site(bid).xpos if "site" in name else data.body(bid).xpos
    print(name, np.round(xpos, 3))
