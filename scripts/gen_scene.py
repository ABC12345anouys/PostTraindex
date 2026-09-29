"""Generate assets/bimanual_scene.xml from third-party MJCF sources.

Sources:
  - third_party/panda/Panda.xml          (robopal Franka Panda, motor actuators)
  - third_party/mujoco_menagerie/shadow_hand/{right,left}_hand.xml

The generator prefixes all names per side (l_/r_, rh_/lh_), replaces Panda
motor actuators with position actuators, mounts each Shadow hand on the Panda
flange (attachment body), and emits one self-contained scene with the 54
actuators in the paper's order: [l_arm7, l_hand20, r_arm7, r_hand20].

Edit the constants below (base poses, hand mount orientation, cameras), then:
  python scripts/gen_scene.py
"""

import os
import re
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDA_XML = os.path.join(ROOT, "third_party/panda/Panda.xml")
HAND_DIR = os.path.join(ROOT, "third_party/mujoco_menagerie/shadow_hand")
OUT_XML = os.path.join(ROOT, "assets/bimanual_scene.xml")

# ---------------------------------------------------------------------------
# Layout constants (world frame: table center at origin, z up)
# ---------------------------------------------------------------------------
TABLE_POS = (0.0, 0.0, 0.45)      # box halfsize -> top at z = 0.50
TABLE_SIZE = (0.45, 0.40, 0.05)
TABLE_TOP_Z = TABLE_POS[2] + TABLE_SIZE[2]

PANDA_BASE = {
    "l": {"pos": (-0.62, 0.0, 0.0), "quat": "1 0 0 0"},
    "r": {"pos": (0.62, 0.0, 0.0), "quat": "0 0 0 1"},  # yaw pi: face -x
}

# Collision-free "ready" keyframe (found by CEM over arm joints, hands zero):
# palms ~0.30 m to each side at z~0.69, fingers point down, fingertips 3 cm
# above the table. The naive "arm straight toward center" pose penetrates the
# table / own arm links / the opposite flange, so do not revert to it.
ARM_HOME = {
    "l": [-0.5841, -0.3123, 0.2378, -1.3030, -0.9488, 1.8973, 0.4535],
    "r": [-0.5929, -0.3362, 0.1876, -1.3412, -0.8648, 2.2275, 0.3358],
}

# Shadow hand root (forearm body) offset relative to the Panda flange
# (attachment body frame: pos "0 0 0.107", quat 45deg about x).
FOREARM_OFFSET = {
    "rh": {"pos": "0 0 0.02", "quat": "0 0 0.7071068 0.7071068"},
    "lh": {"pos": "0 0 0.02", "quat": "0 0 0.7071068 0.7071068"},
}

ARM_KP = [300.0, 300.0, 300.0, 300.0, 100.0, 100.0, 100.0]
ARM_FORCE = [(87.0, 87.0)] * 5 + [(12.0, 12.0)] * 2

HEAD_CAMERA = {"pos": "0 0 1.6", "xyaxes": "1 0 0 0 1 0", "fovy": "60"}
# wrist cameras: mounted on the dorsal side (+y) of the palm body, looking
# past the fingertips (local +z), tilted ~15 deg toward the palm face (-y).
# Palm-frame geometry: knuckles at z~0.095, distal tips at z~0.17. The view at
# the ready pose looks straight down; during the inward approach the palm
# rotates toward center and the pen enters above the fingertips in frame.
WRIST_CAMERA = {"pos": "0.0 0.055 -0.015", "xyaxes": "1 0 0 0 -0.966 -0.259", "fovy": "75"}

PEN_SPAWN = (0.0, 0.0, TABLE_TOP_Z + 0.009)  # fallback pose; env randomizes
PEN_QUAT_FLAT = "0.7071068 0 0.7071068 0"    # cylinder axis z -> x (lying flat)


def serialize(elem, indent="  "):
    """Serialize an ElementTree element as an indented string."""
    from xml.dom import minidom

    rough = ET.tostring(elem, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent=indent)
    lines = [ln for ln in pretty.splitlines() if ln.strip() and not ln.startswith("<?")]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Panda arm processing
# ---------------------------------------------------------------------------
PANDA_CLASSES = ("panda", "panda/visual", "panda/collision")
PANDA_MATERIALS = ("white", "off_white", "dark_grey", "green", "light_blue")


def process_arm(prefix):
    """Return dict with asset/default/worldbody/actuator XML strings."""
    tree = ET.parse(PANDA_XML)
    root = tree.getroot()

    def pf(name):
        return f"{prefix}{name}"

    # --- collect mesh names (explicit + from file basenames) ---
    mesh_names = set()
    for mesh in root.iter("mesh"):
        mesh.attrib["file"] = "../third_party/panda/" + mesh.attrib["file"]
        if "name" in mesh.attrib:
            mesh_names.add(mesh.attrib["name"])
        else:
            base = os.path.splitext(os.path.basename(mesh.attrib["file"]))[0]
            mesh.attrib["name"] = pf(base)
            mesh_names.add(base)

    named = re.compile(r"^(link\d+(?:_c\d*|_c|_\d+)?|joint\d+|attachment(?:_site)?|actuator\d+)$")

    for elem in root.iter():
        # element names
        if "name" in elem.attrib and named.match(elem.attrib["name"]):
            elem.attrib["name"] = pf(elem.attrib["name"])
        elif elem.tag == "material" and "name" in elem.attrib:
            elem.attrib["name"] = pf(elem.attrib["name"])
        # name references
        for attr in ("mesh", "material", "joint"):
            if elem.attrib.get(attr) in mesh_names or (
                attr == "material" and elem.attrib.get(attr) in PANDA_MATERIALS
            ):
                elem.attrib[attr] = pf(elem.attrib[attr])
        # default classes
        for attr in ("class", "childclass"):
            v = elem.attrib.get(attr)
            if v in PANDA_CLASSES:
                elem.attrib[attr] = pf(v)

    asset = root.find("asset")
    default = root.find("default")
    # --- worldbody: wrap in a positioned base body ---
    worldbody = root.find("worldbody")
    base = ET.Element("body", name=f"{prefix}base",
                      pos=" ".join(f"{v}" for v in PANDA_BASE[prefix.rstrip("_")]["pos"]),
                      quat=PANDA_BASE[prefix.rstrip("_")]["quat"])
    for child in list(worldbody):
        worldbody.remove(child)
        base.append(child)
    worldbody.append(base)

    # --- actuators: motor -> position ---
    old_act = root.find("actuator")
    joint_ranges = {}
    for j in root.iter("joint"):
        if "name" in j.attrib:
            joint_ranges[j.attrib["name"]] = j.attrib.get("range", "-2.8973 2.8973")
    actuator = ET.Element("actuator")
    for i in range(1, 8):
        lo, hi = (float(x) for x in joint_ranges[pf(f"joint{i}")].split())
        ET.SubElement(
            actuator, "position",
            name=pf(f"actuator{i}"), joint=pf(f"joint{i}"),
            ctrlrange=f"{lo} {hi}", kp=str(ARM_KP[i - 1]),
            forcerange=f"{-ARM_FORCE[i-1][0]} {ARM_FORCE[i-1][1]}",
        )
    root.remove(old_act)

    return {
        "asset": "\n".join(serialize(c) for c in list(asset)),
        "default": "\n".join(serialize(c) for c in list(default)),
        "worldbody": "\n".join(serialize(c) for c in list(worldbody)),
        "actuator": "\n".join(serialize(c) for c in list(actuator)),
    }


# ---------------------------------------------------------------------------
# Shadow hand processing
# ---------------------------------------------------------------------------
HAND_CLASSES = (
    "right_hand", "left_hand", "wrist", "wrist_y", "wrist_x",
    "thumb", "thbase", "thproximal", "thhub", "thmiddle", "thdistal",
    "metacarpal", "knuckle", "proximal", "middle_distal",
    "plastic", "plastic_visual", "plastic_collision",
)
HAND_MATERIALS = ("black", "gray", "metallic")


def process_hand(side):
    """side: 'rh' or 'lh'. Returns dict of XML section strings."""
    fname = "right_hand.xml" if side == "rh" else "left_hand.xml"
    tree = ET.parse(os.path.join(HAND_DIR, fname))
    root = tree.getroot()
    cls_root = "right_hand" if side == "rh" else "left_hand"
    hand_cls = f"{side}_hand"

    mesh_names = {}
    for mesh in root.iter("mesh"):
        if "file" not in mesh.attrib:
            continue
        mesh.attrib["file"] = "../third_party/mujoco_menagerie/shadow_hand/assets/" + mesh.attrib["file"]
        base = os.path.splitext(os.path.basename(mesh.attrib["file"]))[0]
        mesh.attrib["name"] = f"{side}_{base}"
        mesh_names[base] = mesh.attrib["name"]

    for elem in root.iter():
        for attr in ("class", "childclass"):
            v = elem.attrib.get(attr)
            if v in HAND_CLASSES:
                elem.attrib[attr] = hand_cls if v == cls_root else f"{side}_{v}"
        mat = elem.attrib.get("material")
        if mat in HAND_MATERIALS:
            elem.attrib["material"] = f"{side}_{mat}"
        mesh_ref = elem.attrib.get("mesh")
        if mesh_ref in mesh_names:
            elem.attrib["mesh"] = mesh_names[mesh_ref]
        if elem.tag == "material" and elem.attrib.get("name") in HAND_MATERIALS:
            elem.attrib["name"] = f"{side}_{elem.attrib['name']}"
        if elem.tag == "site" and elem.attrib.get("name") == "grasp_site":
            # 右手 XML 自带 grasp_site；统一标记移除，之后由两只手掌同位置
            # 重新挂载，保证 lh/rh 的 IK 目标帧完全一致。
            elem.attrib["_drop"] = "1"

    # 偏离论文/Menagerie 标注：Shadow 原配腕位置伺服偏弱（kp 10/8、无速度反馈），
    # 握拳时腱路交叉张力使腕稳态偏差达 0.2rad 且欠阻尼振荡，脚本专家无法可靠
    # 对准。统一提高腕 kp 并加 kv 速度反馈（仅改伺服参数，不动关节/几何）。
    wrist_tune = {f"{side}_wrist_y": ("60", "5", "-12 12"),
                  f"{side}_wrist_x": ("40", "4", "-6 6")}
    for pos in root.iter("position"):
        tune = wrist_tune.get(pos.attrib.get("class"))
        if tune is not None:
            pos.attrib["kp"], pos.attrib["kv"], pos.attrib["forcerange"] = tune

    # 偏离论文/Menagerie 标注：Shadow 原配手指位置伺服 kp 0.4~1.5、力限 ±1~3Nm，
    # 实测 MFJ3 等指令 1.47rad 时持续力饱和、2.5s 只走到 0.02rad，无法完成捏取。
    # 统一提高手指 kp/力限（腕执行器除外），仅改伺服参数以匹配「位置控制手」设定。
    finger_tune = {
        "THJ5": ("2.0", "-5 5"), "THJ4": ("4.0", "-6 6"),
        "THJ3": ("2.0", "-2 2"), "THJ2": ("4.0", "-3 3"),
        "THJ1": ("3.0", "-3 3"),
        "FFJ4": ("3.0", "-2 2"), "FFJ3": ("4.0", "-3 3"), "FFJ0": ("2.0", "-2 2"),
        "MFJ4": ("3.0", "-2 2"), "MFJ3": ("4.0", "-3 3"), "MFJ0": ("2.0", "-2 2"),
        "RFJ4": ("3.0", "-2 2"), "RFJ3": ("4.0", "-3 3"), "RFJ0": ("2.0", "-2 2"),
        "LFJ5": ("3.0", "-2 2"), "LFJ4": ("3.0", "-2 2"),
        "LFJ3": ("4.0", "-3 3"), "LFJ0": ("2.0", "-2 2"),
    }
    for pos in root.iter("position"):
        nm = pos.attrib.get("name", "")
        if nm.startswith(f"{side}_A_"):
            tune = next((v for k, v in finger_tune.items() if nm.endswith(k)), None)
            if tune is not None:
                pos.attrib["kp"], pos.attrib["forcerange"] = tune

    def _strip(node):
        for child in list(node):
            if child.tag == "site" and child.attrib.get("_drop") == "1":
                node.remove(child)
            else:
                _strip(child)
    _strip(root)

    default = None
    for d in root.iter("default"):
        if d.attrib.get("class") == hand_cls:
            default = d
            break
    assert default is not None

    worldbody = root.find("worldbody")
    forearm = None
    for b in worldbody.iter("body"):
        if b.attrib["name"] == f"{side}_forearm":
            forearm = b
            break
    assert forearm is not None
    forearm.attrib["pos"] = FOREARM_OFFSET[side]["pos"]
    forearm.attrib["quat"] = FOREARM_OFFSET[side]["quat"]

    cam_parent = None
    for b in worldbody.iter("body"):
        if b.attrib.get("name") == f"{side}_palm":
            cam_parent = b
            break
    assert cam_parent is not None
    cam = ET.SubElement(cam_parent, "camera",
                        name=f"{'left' if side=='lh' else 'right'}_wrist")
    cam.attrib.update({"pos": WRIST_CAMERA["pos"], "xyaxes": WRIST_CAMERA["xyaxes"],
                       "fovy": WRIST_CAMERA["fovy"]})
    site = ET.SubElement(cam_parent, "site", name=f"{side}_grasp_site",
                         pos="0 -0.035 0.09", size="0.005", rgba="0.1 0.9 0.1 0.6")

    return {
        "asset": "\n".join(serialize(c) for c in list(root.find("asset"))),
        "default": serialize(default),
        "worldbody": serialize(forearm),
        "actuator": "\n".join(serialize(c) for c in list(root.find("actuator"))),
        "contact": "\n".join(serialize(c) for c in list(root.find("contact"))),
        "tendon": "\n".join(serialize(c) for c in list(root.find("tendon"))),
    }


# ---------------------------------------------------------------------------
# Assemble scene
# ---------------------------------------------------------------------------
def main():
    arm = {"l": process_arm("l_"), "r": process_arm("r_")}
    hand = {s: process_hand(s) for s in ("rh", "lh")}

    pen_qpos = " ".join(
        [f"{PEN_SPAWN[0]} {PEN_SPAWN[1]} {PEN_SPAWN[2]}", PEN_QUAT_FLAT])
    arm_home_l = " ".join(str(v) for v in ARM_HOME["l"])
    arm_home_r = " ".join(str(v) for v in ARM_HOME["r"])
    zero24 = " ".join(["0"] * 24)
    # joint order in compiled model: pen (freejoint) comes first in worldbody
    qpos = f"{pen_qpos} {arm_home_l} {zero24} {arm_home_r} {zero24}"
    zero20 = " ".join(["0"] * 20)
    ctrl = f"{arm_home_l} {zero20} {arm_home_r} {zero20}"

    actuator_order = (
        arm["l"]["actuator"], hand["lh"]["actuator"],
        arm["r"]["actuator"], hand["rh"]["actuator"],
    )

    def mount_hand(arm_xml, hand_xml, side):
        """Insert the hand forearm body inside the arm's attachment body."""
        pat = re.compile(r'(<body name="' + side + r'_attachment"[^>]*>)')
        out, n = pat.subn(lambda m: m.group(1) + "\n" + hand_xml, arm_xml, count=1)
        assert n == 1, f"attachment body not found for {side}"
        return out

    worldbody_order = (
        mount_hand(arm["l"]["worldbody"], hand["lh"]["worldbody"], "l"),
        mount_hand(arm["r"]["worldbody"], hand["rh"]["worldbody"], "r"),
    )

    scene = f"""<mujoco model="bimanual_dex">
  <compiler angle="radian" autolimits="true"/>

  <visual>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5" specular="0.15 0.15 0.15"/>
    <quality shadowsize="1024"/>
    <rgba haze="0.35 0.35 0.35 1"/>
  </visual>

  <option integrator="implicitfast" solver="Newton" iterations="200" ls_iterations="50"
    ls_tolerance="1e-4" cone="elliptic" impratio="10" timestep="0.002"/>

  <size memory="100M"/>

  <default>
    <geom solimp="0.9 0.95 0.001 0.5 2" solref="0.01 1"/>
    <default class="table">
      <geom type="box" material="table_wood" friction="1.0 0.3 0.05"/>
    </default>
{arm['l']['default']}
{arm['r']['default']}
{hand['lh']['default']}
{hand['rh']['default']}
  </default>

  <asset>
    <material name="table_wood" rgba="0.6 0.45 0.3 1" specular="0.2" shininess="0.1"/>
{arm['l']['asset']}
{arm['r']['asset']}
{hand['lh']['asset']}
{hand['rh']['asset']}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.85 0.85 0.87 1" friction="1.2 0.3 0.05"/>
    <geom name="table" class="table" size="{' '.join(str(v) for v in TABLE_SIZE)}" pos="{' '.join(str(v) for v in TABLE_POS)}"/>

    <body name="pen" pos="{PEN_SPAWN[0]} {PEN_SPAWN[1]} {PEN_SPAWN[2]}" quat="{PEN_QUAT_FLAT}">
      <freejoint name="pen_joint"/>
      <geom name="pen_geom" type="cylinder" size="0.008 0.075" mass="0.015"
        rgba="0.85 0.08 0.08 1" friction="1.0 0.3 0.05" solref="0.01 1"/>
    </body>

    <light name="key_light" directional="true" pos="-0.5 -0.7 2.2"
      dir="0.25 0.35 -1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <light name="fill_light" directional="true" pos="0.6 0.5 1.8"
      dir="-0.3 -0.25 -1" diffuse="0.3 0.3 0.3" specular="0.05 0.05 0.05"/>

    <camera name="head" pos="{HEAD_CAMERA['pos']}" xyaxes="{HEAD_CAMERA['xyaxes']}" fovy="{HEAD_CAMERA['fovy']}"/>
{chr(10).join(worldbody_order)}
  </worldbody>

  <contact>
{hand['lh']['contact']}
{hand['rh']['contact']}
  </contact>

  <tendon>
{hand['lh']['tendon']}
{hand['rh']['tendon']}
  </tendon>

  <actuator>
{chr(10).join(actuator_order)}
  </actuator>

  <keyframe>
    <key name="home" qpos="{qpos}" ctrl="{ctrl}"/>
  </keyframe>
</mujoco>
"""
    os.makedirs(os.path.dirname(OUT_XML), exist_ok=True)
    with open(OUT_XML, "w") as f:
        f.write(scene)
    print(f"wrote {OUT_XML} ({len(scene)} bytes)")

    import mujoco

    model = mujoco.MjModel.from_xml_path(OUT_XML)
    print(f"nq={model.nq} nv={model.nv} nactuator={model.nu} ncam={model.ncam}")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    print("actuators:", names)


if __name__ == "__main__":
    main()
