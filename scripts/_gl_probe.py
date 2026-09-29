import mujoco, numpy as np
from mujoco import egl
from OpenGL import GL

xml = """
<mujoco>
  <visual><headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4"/></visual>
  <worldbody>
    <camera name="cam" pos="1 1 1" xyaxes="1 0 0 0 1 0" mode="fixed"/>
    <geom type="box" size="0.5 0.5 0.5" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>
"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
ctx = egl.GLContext(224, 224)
ctx.make_current()
print("GL_RENDERER:", GL.glGetString(GL.GL_RENDERER))
print("GL_VERSION:", GL.glGetString(GL.GL_VERSION))
con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
print("offscreen:", con.offWidth, con.offHeight)
scene = mujoco.MjvScene(m, 1000)
mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, mujoco.MjvCamera(), 7, scene)
print("ngeom:", scene.ngeom)
vp = mujoco.MjrRect(0, 0, 224, 224)
mujoco.mjr_render(vp, scene, con)
print("glError after render:", GL.glGetError())
rgb = np.zeros((224, 224, 3), dtype=np.uint8)
dep = np.zeros((224, 224), dtype=np.float32)
mujoco.mjr_readPixels(rgb, dep, vp, con)
print("rgb max:", rgb.max(), "depth min/max:", round(float(dep.min()), 3), round(float(dep.max()), 3))
