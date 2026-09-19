import sys, math, numpy as np, mujoco
sys.path.insert(0, sys.argv[1])
from abc_minimal import ladder as L
from abc_minimal.config import LadderSimConfig
from abc_minimal.mug_flip import _quat_mul
scene = LadderSimConfig()
for name in ["w02_wipe_plate_sponge", "w02b_wipe_plate_sponge", "w01b_wipe_board_towel", "w03_wipe_spatula_sponge"]:
    task = L.get_task(name); seed = 20260514
    draw = L.sample_draw(np.random.default_rng(seed), scene, task, seed)
    m = mujoco.MjModel.from_xml_string(L.ladder_scene_xml(scene, task, draw)); d = mujoco.MjData(m)
    def setfj(j, pos, quat):
        a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]; d.qpos[a:a+3] = pos; d.qpos[a+3:a+7] = quat
    if draw.container is not None: setfj("container_joint", draw.container.pos, draw.container.quat)
    for n, o in draw.objects.items(): setfj(f"{n}_joint", o.pos, o.quat)
    mujoco.mj_forward(m, d); ev = L.LadderEvaluator(m, scene, task, draw, data=d); ev.place_dirt(d)
    o = draw.objects[draw.target]; j = f"{draw.target}_joint"
    def at(x, y, z, q):
        setfj(j, [x, y, z], q); mujoco.mj_forward(m, d); return ev.evaluate(d.qpos.copy())
    ev.evaluate(d.qpos.copy()); at(o.pos[0], o.pos[1], o.pos[2] + 0.10, o.quat)
    spots = ev._dirt_world(d.qpos); c = spots[:, :2].mean(0)
    # sponge on its edge (rolled 90 deg about its long axis), 30-40 cm to the side, lowered
    # to table height and swept sideways: its normal now points at the dirt.
    edge = np.zeros(4); mujoco.mju_axisAngle2Quat(edge, np.array([1.0, 0, 0]), math.pi / 2)
    q = _quat_mul(np.asarray(o.quat), edge)
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); u = R.reshape(3, 3)[:, 2]  # the pad's normal, now horizontal
    u = u / np.linalg.norm(u[:2]) if np.linalg.norm(u[:2]) > 1e-6 else u
    for dist in np.linspace(0.30, 0.40, 15):  # pad 30-40 cm away, normal aimed AT the dirt
        r = at(c[0] - dist * u[0], c[1] - dist * u[1], scene.table_z + ev.pad_half[1] + 0.002, q)
    print(f"[{name}] sponge on edge 30-40 cm from the dirt: cleaned {r['wipe_n_cleaned']}/{r['wipe_n_dirt']}")
    assert r["wipe_n_cleaned"] == 0, "edge-on sponge still wipes at a distance"
print("EDGE TEST OK")
