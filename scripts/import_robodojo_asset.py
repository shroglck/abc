"""Import a RoboDojo object (USDZ) into abc/assets/robodojo/<category>_<id>/ as ladder `msobj`.

RoboDojo's asset dataset is Apache-2.0 on HuggingFace (the benchmark CODE is under a
non-commercial licence; the assets are not), z-up, metres, one visual mesh + one texture
per object, with `metadata.json` carrying mass/friction and an aligned bbox.

We write the same layout the MolmoSpaces importer consumes, so `ObjectSpec(kind="msobj",
asset=...)` works unchanged:

    <dir>/<name>.xml              MJCF fragment: <asset> meshes/texture + <worldbody><body>
    <dir>/<name>_visual.obj       triangulated visual mesh (+ _visual_0.png if textured)
    <dir>/<name>_collider<N>.obj  convex parts (trimesh/VHACD when available, else the hull)
    <dir>/metadata.json           RoboDojo's own metadata, verbatim

The ladder rotates y-up assets to z-up; RoboDojo is already z-up, so the MJCF sets
`mjt_zup="1"` in a comment and the ladder's MS_ZUP_ASSETS set lists them.

    .venv/bin/python scripts/import_robodojo_asset.py Geometry/basket/00002 --name basket
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HF = "https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/main/Assets/Object/RoboDojo"
OUT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "robodojo"


def fetch(rel: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        subprocess.run(["curl", "-sfL", "-o", str(dest), f"{HF}/{rel}"], check=True)
    return dest


def usd_to_mesh(usdz: Path):
    """World-space vertices, faces and (u,v) of every UsdGeom.Mesh, concatenated."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usdz))
    verts, faces, uvs, off = [], [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if len(pts) == 0 or len(counts) == 0:
            continue
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        pts = (np.c_[pts, np.ones(len(pts))] @ xf.T)[:, :3]
        # Fan-triangulate each polygon.
        tri, c = [], 0
        for n in counts:
            poly = idx[c : c + n]
            tri += [[poly[0], poly[i], poly[i + 1]] for i in range(1, n - 1)]
            c += n
        st = None
        for name in ("st", "primvars:st", "UVMap"):
            attr = prim.GetAttribute(name)
            if attr and attr.Get() is not None:
                st = np.asarray(attr.Get(), dtype=np.float64)
                break
        verts.append(pts)
        faces.append(np.asarray(tri, dtype=np.int64) + off)
        uvs.append(st if st is not None and len(st) == len(pts) else np.zeros((len(pts), 2)))
        off += len(pts)
    return np.vstack(verts), np.vstack(faces), np.vstack(uvs)


def write_obj(path: Path, v: np.ndarray, f: np.ndarray, uv: np.ndarray | None = None, mtl: str | None = None) -> None:
    with open(path, "w") as fh:
        if mtl:
            fh.write(f"mtllib {mtl}\nusemtl mat0\n")
        fh.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in v)
        if uv is not None:
            fh.writelines(f"vt {u:.6f} {w:.6f}\n" for u, w in uv)
            fh.writelines(f"f {a+1}/{a+1} {b+1}/{b+1} {c+1}/{c+1}\n" for a, b, c in f)
        else:
            fh.writelines(f"f {a+1} {b+1} {c+1}\n" for a, b, c in f)


def decompose(v: np.ndarray, f: np.ndarray, max_parts: int) -> list[np.ndarray]:
    """Convex parts for collision (CoACD). A single hull would fill every basket/dustbin,
    so a hollow receptacle is only usable if the decomposition succeeds."""
    import trimesh

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=True)
    try:
        import coacd

        coacd.set_log_level("error")
        parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), max_convex_hull=max_parts,
                                threshold=0.05, preprocess_mode="auto")
        if parts:
            return [np.asarray(vs) for vs, _ in parts[:max_parts]]
    except Exception as exc:
        print(f"[import] CoACD failed ({type(exc).__name__}: {exc}); falling back to the hull")
    return [np.asarray(mesh.convex_hull.vertices)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset", help="RoboDojo path, e.g. Geometry/basket/00002")
    ap.add_argument("--name", required=True, help="local asset dir name, e.g. basket")
    ap.add_argument("--max-parts", type=int, default=16)
    ap.add_argument("--decimate", type=int, default=20000, help="target visual vertex budget")
    args = ap.parse_args()

    out = OUT_ROOT / args.name
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        usdz = fetch(f"{args.asset}/object.usdz", Path(tmp) / "object.usdz")
        meta = json.loads(fetch(f"{args.asset}/metadata.json", out / "metadata.json").read_text())
        v, f, uv = usd_to_mesh(usdz)
        print(f"[import] {args.asset}: {len(v)} verts, {len(f)} faces, bbox {np.round(v.max(0)-v.min(0),3)}")

        import trimesh

        vis = trimesh.Trimesh(vertices=v, faces=f, process=False)
        if args.decimate and len(v) > args.decimate:
            vis = vis.simplify_quadric_decimation(face_count=args.decimate)
            print(f"[import]   decimated visual to {len(vis.vertices)} verts")
            uv = None
        write_obj(out / f"{args.name}_visual.obj", np.asarray(vis.vertices), np.asarray(vis.faces),
                  uv if uv is not None and len(uv) == len(vis.vertices) else None)
        for i, hull in enumerate(decompose(v, f, args.max_parts)):
            h = trimesh.Trimesh(vertices=hull).convex_hull
            write_obj(out / f"{args.name}_collider{i}.obj", np.asarray(h.vertices), np.asarray(h.faces))
        n_col = len(list(out.glob(f"{args.name}_collider*.obj")))

    phys = meta.get("physics", {})
    xml = [f'<mujoco model="{args.name}">', "  <asset>",
           f'    <mesh name="{args.name}_visual_mesh" content_type="model/obj" file="{args.name}_visual.obj"/>']
    xml += [f'    <mesh name="{args.name}_collider{i}" content_type="model/obj" file="{args.name}_collider{i}.obj"/>'
            for i in range(n_col)]
    xml += ["  </asset>", "  <worldbody>", f'    <body name="{args.name}">',
            f'      <geom name="{args.name}_visual" type="mesh" mesh="{args.name}_visual_mesh" contype="0" conaffinity="0" group="2" density="0"/>']
    xml += [f'      <geom name="{args.name}_collider{i}" type="mesh" mesh="{args.name}_collider{i}" group="3" rgba="0 0 0 0"/>'
            for i in range(n_col)]
    xml += ["    </body>", "  </worldbody>", "</mujoco>"]
    (out / f"{args.name}.xml").write_text("\n".join(xml) + "\n")
    (out / "SOURCE.txt").write_text(
        f"RoboDojo asset {args.asset} (HuggingFace datasets/RoboDojo-Benchmark/RoboDojo, Apache-2.0).\n"
        f"z-up, metres. mass={phys.get('mass')} friction={phys.get('friction')}.\n"
        f"Converted by scripts/import_robodojo_asset.py: USDZ -> visual OBJ + {n_col} convex collider OBJ(s).\n")
    print(f"[import] wrote {out} ({n_col} colliders)")


if __name__ == "__main__":
    main()
