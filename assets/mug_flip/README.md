# mug_flip assets

`mug/textured.obj`, `mug/texture_map.png` — YCB object **025_mug**, copied from the
ManiSkill YCB asset pack (`mani_skill2_ycb/models/025_mug`).

* Source: the YCB Object and Model Set (Calli et al.), redistributed by ManiSkill.
* Geometry: centered on its own origin, +z out of the opening, half-height 0.0407 m,
  body radius 0.0467 m, handle reaching to 0.0526 m in +x. The scan is an open shell
  (no interior floor), so it is used for the VISUAL geom only; collision is a
  cylinder plus a capsule handle, built in `abc_minimal/mug_flip.py`.

The rest of the scene (arms, table, cameras, lights) is not duplicated here: the
mug task loads `assets/put_bottles/put_bottle.xml` and swaps the objects out.
