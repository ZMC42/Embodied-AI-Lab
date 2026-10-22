"""Fix Newton 1.5.2 MuJoCo export and external-contact initialization."""

from importlib.metadata import distribution


package = distribution("newton")
assert package.version == "1.5.2", "Revalidate the MuJoCo export fixes when upgrading Newton"
path = package.locate_file("newton/_src/solvers/mujoco/solver_mujoco.py")
original = """                    spec.add_mesh(
                        name=name,
                        uservert=vertices.flatten(),
                        userface=indices.flatten(),
                        maxhullvert=maxhullvert,
                    )"""
replacement = """                    spec.add_mesh(
                        name=name,
                        uservert=vertices.flatten(),
                        userface=indices.flatten(),
                        maxhullvert=maxhullvert,
                        # Newton supplies contacts and body inertias; open/thin meshes
                        # only need an auxiliary MuJoCo surface-inertia frame.
                        inertia=(
                            mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                            if not self._use_mujoco_contacts
                            else mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY
                        ),
                    )"""
inertia_original = """            inertia = body_inertia[child]
            mass = body_mass[child]"""
inertia_replacement = """            inertia = body_inertia[child].astype(np.float64)
            # A planar body's float32 tensor can violate MuJoCo's strict triangle
            # check by rounding. Correct only that rounding, not invalid inertias.
            principal = np.linalg.eigvalsh(inertia)
            violation = principal[2] - principal[0] - principal[1]
            if 0.0 < violation <= np.finfo(np.float32).eps * np.trace(inertia):
                inertia += np.eye(3) * (violation + 8 * np.finfo(np.float64).eps * np.trace(inertia))
            mass = body_mass[child]"""
forward_original = """        mujoco.mj_forward(self.mj_model, self.mj_data)"""
forward_replacement = """        # Newton generates contacts externally. The initial CPU forward only
        # initializes dynamics; temporary MuJoCo contacts can overflow its stack.
        initial_disableflags = self.mj_model.opt.disableflags
        if not self._use_mujoco_contacts:
            self.mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        try:
            mujoco.mj_forward(self.mj_model, self.mj_data)
        finally:
            self.mj_model.opt.disableflags = initial_disableflags"""
source = path.read_text()
for original, replacement in ((original, replacement), (inertia_original, inertia_replacement),
                              (forward_original, forward_replacement)):
    if replacement in source:
        continue
    assert source.count(original) == 1, f"Unexpected Newton solver source: {path}"
    source = source.replace(original, replacement)
if source != path.read_text():
    # uv may hard-link installed files to its wheel cache and other environments.
    updated = path.with_suffix(".py.updated")
    updated.write_text(source)
    updated.replace(path)
print(f"Newton MuJoCo export compatibility fix: {path}")
