"""Compile open warehouse meshes and planar-body inertia through Newton's real solver."""

import unittest

import mujoco
import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo


class NewtonStaticMeshTest(unittest.TestCase):
    def assert_compiled_mesh_vertices(self, solver, vertices, indices):
        mesh_geom = np.flatnonzero(solver.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)[0]
        mesh_id = solver.mj_model.geom_dataid[mesh_geom]
        start = solver.mj_model.mesh_vertadr[mesh_id]
        count = solver.mj_model.mesh_vertnum[mesh_id]
        compiled_vertices = solver.mj_model.mesh_vert[start:start + count]
        rotation = solver.mj_data.geom_xmat[mesh_geom].reshape(3, 3)
        world_vertices = compiled_vertices @ rotation.T + solver.mj_data.geom_xpos[mesh_geom]
        referenced = vertices[np.unique(indices)]
        distances = np.linalg.norm(referenced[:, None] - world_vertices[None], axis=-1)
        # MuJoCo may reorder/deduplicate vertices and drop unreferenced vertices.
        self.assertLess(float(distances.min(axis=1).max()), 2e-5)

    def test_nonwatertight_world_mesh_keeps_collision_geometry_and_body_inertia(self):
        # POLYFACE_4085 from Galileo: a single quad and an unused origin vertex.
        vertices = np.array([
            [37.071392, 123.194023, 9.15],
            [37.098457, 123.209648, 9.15],
            [37.098457, 123.209648, 10.983477],
            [37.071392, 123.194023, 10.983477],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        legacy = mujoco.MjSpec()
        legacy.add_mesh(name="wall", uservert=vertices.flatten(), userface=indices,
                        inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_LEGACY)
        legacy.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="wall")
        with self.assertRaisesRegex(ValueError, "mesh (volume|inertia)"):
            legacy.compile()
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.begin_world()
        builder.add_shape_mesh(-1, mesh=mesh, label="nonwatertight_wall")
        body = builder.add_link(xform=wp.transform(wp.vec3(0, 0, 10), wp.quat_identity()), mass=1.0)
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        builder.add_shape_sphere(body, radius=0.1)
        builder.end_world()
        model = builder.finalize(device="cpu")
        mass = model.body_mass.numpy().copy()
        inertia = model.body_inertia.numpy().copy()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False,
                              nconmax=40, njmax=60)
        self.assertEqual(solver.mj_model.ngeom, 2)
        self.assert_compiled_mesh_vertices(solver, vertices, indices)
        np.testing.assert_array_equal(model.shape_source[0].vertices, vertices)
        np.testing.assert_array_equal(model.shape_source[0].indices.flatten(), indices)
        np.testing.assert_array_equal(model.body_mass.numpy(), mass)
        np.testing.assert_array_equal(model.body_inertia.numpy(), inertia)
        self.assertAlmostEqual(float(solver.mj_model.body_mass[1]), float(mass[body]))
        np.testing.assert_allclose(solver.mj_model.body_inertia[1], np.diag(inertia[body]), rtol=1e-6)

        # A dynamic background body also contains open CoACD fragments. Its authored
        # mass/inertia must survive changing the mesh's auxiliary inertia frame.
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.begin_world()
        authored_inertia = np.diag([0.2, 0.3, 0.4]).astype(np.float32)
        body = builder.add_link(mass=2.0, inertia=wp.mat33(authored_inertia), lock_inertia=True)
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        builder.add_shape_mesh(body, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        builder.end_world()
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False,
                              nconmax=40, njmax=60)
        self.assertEqual(float(solver.mj_model.body_mass[1]), 2.0)
        self.assert_compiled_mesh_vertices(solver, vertices, indices)
        principal_rotation = np.empty(9)
        mujoco.mju_quat2Mat(principal_rotation, solver.mj_model.body_iquat[1])
        rotation = principal_rotation.reshape(3, 3)
        compiled_inertia = rotation @ np.diag(solver.mj_model.body_inertia[1]) @ rotation.T
        np.testing.assert_allclose(compiled_inertia, authored_inertia, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(model.shape_source[0].vertices, vertices)
        np.testing.assert_array_equal(model.shape_source[0].indices.flatten(), indices)
        with self.assertRaisesRegex(ValueError, "mesh (volume|inertia)"):
            SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True,
                         nconmax=40, njmax=60)


    def test_planar_table_float32_inertia_rounding(self):
        # Actual warehouse table tensor: A+B-C=-7.68e-8 after float64 eigendecomposition.
        inertia = np.array([
            [12.814079284667969, -6.01907231612131e-06, -1.9585551854106598e-05],
            [-6.01907231612131e-06, 32.29090118408203, -0.0007015690789557993],
            [-1.9585551854106598e-05, -0.0007015690789557993, 45.10498046875],
        ], dtype=np.float32)
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(
            mass=110.22291,
            fullinertia=[inertia[0, 0], inertia[1, 1], inertia[2, 2],
                         inertia[0, 1], inertia[0, 2], inertia[1, 2]],
            explicitinertial=True,
        )
        body.add_freejoint()
        with self.assertRaisesRegex(ValueError, "inertia must satisfy"):
            spec.compile()
        builder = newton.ModelBuilder()
        builder.add_body(mass=110.22291, inertia=wp.mat33(inertia), lock_inertia=True)
        model = builder.finalize(device="cpu")
        original = model.body_inertia.numpy().copy()
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False)
        np.testing.assert_array_equal(model.body_inertia.numpy(), original)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, solver.mj_model.body_iquat[1])
        rotation = rotation.reshape(3, 3)
        compiled = rotation @ np.diag(solver.mj_model.body_inertia[1]) @ rotation.T
        # Newton stores the resulting principal-frame quaternion in float32.
        self.assertLess(np.linalg.norm(compiled - inertia) / np.linalg.norm(inertia), 1e-6)
        invalid = original.copy()
        invalid[0] = np.diag([1.0, 1.0, 4.0])
        model.body_inertia.assign(invalid)
        with self.assertRaisesRegex(ValueError, "inertia must satisfy"):
            SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False)


    def test_external_contacts_skip_only_initial_cpu_collision_detection(self):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        for x in (0.0, 0.1):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=False)
        # Export initialization does not need MuJoCo-generated contacts.
        self.assertEqual(solver.mj_data.ncon, 0)
        self.assertFalse(solver.mj_model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mujoco.mj_forward(solver.mj_model, solver.mj_data)
        self.assertGreater(solver.mj_data.ncon, 0)
        # Native MuJoCo contact mode retains its initial contacts.
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, use_mujoco_contacts=True)
        self.assertGreater(solver.mj_data.ncon, 0)


if __name__ == "__main__":
    unittest.main()
