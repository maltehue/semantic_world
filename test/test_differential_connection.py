import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    PrismaticConnection,
    DifferentialConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import Derivatives


def make_simple_world():
    world = World()
    parent = Body(name=PrefixedName("parent", prefix="test"))
    child_a = Body(name=PrefixedName("child_a", prefix="test"))
    child_b = Body(name=PrefixedName("child_b", prefix="test"))

    with world.modify_world():
        world.add_kinematic_structure_entity(parent)
        world.add_kinematic_structure_entity(child_a)
        world.add_kinematic_structure_entity(child_b)

        # A regular prismatic joint (reference joint)
        ref_conn = PrismaticConnection.create_with_dofs(
            world=world,
            parent=parent,
            child=child_a,
            axis=Vector3.X(reference_frame=parent),
            name=PrefixedName("ref", prefix="test"),
        )
        world.add_connection(ref_conn)

        # A differential joint whose velocity depends on state and time
        def law(world_obj: World, t: float, self_conn: DifferentialConnection1DOF) -> float:
            # Default: zero; overridden by tests
            return 0.0

        # Create the differential connection once using the helper to ensure
        # its DOF is properly created and registered in the world.
        diff_conn = DifferentialConnection1DOF.create_with_dofs(
            world=world,
            parent=parent,
            child=child_b,
            axis=Vector3.X(reference_frame=parent),
            name=PrefixedName("diff", prefix="test"),
        )
        # Assign the rate function
        diff_conn.rate_function = law
        world.add_connection(diff_conn)

    return world, ref_conn, diff_conn


def test_coupled_velocity_depends_on_other_joint_position():
    world, ref_conn, diff_conn = make_simple_world()

    # Define velocity law: v_diff = position of ref joint
    def v_law(world_obj: World, t: float, self_conn: DifferentialConnection1DOF) -> float:
        return ref_conn.position

    diff_conn.rate_function = v_law

    # Set reference joint position directly
    ref_conn.position = 2.0

    # Apply zero commands; update should integrate diff by v*dt
    dt = 0.1
    num_free = len(world.state)
    commands = np.zeros(num_free)
    world.apply_control_commands(commands, dt, Derivatives.velocity)

    assert np.isclose(diff_conn.position, 0.2)


def test_time_dependent_velocity_integrates_over_time():
    world, ref_conn, diff_conn = make_simple_world()

    # v = t, integrate over 10 steps with dt=0.1
    def v_law(world_obj: World, t: float, self_conn: DifferentialConnection1DOF) -> float:
        return t

    diff_conn.rate_function = v_law

    dt = 0.1
    steps = 10
    num_free = len(world.state)
    commands = np.zeros(num_free)

    for _ in range(steps):
        world.apply_control_commands(commands, dt, Derivatives.velocity)

    # Position should be sum_{k=1..steps} (k*dt)*dt = dt^2 * steps*(steps+1)/2
    expected = (dt ** 2) * steps * (steps + 1) / 2.0
    assert np.isclose(diff_conn.position, expected)
