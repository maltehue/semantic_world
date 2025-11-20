import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    DifferentialConnection1DOF,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import Derivatives


def make_pouring_world():
    """
    Creates a simple world with:
    - A box body that can rotate around Z (hinge/rotary DoF).
    - A differential connection that represents the fill level in the box.

    The differential connection's velocity (rate of change of fill level)
    will be defined by a test-provided rate_function.
    """
    world = World()
    world_root = Body(name=PrefixedName("root", prefix="world"))
    box = Body(name=PrefixedName("box", prefix="test"))
    fill = Body(name=PrefixedName("fill", prefix="test"))

    with world.modify_world():
        world.add_kinematic_structure_entity(world_root)
        world.add_kinematic_structure_entity(box)
        world.add_kinematic_structure_entity(fill)

        # Rotary joint for the box tilt (around Z)
        tilt = RevoluteConnection.create_with_dofs(
            world=world,
            parent=world_root,
            child=box,
            axis=Vector3.Z(reference_frame=world_root),
            name=PrefixedName("tilt", prefix="test"),
        )
        world.add_connection(tilt)

        # Differential connection representing fill level (prismatic-like along X, but used as scalar)
        diff = DifferentialConnection1DOF.create_with_dofs(
            world=world,
            parent=world_root,
            child=fill,
            axis=Vector3.X(reference_frame=world_root),
            name=PrefixedName("fill_level", prefix="test"),
        )
        world.add_connection(diff)

    return world, tilt, diff


def test_pouring_fill_level_decreases_when_tilted():
    """
    When the tilt angle exceeds a small threshold (0.1 rad), the fill level should
    decrease at a constant rate scaled by factor k.
    """
    world, tilt, fill_conn = make_pouring_world()

    # Parameters for the pouring law
    threshold = 0.1
    k = 0.5  # scaling factor for outflow rate

    # Define the rate function: negative (decreasing) when angle > threshold
    def rate_fn(w: World, t: float, self_conn: DifferentialConnection1DOF) -> float:
        angle = tilt.position
        return -k if angle > threshold else 0.0

    fill_conn.rate_function = rate_fn

    # Set initial fill level
    fill_conn.position = 1.0

    # Tilt the box above threshold
    tilt.position = 0.2

    # Simulate for some steps
    dt = 0.1
    steps = 20
    num_free = len(world.state)
    commands = np.zeros(num_free)
    for _ in range(steps):
        world.apply_control_commands(commands, dt, Derivatives.velocity)

    expected = 1.0 + (-k) * dt * steps
    assert np.isclose(fill_conn.position, expected)


def test_pouring_fill_level_constant_below_threshold():
    """
    If the tilt angle is below or equal to the threshold, the fill level should not change.
    """
    world, tilt, fill_conn = make_pouring_world()

    threshold = 0.1
    k = 0.5

    def rate_fn(w: World, t: float, self_conn: DifferentialConnection1DOF) -> float:
        angle = tilt.position
        return -k if angle > threshold else 0.0

    fill_conn.rate_function = rate_fn

    # Initialize fill level
    fill_conn.position = 1.0

    # Keep angle at threshold (no pouring)
    tilt.position = threshold

    dt = 0.1
    steps = 20
    num_free = len(world.state)
    commands = np.zeros(num_free)
    for _ in range(steps):
        world.apply_control_commands(commands, dt, Derivatives.velocity)

    assert np.isclose(fill_conn.position, 1.0)
