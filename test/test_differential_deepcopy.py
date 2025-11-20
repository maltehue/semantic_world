import numpy as np

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    DifferentialConnection1DOF,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import Derivatives


def test_world_deepcopy_with_differential_connection_serializes_and_restores():
    # Build a small world with a prismatic reference and a differential connection
    w = World()
    parent = Body(name=PrefixedName("parent", prefix="test"))
    a = Body(name=PrefixedName("a", prefix="test"))
    b = Body(name=PrefixedName("b", prefix="test"))

    with w.modify_world():
        w.add_kinematic_structure_entity(parent)
        w.add_kinematic_structure_entity(a)
        w.add_kinematic_structure_entity(b)

        ref = PrismaticConnection.create_with_dofs(
            world=w,
            parent=parent,
            child=a,
            axis=Vector3.X(reference_frame=parent),
            name=PrefixedName("ref", prefix="test"),
        )
        w.add_connection(ref)

        diff = DifferentialConnection1DOF.create_with_dofs(
            world=w,
            parent=parent,
            child=b,
            axis=Vector3.X(reference_frame=parent),
            name=PrefixedName("diff", prefix="test"),
        )

        # Provide some runtime-only law; it should not be serialized
        def law(world_obj: World, t: float, self_conn: DifferentialConnection1DOF) -> float:
            return ref.position

        diff.rate_function = law
        w.add_connection(diff)

    # Set some state and step once
    ref.position = 1.0
    dt = 0.1
    commands = np.zeros(len(w.state))
    w.apply_control_commands(commands, dt, Derivatives.velocity)

    # Deepcopy should not raise and should preserve structure and state
    w2 = w.__deepcopy__({})

    # Find corresponding connections by name
    diff2 = next(c for c in w2.connections if isinstance(c, DifferentialConnection1DOF))

    # The runtime callback is not serialized
    assert diff2.rate_function is None

    # Positions should match after deepcopy
    assert np.isclose(diff2.position, diff.position)

    # Symbols are present and connected to the child transform expression
    assert diff2.dof.symbols.position is not None
    assert diff2.connection_T_child_expression is not None
