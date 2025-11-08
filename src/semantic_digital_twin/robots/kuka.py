from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Self

from .robot_mixins import HasNeck, SpecifiesLeftRightArm, HasArms
from ..datastructures.prefixed_name import PrefixedName
from ..robots.abstract_robot import (
    Neck,
    Finger,
    ParallelGripper,
    Arm,
    Camera,
    FieldOfView,
    Torso,
    AbstractRobot,
)
from ..spatial_types import Quaternion, Vector3
from ..world import World


@dataclass
class Kuka(AbstractRobot, HasArms):
    """
    Represents the Personal Robot 2 (PR2), which was originally created by Willow Garage.
    The PR2 robot consists of two arms, each with a parallel gripper, a head with a camera, and a prismatic torso
    """

    def load_srdf(self):
        pass

    def __hash__(self):
        return hash(
            tuple(
                [self.__class__]
                + sorted([kse.name for kse in self.kinematic_structure_entities])
            )
        )

    @classmethod
    def from_world(cls, world: World) -> Self:
        """
        Creates a Kuka robot view from the given world.

        :param world: The world from which to create the robot view.

        :return: A Kuka robot view.
        """

        robot = cls(
            name=PrefixedName(name="kuka", prefix=world.name),
            root=world.get_body_by_name("world"),
            _world=world,
        )

        #arm = Arm(
        #    name=PrefixedName("arm", prefix=robot.name.name),
        #    root=world.get_body_by_name("base_link"),
        #    tip=world.get_body_by_name("link_6"),
        #    _world=world,
        #)

        #robot.add_arm(arm)
        world.add_semantic_annotation(robot, exists_ok=True)

        return robot
