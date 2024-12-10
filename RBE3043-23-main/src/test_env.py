import math
from math import pi
from random import choice, uniform
from typing import Any, Dict, Tuple, Optional

import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class TargetObject:
    """
    TargetObject tracks the lifecycle of a target object (not a goal)
    """

    def __init__(
        self,
        id: str,
        name: str,
        shape: int,
        size: np.array,
        position: np.array,
        color: np.array,
        removed: bool = False,
    ):
        self.id = id
        self.name = name
        self.shape = shape
        self.size = size
        self.position = position
        self.color = color
        self.removed = removed


class Pick_And_Place(Task):
    def __init__(
        self,
        sim: PyBullet,
        observation_type: int,
        robot: Panda,
        objects_count: int = 3,
        img_size: Tuple[int, int] = (256, 256),
        blocker_bar: bool = True,
        sorting_count: int  = 1
    ):
        if observation_type not in [OBSERVATION_POSES, OBSERVATION_IMAGE]:
            raise Exception(
                f"Invalid output type {observation_type}. Must be one of "
                + f"{OBSERVATION_POSES} for values and {OBSERVATION_IMAGE} "
                + "four images."
            )

        super().__init__(sim)

        self.robot = robot
        self.observation_type = observation_type
        self.sorting_count = sorting_count
        self.score: float = 0.0

        self.objects_count: int = objects_count
        if observation_type == OBSERVATION_IMAGE:
            self.img_size = img_size
        self.object_opacity = 0.8

        self.sim.create_plane(z_offset=-0.4)

        # goal_positions will contain one of the three
        # preset strings (set below) w/ the resulting
        # starting position each goal is expected to be
        self.sorter_positions: Dict[str, np.array] = {}
        self.blocker_bar = blocker_bar
        self._init_sorting_areas()

        # Track each target object as part of our goal
        self.goal: Dict[int, TargetObject] = {}

        # Size multiplier is the range of sizes allowed for
        # target objects
        self.size_multiplier: Tuple[float, float] = (0.5, 1)

        self.task_init()

    def task_init(self):
        # Create our plane and table for the scenario
        self.sim.create_table(length=0.8, width=0.8, height=0.4, x_offset=-0.3)

        # These position_limits are where the objects are allowed
        # to spawn. This reads as (x, y), where each axis
        # in turn is a tuple of min/max placement.
        self.object_position_limits: Tuple[Tuple[float, float]] = (
            (-0.06, 0.06),
            (-0.2, 0.2),
        )
        # self.sim.create_sphere(
        #     body_name="marker",
        #     radius=0.025,
        #     mass=0.0,
        #     ghost=True,
        #     position=(0.06, 0.2, 0.01),
        #     rgba_color=(255, 255, 0, 1.0),
        # )

    def _init_sorting_areas(self):
        if self.sorting_count == 3:
            self.sorter_positions = {
                SORTING_ONE: np.array([-0.25, -0.2, 0.01]),
                SORTING_TWO: np.array([-0.25, 0.00, 0.01]),
                SORTING_THREE: np.array([-0.25, 0.2, 0.01]),
            }
        if self.sorting_count == 2:
            self.sorter_positions = {
                SORTING_ONE: np.array([-0.25, -0.2, 0.01]),
                SORTING_TWO: np.array([-0.25, 0.00, 0.01]),
            }
        if self.sorting_count == 1:
            self.sorter_positions = {
                SORTING_ONE: np.array([-0.25, -0.2, 0.01]),
            }
        if self.blocker_bar:
            self.sorter_positions["blocker"] = np.array([-0.2, 0.0, 0.01])
        count = self.sorting_count
        if (count == 3):
            self.sim.create_box(
                body_name=SORTING_THREE,
                half_extents=np.array([0.05, 0.1, 0.01]),
                mass=0.0,
                ghost=False,
                position=self.sorter_positions[SORTING_THREE],
                rgba_color=np.array([1.0, 0, 0, 0.4]),
            )
            count -=1
        if count == 2:
            self.sim.create_box(
                body_name=SORTING_TWO,
                half_extents=np.array([0.05, 0.1, 0.01]),
                mass=0.0,
                ghost=False,
                position=self.sorter_positions[SORTING_TWO],
                rgba_color=np.array([0.0, 1.0, 0, 0.4]),
            )
            count-=1
        if count == 1:
            self.sim.create_box(
                body_name=SORTING_ONE,
                half_extents=np.array([0.05, 0.1, 0.01]),
                mass=0.0,
                ghost=False,
                position=self.sorter_positions[SORTING_ONE],
                rgba_color=np.array([0, 0, 1.0, 0.5]),
            )

        if self.blocker_bar:
            # Create the blocking bar
            self.sim.create_box(
                body_name="blocker",
                half_extents=np.array([0.01, 0.3, 0.005]),
                mass=0.0,
                ghost=False,
                position=self.sorter_positions["blocker"],
                rgba_color=np.array([0.0, 0.0, 0.0, 0.8]),
            )

    def set_sorter_positions(self):
        """
        set_goal_positions will ensure that goals are placed
        in the appropriate place in the environment
        """
        # for count in range(self.sorting_count):
        #     self.sim.set_base_pose(
        #         GOALS[count],
        #         position=self.sorter_positions[GOALS[count]],
        #         orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        #     )
        for sorter in self.sorter_positions:
            self.sim.set_base_pose(
                sorter,
                position=self.sorter_positions[sorter],
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            )

    def setup_target_objects(self):
        """
        Generate self.objects_count objects randomly on the table of
        varying sizes, colors, and shapes. The shapes check to ensure
        that they do NOT collide with one another to start.
        """
        base_size = 0.025
        base_mass = 0.5
        base_box_volume = base_size**3
        base_cylinder_volume = pi * base_size**3  # h == r in base cylinder
        # First, delete each object to cleanup
        self.delete_all_objects()

        for object in range(0, self.objects_count):
            # Attempt to create the object. If it collides with
            # another, delete it and try again
            while True:
                name = f"object_{object}"
                color = self.get_random_color()
                if (self.sorting_count == 1):
                    shape = choice(SHAPES[0:1])
                if (self.sorting_count == 2):
                    shape = choice(SHAPES[0:2])
                if (self.sorting_count == 3):
                    shape = choice(SHAPES)
                position = self.get_random_object_position()

                if shape == CUBE:
                    x = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    y = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    z = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    size = np.array([x, y, z]).astype(np.float32)

                    volume = x * y * z
                    mass_multiplier = volume / base_box_volume

                    self.sim.create_box(
                        body_name=name,
                        half_extents=np.array([x, y, z]),
                        mass=base_mass * mass_multiplier,
                        ghost=False,
                        position=position,
                        rgba_color=color,
                    )
                elif shape == CYLINDER:
                    height = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    radius = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    size = np.array([height, radius, 0.0]).astype(np.float32)

                    volume = pi * radius**2 * height
                    mass_multiplier = volume / base_cylinder_volume

                    self.sim.create_cylinder(
                        body_name=name,
                        radius=radius,
                        height=height,
                        mass=base_mass * mass_multiplier,
                        ghost=False,
                        position=position,
                        rgba_color=color,
                    )
                elif shape == SPHERE:
                    multiplier = uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    size = np.array([multiplier, 0.0, 0.0]).astype(np.float32)

                    self.sim.create_sphere(
                        body_name=name,
                        radius=base_size * multiplier,
                        mass=base_mass * multiplier,
                        ghost=False,
                        position=position,
                        rgba_color=color,
                    )
                else:
                    raise Exception("Improper shape chosen")

                id = self.sim._bodies_idx[name]

                # Now ensure that the shape created does not
                # intersect any of the existing shapes
                collisions = False
                # If this is the first, we're good; move on
                if len(self.goal) <= 0:
                    break
                # ...otherwise we're going to compare it
                # against all known objects. If there's
                # overlap we delete this and move on
                for other in self.goal:
                    other_id = self.goal[other].id
                    if self.check_collision(id, other_id):
                        collisions = True
                        break

                if collisions:
                    self.sim.physics_client.removeBody(id)
                    continue
                else:
                    break

            self.goal[object] = TargetObject(id, name, shape, size, position, color)

    def check_collision(self, object1: str, object2: str) -> bool:
        """
        check_collision will check if the two objects overlap at all
        and returns a boolean to that effect
        """
        contacts = self.sim.physics_client.getContactPoints(object1, object2)
        return contacts is not None and len(contacts) > 0

    def delete_all_objects(self):
        for object in self.goal:
            self.sim.physics_client.removeBody(self.goal[object].id)
        self.goal = {}

    def get_random_color(self) -> np.array:
        """
        Returns an appropriate color from a list of decent color choices
        in the form of a 4 dimensional RGBA array (colors are (0,255) ->
        (0, 1) scaled)
        """
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (178, 102, 255),
            (102, 255, 255),
            (102, 0, 204),
            (255, 128, 0),
            (204, 0, 102),
        ]
        color = choice(colors)

        return np.array([color[0], color[1], color[2], self.object_opacity])

    def get_random_object_position(self) -> np.array:
        """
        get_random_object_position returns a random np.array of an object's
        permissions within the permissive bounds set at instantiation.
        """
        x = uniform(
            self.object_position_limits[0][0], self.object_position_limits[0][1]
        )
        y = uniform(
            self.object_position_limits[1][0], self.object_position_limits[1][1]
        )
        z = 0.01
        return np.array([x, y, z])

    def reset(self):
        # Ensure each goal hasn't moved
        self.set_sorter_positions()

        # Generate new objects
        self.setup_target_objects()

        # Clear our score
        self.score = 0.0

    def get_obs(self) -> Tuple[np.array, float]:
        """
        Determines if any objects collided, adjusts score and reward accordingly,
        and returns an observation along with the reward for this step.

        Returns:
            np.array: The observation at this step.
            float: The reward for this step.
        """
        reward = 0.0  # Initialize the reward

        # Handle floor collisions
        reward += self._handle_floor_collisions()

        # Handle goal collisions
        reward += self._handle_goal_collisions()

        # Reward for moving towards the closest object
        reward += self._reward_closer_to_object()

        # Reward for successful grasping
        reward += self._reward_grasping_success()

        # Reward for moving an object towards its goal
        reward += self._reward_object_towards_goal()

        # Penalize time-step to encourage efficiency
        reward += STEP_PENALTY

        # Ensure that each goal stays in position
        self.set_sorter_positions()

        # Return the observation and the reward
        if self.observation_type == OBSERVATION_IMAGE:
            observation = self._get_img()
        else:
            observation = self._get_poses_output()
        self.score += reward
        return observation, reward

    def _handle_floor_collisions(self) -> float:
        """Checks for floor collisions and penalizes accordingly."""
        reward = 0.0
        floor_id = self.sim._bodies_idx["plane"]
        for object_key in self.goal:
            if self.goal[object_key].removed:
                continue

            object_id = self.goal[object_key].id
            if self.check_collision(object_id, floor_id):
                reward += FLOOR_COLLISION_PENALTY
                self.sim.physics_client.removeBody(object_id)
                self.goal[object_key].removed = True
                print(f"Object {object_key} dropped to the floor")
        return reward

    def _handle_goal_collisions(self) -> float:
        """Checks for collisions between objects and goals, rewarding or penalizing as necessary."""
        reward = 0.0
        for object_key in self.goal:
            if self.goal[object_key].removed:
                continue

            for i in range(self.sorting_count):
                goal = GOALS[i]
                object = self.goal[object_key]
                object_id = object.id
                goal_id = self.sim._bodies_idx[goal]

                if self.check_collision(object_id, goal_id):
                    self.sim.physics_client.removeBody(object_id)
                    self.goal[object_key].removed = True

                    # Reward or penalize based on correct/incorrect sorting
                    if CORRECT_SORTS[goal] == object.shape:
                        reward += DROP_SUCCESS_REWARD
                        print(f"Object {object_key} correctly sorted into {goal}")
                    else:
                        reward += WRONG_DROP_PENALTY
                        print(f"Object {object_key} incorrectly sorted into {goal}")
        return reward

    def _reward_closer_to_object(self) -> float:
        """Rewards the agent for moving closer to the nearest object."""
        ee_position = self.robot.get_ee_position()
        closest_object, closest_distance = self._get_closest_object(ee_position)
        if closest_object:
            distance_to_object = np.linalg.norm(ee_position - self.get_object_pose(closest_object)[:3])
            distance_delta = abs(closest_distance - distance_to_object)
            return MOVE_TOWARD_OBJECT_REWARD * distance_delta
        return 0.0

    def _reward_grasping_success(self) -> float:
        """Rewards the agent for successfully grasping an object."""
        return 0.0
        closest_object, _ = self._get_closest_object(self.robot.get_ee_position())
        if closest_object and self._is_object_grasped(closest_object):
            print(f"Object {closest_object} successfully grasped")
            return GRASP_SUCCESS_REWARD

    def _reward_object_towards_goal(self) -> float:
        """Rewards the agent for moving objects closer to their goals."""
        closest_object, closest_distance = self._get_closest_object(self.robot.get_ee_position())
        if closest_object and not closest_object.removed:
            object_pos = self.get_object_pose(closest_object)[:3]
            goal_pos = self.sorter_positions[GOALS[closest_object.shape]]
            distance_to_goal = np.linalg.norm(object_pos - goal_pos)
            distance_delta = abs(closest_distance - distance_to_goal)
            return MOVE_OBJECT_TO_GOAL_REWARD * distance_delta

        return 0.0

    def _get_closest_object(self, ee_position: np.array) -> Tuple[Optional[TargetObject], float]:
        """ Return the closest object to the end-effector (EE). """
        closest_object = None
        closest_distance = float('inf')
        for object in self.goal.values():
            if object.removed:
                continue
            object_pos = self.get_object_pose(object)[:3]  # x, y, z position
            distance = np.linalg.norm(ee_position - object_pos)
            if distance < closest_distance:
                closest_object = object
                closest_distance = distance
        return closest_object, closest_distance

    def _is_object_grasped(self, object) -> bool:
        """ Check if the gripper has grasped an object. """
        return self.robot.get_fingers_width() < GRASP_THRESHOLD

    def get_object_pose(self, object: TargetObject) -> np.array:
        object_position = self.sim.get_base_position(object.name)
        object_rotation = self.sim.get_base_rotation(object.name)
        object_velocity = self.sim.get_base_velocity(object.name)
        object_angular_velocity = self.sim.get_base_angular_velocity(object.name)
        observation = np.concatenate(
            [object_position, object_rotation, object_velocity, object_angular_velocity]
        )
        return observation.astype(np.float32)



    def _get_poses_output(self) -> np.array:
        """
        _get_poses_output will return the poses of all objects in the scene,
        as well as their identity and size. It will be a series of values for
        the raw (x, y, z, theta, phi, psi) pose of the object, as well as an
        identity (type of shape), and size (0-1) for min/max size, and the pose
        of the robot's end effector, and a 0-1 value for its gripper open/close
        state. An example of the return would be:

        [[x, y, z, theta, phi, psi,
          xd, yd, zd, thetad, phid, psid, <~ velocities
        [identity], [size]] (times # of set objects), ...,
        (ee_x, ee_y, ee_z, ee_theta, ee_phi, ee_psi,
        ee_xd, ee_yd, ee_zd, ee_thetad, ee_phid, ee_psid), # <~ velocities
        gripper_status]

        Note that the identity is a one-hot encoded list of shape [CUBE, CYLINDER,
        SPHERE] and that size is a three value array of varying meaning based
        on size: [x, y, z] for CUBE, [radius, height] for CYLINDER, [radius]
        for SPHERE. Unused values are 0.0.
        """
        # The size of this vector is determined by the number of objects expected
        pose_values = 12
        shape_values = 3
        size_values = 3
        # End effector values
        ee_values = 12
        finger_values = 1
        # The length of our vector is (pose_values + (identity, size)) for each
        # object, pose_values for the end effector, and one additional value for
        # the gripper finger state (distance between fingers)
        size = (
            (len(self.goal) * (pose_values + shape_values + size_values))
            + ee_values
            + finger_values
        )
        observation: np.array = np.zeros((size,), dtype="float32")

        index = 0
        for object in self.goal.values():
            # If the object has been removed, just report 0's for its existence
            if object.removed:
                index += 1
                continue

            pose = self.get_object_pose(object)
            object_index = index * (pose_values + shape_values + size_values)
            observation[object_index : object_index + pose_values] = pose

            # The shape is a one hot encoded vector of [CUBE, CYLINDER, SPHERE]
            if object.shape == CUBE:
                shape_type = [1, 0, 0]
            elif object.shape == CYLINDER:
                shape_type = [0, 1, 0]
            elif object.shape == SPHERE:
                shape_type = [0, 0, 1]
            shapes_index = object_index + pose_values
            observation[shapes_index : shapes_index + shape_values] = shape_type
            size_index = shapes_index + shape_values
            observation[size_index : size_index + size_values] = object.size
            index += 1

        # Get the end effector position
        ee_position = self.robot.get_ee_position()
        ee_rotation_quaternion = self.sim.get_link_orientation(
            self.robot.body_name, self.robot.ee_link
        )
        ee_rotation = self._quaternion_to_euler(ee_rotation_quaternion)
        # print("rot", ee_rotation)
        # print("rot other", self.sim.get_base_rotation(self.robot.ee_link))
        ee_velocity = self.robot.get_ee_velocity()
        ee_rotational_velocity = self.sim.get_link_angular_velocity(
            self.robot.body_name, self.robot.ee_link
        )

        # ee_angulary_velocity = 0.0
        fingers_width = self.robot.get_fingers_width()
        ee_index = (pose_values + shape_values + size_values) * len(self.goal)
        observation[ee_index : ee_index + 3] = ee_position
        observation[ee_index + 3 : ee_index + 6] = ee_rotation
        observation[ee_index + 6 : ee_index + 9] = ee_velocity
        observation[ee_index + 9 : ee_index + 12] = ee_rotational_velocity
        observation[ee_index + 12] = fingers_width

        return observation

    def _quaternion_to_euler(self, quaternion: np.array):
        """
        _quaternion_to_euler will convert a quaternion to euler angless
        """
        x, y, z, w = quaternion
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        X = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)

        return np.array([X, Y, Z]).astype(np.float32)

    def _get_img(self) -> np.array:
        """
        _get_img will return the image from the camera in human rendering
        mode
        """
        # We have to swap render mode if it's set to human mode
        # to get it to draw for us.
        original_render_mode = self.sim.render_mode
        self.sim.render_mode = "rgb_array"
        img = self.sim.render(
            self.img_size[0],
            self.img_size[1],
            # target_position=self.camera_position,
            target_position=None,
            distance=0.0,
            yaw=45,
            pitch=-30,
            roll=0.0,
        )
        self.sim.render_mode = original_render_mode
        return img

    def get_achieved_goal(self) -> np.ndarray:
        return np.array(
            all(target.removed for target in self.goal.values()), dtype="bool"
        )

    def is_terminated(self) -> bool:
        """
        is_terminated returns whether or not the episode is
        in a terminal state; this can be due to:
        1. All objects have been removed somehow from the env
        2. The timer has hit 0

        It is not an indication of success
        """

        return all(obj.removed for obj in self.goal.values())

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = ...,
    ) -> np.ndarray:
        """
        is_success is a misnamed function, required as a layover
        from using the panda_gym library. Instead it is best
        to read it as an interface w/ is_terminated, and in no
        way reads whether it was a success, since the episode can
        end via timeout without doing the goals.
        """
        return np.array([self.is_terminated()], dtype="bool")

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = ...,
    ) -> np.ndarray:
        return np.array([self.score], dtype="float32")


class My_Arm_RobotEnv(RobotTaskEnv):
    """Sorter task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "human".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
    """

    def __init__(
        self,
        observation_type: int,
        objects_count: int = 5,
        blocker_bar: bool = True,
        render_mode: str = "human",
        control_type: str = "ee",
        renderer: str = "OpenGL",
        render_width: int = 720,
        render_height: int = 480,
        sorting_count: int = 1
    ) -> None:
        if observation_type not in [OBSERVATION_IMAGE, OBSERVATION_POSES]:
            raise ValueError("observation_type must be one of either images or poses")

        sim = PyBullet(
            render_mode=render_mode,
            background_color=np.array((200, 200, 200)),
            renderer=renderer,
        )
        robot = Panda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )
        task = Pick_And_Place(sim,
                              observation_type,
                              robot,
                              objects_count=objects_count,
                              blocker_bar=blocker_bar,
                              sorting_count=sorting_count)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=None,
            render_distance=0.9,
            render_yaw=45,
            render_pitch=-30,
            render_roll=0.0,
        )
        self.total_score = 0
        self.sim.place_visualizer(
            target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
        )

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        self.total_score = 0
        return observation, None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        observation, _ = self.task.get_obs()
        observation = observation.astype(np.float32)
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
        }

    def get_obs(self) -> np.ndarray:
        observation, _ = self.task.get_obs()
        return observation.astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        score_prior = self.task.score

        if isinstance(action, dict):
            discrete_action = action["discrete"]
            continuous_action = action["continuous"]
            self.robot.set_action(continuous_action)

        self.sim.step()
        observation = self._get_obs()
        score_after = self.task.score

        # An episode is terminated iff the agent has reached the target
        terminated = bool(
            self.task.is_success(observation["achieved_goal"], self.task.get_goal())
        )
        truncated = False
        info = {"is_success": terminated}
        step_penalty = STEP_PENALTY
        reward = (score_after - score_prior) + step_penalty
        self.total_score += reward
        # print("Score: ",self.total_score)
        # print("reward: ", reward)
        return observation, reward, terminated, truncated, info


SORTING_ONE = "sorting_one"
SORTING_TWO = "sorting_two"
SORTING_THREE = "sorting_three"
GOALS = [SORTING_ONE, SORTING_TWO, SORTING_THREE]

CUBE = 0
CYLINDER = 1
SPHERE = 2
SHAPES = [CUBE, CYLINDER, SPHERE]

# This is the expected correct sorting results
CORRECT_SORTS = {
    SORTING_ONE: CYLINDER,
    SORTING_TWO: SPHERE,
    SORTING_THREE: CUBE,
}


# FLOOR_PENALTY = -50
# # WRONG_SORT_REWARD = 25
# # SORT_REWARD = 100
# WRONG_SORT_REWARD = 200
# SORT_REWARD = 500

MOVE_TOWARD_OBJECT_REWARD = -1.0     # Reward for moving EE toward the object
GRASP_SUCCESS_REWARD = 50.0        # Reward for successful grasp
MOVE_OBJECT_TO_GOAL_REWARD = -1.0   # Reward for moving object toward goal
DROP_SUCCESS_REWARD = 50.0       # Reward for successfully placing in correct goal
WRONG_DROP_PENALTY = -20.0        # Penalty for placing object in wrong goal
FLOOR_COLLISION_PENALTY = -50.0   # Penalty for dropping the object on the floor
STEP_PENALTY = -0.1               # Small penalty to encourage efficiency
GRASP_THRESHOLD = 0.02

OBSERVATION_POSES: int = 0
OBSERVATION_IMAGE: int = 1

from utils import add_world_frame
import time
def test_env():

    env = My_Arm_RobotEnv(observation_type=0,
                          render_mode="human",
                          blocker_bar=False,
                          objects_count=1,
                          sorting_count=2
                          )
    add_world_frame()
    observation, info = env.reset()

    for _ in range(10000):
        time.sleep(1/24)
        action = env.action_space.sample()
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Run 1 episode")
            observation, info = env.reset()


import time


def test_fixed_actions():
    env = My_Arm_RobotEnv(
        observation_type=0,
        render_mode="human",
        blocker_bar=False,
        objects_count=1,
        sorting_count=2
    )

    observation, info = env.reset()

    # List of fixed actions to cycle through
    fixed_actions = [
        [0, 0, 0.1, 0],  # Move up
        [0, 0, -0.1, 0],  # Move down
        [-0.1, 0, 0, 0],  # Move left
        [0.1, 0, 0, 0],  # Move right
        [0, 0.1, 0, 0],  # Move forward
        [0, -0.1, 0, 0],  # Move backward
        [0, 0, 0, 0.3],  # Open gripper
        [0, 0, 0, -0.2],  # Close gripper
        [0, 0, 0.1, 0.7],  # Move up + open gripper
        [0, 0, -0.1, -0.5]  # Move down + close gripper
    ]

    for action in fixed_actions:
        for _ in range(50):  # Each action lasts for 50 time steps
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
            time.sleep(1 / 24)  # Delay for rendering

        if terminated or truncated:
            print("Episode ended, resetting environment.")
            observation, info = env.reset()


if __name__ == '__main__':
    test_fixed_actions()
