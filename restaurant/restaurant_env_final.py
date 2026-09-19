from __future__ import annotations

import heapq
import math
from pathlib import Path

import genesis as gs
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from order_system import OrderSystem


class RestaurantDeliveryEnv(gym.Env):
    """Final randomized restaurant environment for SAC+A*+Predictive-DWA control.

    Observation: 16 current LiDAR + 16 LiDAR change values + five navigation
    values. The temporal signal lets a policy distinguish an approaching
    obstacle from one moving away. The action remains [linear, angular].
    """

    metadata = {"render_modes": ["human"]}

    NUM_LIDAR_RAYS = 16
    LIDAR_MAX_DISTANCE = 5.0
    LIDAR_MIN_DISTANCE = 0.05

    WHEEL_RADIUS = 0.18
    TRACK_WIDTH = 0.92
    MAX_LINEAR_SPEED = 1.5
    MAX_ANGULAR_SPEED = 2.0
    ACTION_REPEAT = 4

    ROBOT_COLLISION_HALF_LENGTH = 0.60
    ROBOT_COLLISION_HALF_WIDTH = 0.52
    COLLISION_CONTACT_MARGIN = 0.02

    WORLD_X_MIN = -11.5
    WORLD_X_MAX = 11.5
    WORLD_Y_MIN = -8.5
    WORLD_Y_MAX = 8.5
    PATH_GRID_RESOLUTION = 0.15
    PATH_INFLATION = 0.90
    DYNAMIC_SAFETY_RADIUS = 1.05
    RECIPROCAL_TRIGGER_DISTANCE = 4.20
    RECIPROCAL_TRIGGER_TTC = 3.00
    RECIPROCAL_SIDE_OFFSET = 0.72
    RECIPROCAL_OFFSET_STEP = 0.030
    PERSON_YIELD_SPEED_SCALE = 0.28
    COOPERATIVE_MIN_ACTIVE_STEPS = 25

    TABLE_COUNT = 8
    # Includes the two chair collision boxes, not only the tabletop.
    TABLE_SIZE_UNROTATED = (1.65, 2.55)
    TABLE_MIN_CENTRE_DISTANCE = 4.75
    TABLE_SLOTS = (
        (-7.0, -5.2),
        (0.0, -5.2),
        (7.0, -5.2),
        (-7.0, 0.0),
        (0.0, 0.0),
        (7.0, 0.0),
        (-5.5, 5.2),
        (1.0, 5.2),
        (7.5, 5.2),
    )

    def __init__(
        self,
        render: bool = True,
        robot_urdf: str | Path = "restaurant_delivery_robot.urdf",
        table_urdf: str | Path = "restaurant_table_set.urdf",
        waiter_urdf: str | Path = "restaurant_waiter.urdf",
        customer_urdf: str | Path = "restaurant_customer.urdf",
        table_id: int | None = None,
        item: str | None = None,
        quantity: int | None = None,
        dynamic_people: bool = True,
        active_people_count: int = 4,
        people_speed_scale: float = 1.0,
        randomized_static_clutter: bool = True,
        embedded_gui: bool = False,
        human_cooperation_probability: float = 0.35,
    ) -> None:
        super().__init__()
        if table_id is not None and table_id not in range(1, self.TABLE_COUNT + 1):
            raise ValueError("table_id must be 1-8 or None")

        self.render_enabled = bool(render)
        self.robot_urdf = str(robot_urdf)
        self.table_urdf = str(table_urdf)
        self.waiter_urdf = str(waiter_urdf)
        self.customer_urdf = str(customer_urdf)
        self.requested_table_id = table_id
        self.requested_item = item
        self.requested_quantity = quantity
        self.dynamic_people_enabled = bool(dynamic_people)
        self.active_people_count = int(np.clip(active_people_count, 0, 4))
        self.people_speed_scale = float(np.clip(people_speed_scale, 0.35, 1.35))
        self.randomized_static_clutter = bool(randomized_static_clutter)
        self.embedded_gui = bool(embedded_gui and self.render_enabled)
        self.human_cooperation_probability = float(
            np.clip(human_cooperation_probability, 0.0, 1.0)
        )

        # Start beside the pickup/order counter so each order begins ready for
        # collection instead of requiring a full cross-restaurant pickup trip.
        self.start_position = np.array([-8.3, 5.50], dtype=np.float32)
        self.package_position = np.array([-9.0, 5.50], dtype=np.float32)
        self.selected_table_id = table_id or 1
        self.goal_position = np.array([-5.0, 0.0], dtype=np.float32)
        self.table_centres: dict[int, np.ndarray] = {}
        self.table_yaws: dict[int, float] = {}
        self.table_sizes: dict[int, tuple[float, float]] = {}
        self.previous_slot_assignment: tuple[int, ...] | None = None
        self.previous_person_forward: tuple[bool, bool, bool, bool] | None = None
        self.previous_lidar: np.ndarray | None = None

        self.pickup_distance = 0.80
        self.delivery_distance = 0.85
        self.max_steps = 2400
        # A delivery that is already on its final approach should not fail only
        # because the shared pickup + delivery budget expires a few steps early.
        self.delivery_grace_distance = 2.50
        self.delivery_grace_steps = 300

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        lidar_low = np.zeros(self.NUM_LIDAR_RAYS, dtype=np.float32)
        lidar_high = np.ones(self.NUM_LIDAR_RAYS, dtype=np.float32)
        lidar_delta_low = -np.ones(self.NUM_LIDAR_RAYS, dtype=np.float32)
        lidar_delta_high = np.ones(self.NUM_LIDAR_RAYS, dtype=np.float32)
        extra_low = np.array(
            [0.0, -np.pi, -self.MAX_LINEAR_SPEED, -self.MAX_ANGULAR_SPEED, 0.0],
            dtype=np.float32,
        )
        extra_high = np.array(
            [25.0, np.pi, self.MAX_LINEAR_SPEED, self.MAX_ANGULAR_SPEED, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.concatenate([lidar_low, lidar_delta_low, extra_low]),
            high=np.concatenate([lidar_high, lidar_delta_high, extra_high]),
            dtype=np.float32,
        )

        self.order_system = OrderSystem()
        self.current_order = None
        self.package_picked = False
        self.package_delivered = False
        self.just_picked_package = False
        self.collision = False
        self.collision_type = None
        self.out_of_bounds = False
        self.current_step = 0
        self.sim_time = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        # People keep their natural routes; predictive DWA plans around them.
        self.social_yield_active = False
        self.social_yield_events = 0

        self.static_obstacle_data: list[list[float]] = []
        self.table_obstacle_data: list[list[float]] = []
        self.dynamic_obstacle_data: list[list[float]] = []
        self.decor_obstacle_data: list[list[float]] = []
        self.obstacle_data: list[list[float]] = []
        self.table_entities = []
        self.table_label_entities = []
        self.person_entities = []
        self.person_states: list[dict] = []
        self.decor_entities = []
        self.decor_specs = (
            ((0.68, 0.68, 0.82), (0.28, 0.34, 0.20, 1.0)),
            ((0.90, 0.58, 0.92), (0.36, 0.22, 0.12, 1.0)),
            ((0.55, 0.46, 1.10), (0.15, 0.20, 0.25, 1.0)),
        )

        self._create_scene()

    @staticmethod
    def _surface(color):
        return gs.surfaces.Default(color=color)

    def _add_static_box(self, centre_xy, size, z, color):
        entity = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=size,
                pos=(float(centre_xy[0]), float(centre_xy[1]), float(z)),
                fixed=True,
            ),
            surface=self._surface(color),
        )
        self.static_obstacle_data.append(
            [
                float(centre_xy[0]),
                float(centre_xy[1]),
                float(size[0]),
                float(size[1]),
                float(size[2]),
            ]
        )
        return entity

    def _create_scene(self) -> None:
        self.scene = gs.Scene(
            show_viewer=self.render_enabled,
            sim_options=gs.options.SimOptions(dt=0.01),
            viewer_options=gs.options.ViewerOptions(
                # Closer, lower restaurant view without increasing the CPU
                # cost of the viewer.  The old high aerial camera made the
                # robot, people and furniture look miniature.
                res=(1100, 700),
                camera_pos=(18.0, -22.0, 16.0),
                camera_lookat=(0.0, 0.0, 0.75),
                camera_fov=42,
                refresh_rate=30,
                enable_gui=self.embedded_gui,
            ),
        )
        self.scene.add_entity(
            morph=gs.morphs.Plane(),
            surface=self._surface((0.80, 0.75, 0.66, 1.0)),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.robot_urdf,
                pos=(float(self.start_position[0]), float(self.start_position[1]), 0.0),
            )
        )

        wall_color = (0.25, 0.19, 0.16, 1.0)
        for centre, size in (
            ((0.0, -8.4), (23.0, 0.20, 1.20)),
            ((0.0, 8.4), (23.0, 0.20, 1.20)),
            ((-11.4, 0.0), (0.20, 17.0, 1.20)),
            ((11.4, 0.0), (0.20, 17.0, 1.20)),
        ):
            self._add_static_box(centre, size, size[2] / 2.0, wall_color)

        self._add_static_box(
            (-9.0, 7.0),
            (3.3, 0.85, 1.05),
            0.525,
            (0.34, 0.16, 0.08, 1.0),
        )

        for slot_index in range(self.TABLE_COUNT):
            x, y = self.TABLE_SLOTS[slot_index]
            entity = self.scene.add_entity(
                gs.morphs.URDF(
                    file=self.table_urdf,
                    pos=(x, y, 0.0),
                    fixed=True,
                    prioritize_urdf_material=True,
                )
            )
            self.table_entities.append(entity)

            marker_path = Path(self.table_urdf).resolve().parent / f"table_number_{slot_index + 1}.urdf"
            marker = self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(marker_path),
                    pos=(x, y, 0.865),
                    fixed=True,
                    prioritize_urdf_material=True,
                )
            )
            self.table_label_entities.append(marker)

        for index in range(2):
            waiter = self.scene.add_entity(
                gs.morphs.URDF(
                    file=self.waiter_urdf,
                    pos=(-8.0 + index, -2.25, 0.0),
                    fixed=True,
                    prioritize_urdf_material=True,
                )
            )
            self.person_entities.append(waiter)
        for index in range(2):
            customer = self.scene.add_entity(
                gs.morphs.URDF(
                    file=self.customer_urdf,
                    pos=(-4.0 + 4.0 * index, -6.5, 0.0),
                    fixed=True,
                    prioritize_urdf_material=True,
                )
            )
            self.person_entities.append(customer)

        for size, color in self.decor_specs:
            entity = self.scene.add_entity(
                morph=gs.morphs.Box(
                    size=size,
                    pos=(30.0, 30.0, float(size[2]) / 2.0),
                    fixed=True,
                ),
                surface=self._surface(color),
            )
            self.decor_entities.append(entity)

        self.package = self.scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.34, 0.34, 0.24),
                pos=(float(self.package_position[0]), float(self.package_position[1]), 1.05),
                fixed=True,
            ),
            surface=self._surface((0.96, 0.68, 0.08, 1.0)),
        )

        self.scene.build()
        fl = self.robot.get_joint("front_left_wheel_joint")
        fr = self.robot.get_joint("front_right_wheel_joint")
        rl = self.robot.get_joint("rear_left_wheel_joint")
        rr = self.robot.get_joint("rear_right_wheel_joint")
        self.wheel_dofs = [
            fl.dofs_idx_local[0],
            rl.dofs_idx_local[0],
            fr.dofs_idx_local[0],
            rr.dofs_idx_local[0],
        ]
        self._refresh_obstacle_data()

    def _refresh_obstacle_data(self) -> None:
        self.obstacle_data = [row.copy() for row in self.static_obstacle_data]
        self.obstacle_data.extend(row.copy() for row in self.table_obstacle_data)
        self.obstacle_data.extend(row.copy() for row in self.decor_obstacle_data)
        self.obstacle_data.extend(row.copy() for row in self.dynamic_obstacle_data)

    def _randomize_decor_obstacles(self) -> None:
        if not self.randomized_static_clutter:
            self.decor_obstacle_data = []
            for index, entity in enumerate(self.decor_entities):
                entity.set_pos(np.array([32.0 + index, 32.0, 0.5], dtype=np.float32))
            return

        candidate_slots = (
            (-9.0, -3.2),
            (9.0, -3.2),
            (-9.0, 1.5),
            (9.0, 1.5),
            (9.0, 5.6),
            (-4.0, 6.6),
            (0.0, 6.6),
            (4.0, 6.6),
        )
        available = [int(v) for v in self.np_random.permutation(len(candidate_slots))]
        rows = []
        placements = []
        for decor_index, (size, _) in enumerate(self.decor_specs):
            sx, sy, sz = size
            selected = None
            while available and selected is None:
                slot = np.asarray(candidate_slots[available.pop()], dtype=np.float32)
                centre = slot + self.np_random.uniform(-0.18, 0.18, size=2)
                overlaps_table = any(
                    abs(float(centre[0] - table[0])) < (sx + table[2]) / 2.0 + 0.35
                    and abs(float(centre[1] - table[1])) < (sy + table[3]) / 2.0 + 0.35
                    for table in self.table_obstacle_data
                )
                if not overlaps_table:
                    selected = centre
            if selected is None:
                selected = np.array([31.0 + decor_index, 31.0], dtype=np.float32)
            placements.append((selected, sz))
            rows.append(
                [float(selected[0]), float(selected[1]), float(sx), float(sy), float(sz)]
            )
        self.decor_obstacle_data = rows
        for entity, (centre, sz) in zip(self.decor_entities, placements, strict=True):
            entity.set_pos(
                np.array([float(centre[0]), float(centre[1]), float(sz) / 2.0], dtype=np.float32)
            )

    def _sample_slot_assignment(self) -> tuple[int, ...]:
        for _ in range(200):
            assignment = tuple(
                int(value)
                for value in self.np_random.choice(
                    len(self.TABLE_SLOTS), size=self.TABLE_COUNT, replace=False
                )
            )
            # Keep the restaurant visually balanced while preserving a fresh
            # randomized eight-table layout.  Every horizontal area is used
            # and no complete row is left looking empty.
            column_counts = [
                sum(slot % 3 == column for slot in assignment)
                for column in range(3)
            ]
            row_counts = [
                sum(slot // 3 == row for slot in assignment)
                for row in range(3)
            ]
            if min(column_counts) < 2 or min(row_counts) < 2:
                continue
            if self.previous_slot_assignment is None:
                return assignment
            if all(
                assignment[index] != self.previous_slot_assignment[index]
                for index in range(self.TABLE_COUNT)
            ):
                return assignment
        # Continuous jitter still changes every pose if a strict derangement is
        # unusually not sampled within the attempt budget.
        return assignment

    def _table_service_candidates(self, table_id: int) -> list[np.ndarray]:
        centre = self.table_centres[table_id]
        sx, sy = self.table_sizes[table_id]
        extra = self.PATH_INFLATION + 0.35
        return [
            centre + np.array([-(sx / 2.0 + extra), 0.0], dtype=np.float32),
            centre + np.array([(sx / 2.0 + extra), 0.0], dtype=np.float32),
            centre + np.array([0.0, -(sy / 2.0 + extra)], dtype=np.float32),
            centre + np.array([0.0, (sy / 2.0 + extra)], dtype=np.float32),
        ]

    def _candidate_inside_world(self, point: np.ndarray) -> bool:
        margin = 1.0
        return bool(
            self.WORLD_X_MIN + margin < point[0] < self.WORLD_X_MAX - margin
            and self.WORLD_Y_MIN + margin < point[1] < self.WORLD_Y_MAX - margin
        )

    def _randomize_tables_and_order(self) -> None:
        for _ in range(120):
            assignment = self._sample_slot_assignment()
            centres = {}
            yaws = {}
            sizes = {}
            obstacle_rows = []

            for index, slot_index in enumerate(assignment):
                table_id = index + 1
                base = np.asarray(self.TABLE_SLOTS[slot_index], dtype=np.float32)
                jitter = self.np_random.uniform(-0.38, 0.38, size=2).astype(np.float32)
                centre = base + jitter
                # Chairs use the long dimension of the table collision box.
                # Keeping that dimension horizontal reserves two genuinely
                # passable east-west traffic aisles for robot + pedestrian.
                yaw_deg = 90.0
                sx, sy = self.TABLE_SIZE_UNROTATED
                if yaw_deg == 90.0:
                    sx, sy = sy, sx
                centres[table_id] = centre
                yaws[table_id] = yaw_deg
                sizes[table_id] = (float(sx), float(sy))
                obstacle_rows.append(
                    [float(centre[0]), float(centre[1]), float(sx), float(sy), 1.25]
                )

            centre_values = list(centres.values())
            if any(
                float(np.linalg.norm(centre_values[first] - centre_values[second]))
                < self.TABLE_MIN_CENTRE_DISTANCE
                for first in range(len(centre_values))
                for second in range(first + 1, len(centre_values))
            ):
                continue

            self.table_centres = centres
            self.table_yaws = yaws
            self.table_sizes = sizes
            self.table_obstacle_data = obstacle_rows
            self.dynamic_obstacle_data = []
            self._randomize_decor_obstacles()
            self._refresh_obstacle_data()

            if self.plan_path(self.start_position, self.package_position) is None:
                continue

            if self.requested_table_id is None:
                table_order = [
                    int(value)
                    for value in self.np_random.permutation(
                        np.arange(1, self.TABLE_COUNT + 1)
                    )
                ]
            else:
                table_order = [int(self.requested_table_id)]

            chosen_table = None
            chosen_goal = None
            for table_id in table_order:
                valid_options = []
                for candidate in self._table_service_candidates(table_id):
                    if not self._candidate_inside_world(candidate):
                        continue
                    path = self.plan_path(self.package_position, candidate)
                    if path is not None:
                        valid_options.append((len(path), candidate))
                if valid_options:
                    _, chosen_goal = min(valid_options, key=lambda item: item[0])
                    chosen_table = table_id
                    break

            if chosen_table is None:
                continue

            self.selected_table_id = int(chosen_table)
            self.goal_position = np.asarray(chosen_goal, dtype=np.float32)
            self.previous_slot_assignment = assignment

            for table_id, entity in enumerate(self.table_entities, start=1):
                centre = self.table_centres[table_id]
                yaw = math.radians(self.table_yaws[table_id])
                entity.set_pos(
                    np.array([float(centre[0]), float(centre[1]), 0.0], dtype=np.float32)
                )
                entity.set_quat(
                    np.array(
                        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                        dtype=np.float32,
                    )
                )
                marker = self.table_label_entities[table_id - 1]
                marker.set_pos(
                    np.array(
                        [float(centre[0]), float(centre[1]), 0.865],
                        dtype=np.float32,
                    )
                )
                marker.set_quat(
                    np.array(
                        [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                        dtype=np.float32,
                    )
                )

            self.current_order = self.order_system.create_order(
                self.np_random,
                table_id=self.selected_table_id,
                item=self.requested_item,
                quantity=self.requested_quantity,
            )
            self.order_system.mark_preparing()
            return

        raise RuntimeError("Could not generate a solvable eight-table restaurant layout")

    def _randomize_people(self) -> None:
        # One moving person per separated half-aisle.  Routes never overlap,
        # so two people cannot spawn on top of one another or walk as a pair.
        # They still cross the robot's route and remain dynamic obstacles.
        routes = [
            (np.array([-9.0, -2.55]), np.array([-0.9, -2.55])),
            (np.array([0.9, -2.20]), np.array([9.0, -2.20])),
            (np.array([-9.0, 2.20]), np.array([-0.9, 2.20])),
            (np.array([0.9, 2.55]), np.array([9.0, 2.55])),
        ]
        route_ids = self.np_random.choice(len(routes), size=4, replace=False)
        self.person_states = []
        current_directions = []
        for person_index, route_id in enumerate(route_ids):
            start, end = routes[int(route_id)]
            if self.previous_person_forward is None:
                forward = bool(self.np_random.integers(0, 2))
            else:
                forward = not self.previous_person_forward[person_index]
            current_directions.append(forward)
            if not forward:
                start, end = end.copy(), start.copy()
            is_waiter = person_index < 2
            # Normal walking traffic: fast enough to clear a yielded lane,
            # while remaining below the delivery robot's cruise speed.
            speed_range = (0.42, 0.52) if is_waiter else (0.36, 0.46)
            speed = float(self.np_random.uniform(*speed_range)) * self.people_speed_scale
            length = float(np.linalg.norm(end - start))
            # Spread initial positions through their own zones instead of
            # grouping all people close to one route endpoint.
            offset = float(self.np_random.uniform(0.08, 0.92) * length)
            delta = end - start
            if abs(float(delta[0])) >= abs(float(delta[1])):
                direction_label = "left-to-right" if delta[0] > 0 else "right-to-left"
            else:
                direction_label = "bottom-to-top" if delta[1] > 0 else "top-to-bottom"
            self.person_states.append(
                {
                    "start": start.astype(np.float32),
                    "end": end.astype(np.float32),
                    "speed": speed,
                    "offset": offset,
                    "route_progress": offset,
                    "progress_at_pose": offset,
                    "state_time": 0.0,
                    "last_progress_time": 0.0,
                    "current_speed": speed,
                    # The waiter's carried tray is wider than the legs/body.
                    "footprint": 0.68 if is_waiter else 0.54,
                    "type": f"waiter-{person_index + 1}" if is_waiter else f"customer-{person_index - 1}",
                    "initial_direction": direction_label,
                    # Realistic mixed traffic: some people cooperate with the robot,
                    # while others continue normally and must be avoided by DWA.
                    "cooperative": bool(
                        self.np_random.random() < self.human_cooperation_probability
                    ),
                    "avoidance_offset": 0.0,
                    "avoidance_side": 1.0,
                    "reciprocal_active": False,
                    "reciprocal_events": 0,
                    "cooperative_steps": 0,
                    "target_avoidance_offset": 0.0,
                    "lateral_speed_mps": 0.0,
                }
            )
        self.previous_person_forward = tuple(current_directions)

    def _person_side_is_free(self, point: np.ndarray, footprint: float) -> bool:
        margin = footprint / 2.0 + 0.10
        layout = (
            self.static_obstacle_data
            + self.table_obstacle_data
            + self.decor_obstacle_data
        )
        for x, y, sx, sy, _ in layout:
            if (
                abs(float(point[0]) - float(x)) <= float(sx) / 2.0 + margin
                and abs(float(point[1]) - float(y)) <= float(sy) / 2.0 + margin
            ):
                return False
        return bool(
            self.WORLD_X_MIN + margin < float(point[0]) < self.WORLD_X_MAX - margin
            and self.WORLD_Y_MIN + margin < float(point[1]) < self.WORLD_Y_MAX - margin
        )

    def _update_reciprocal_offset(
        self,
        state: dict,
        nominal_xy: np.ndarray,
        heading: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Optionally yield; non-cooperative people keep their nominal route."""
        if not bool(state.get("cooperative", True)):
            state["reciprocal_active"] = False
            state["target_avoidance_offset"] = 0.0
            state["avoidance_offset"] = 0.0
            state["lateral_speed_mps"] = 0.0
            return nominal_xy, 0.0

        robot_xy = self._get_robot_position()
        robot_yaw = float(self._get_robot_yaw())
        robot_forward = np.array(
            [math.cos(robot_yaw), math.sin(robot_yaw)], dtype=np.float32
        )
        relative = nominal_xy - robot_xy
        distance = float(np.linalg.norm(relative))
        person_velocity = heading * float(state.get("current_speed", state["speed"]))
        robot_speed = max(float(self.linear_velocity), 0.0)
        relative_velocity = person_velocity - robot_speed * robot_forward
        velocity_sq = float(np.dot(relative_velocity, relative_velocity))
        raw_cpa_time = (
            99.0
            if velocity_sq < 1e-8
            else -float(np.dot(relative, relative_velocity)) / velocity_sq
        )
        cpa_time = float(np.clip(raw_cpa_time, 0.0, self.RECIPROCAL_TRIGGER_TTC))
        cpa_separation = float(
            np.linalg.norm(relative + relative_velocity * cpa_time)
        )
        combined_distance = (
            math.hypot(self.ROBOT_COLLISION_HALF_LENGTH, self.ROBOT_COLLISION_HALF_WIDTH)
            + float(state["footprint"]) / 2.0
        )
        facing_each_other = float(np.dot(robot_forward, heading)) < -0.55
        # Use the same cooperative yield for an imminent crossing, not only a
        # perfectly head-on approach.  The old facing-only gate produced zero
        # person-yield events whenever the horizontal traffic lane crossed the
        # robot's vertical route.
        collision_course = cpa_separation <= combined_distance + 0.30
        person_can_see_robot = float(np.dot(-relative, heading)) > -0.25
        trigger = bool(
            robot_speed > 0.12
            and distance <= self.RECIPROCAL_TRIGGER_DISTANCE
            and (facing_each_other or collision_course)
            and person_can_see_robot
            and 0.05 <= raw_cpa_time <= self.RECIPROCAL_TRIGGER_TTC
            and collision_course
        )

        active = bool(state.get("reciprocal_active", False))
        if trigger and not active:
            person_left = np.array([-heading[1], heading[0]], dtype=np.float32)
            preferred_side = 1.0  # person's own left, as requested
            preferred_point = (
                nominal_xy
                + person_left * preferred_side * self.RECIPROCAL_SIDE_OFFSET
            )
            if not self._person_side_is_free(
                preferred_point, float(state["footprint"])
            ):
                preferred_side = 0.0
            state["avoidance_side"] = preferred_side
            state["reciprocal_active"] = True
            state["reciprocal_events"] = int(state.get("reciprocal_events", 0)) + 1
            state["cooperative_steps"] = 0
            active = True

        if active:
            state["cooperative_steps"] = int(state.get("cooperative_steps", 0)) + 1
            separating = float(np.dot(relative, relative_velocity)) > 0.0
            held_long_enough = (
                int(state["cooperative_steps"]) >= self.COOPERATIVE_MIN_ACTIVE_STEPS
            )
            if held_long_enough and separating and distance >= 2.05:
                state["reciprocal_active"] = False
                active = False

        target_offset = (
            float(state.get("avoidance_side", 1.0)) * self.RECIPROCAL_SIDE_OFFSET
            if active
            else 0.0
        )
        state["target_avoidance_offset"] = target_offset
        old_offset = float(state.get("avoidance_offset", 0.0))
        change = float(
            np.clip(
                target_offset - old_offset,
                -self.RECIPROCAL_OFFSET_STEP,
                self.RECIPROCAL_OFFSET_STEP,
            )
        )
        new_offset = old_offset + change
        state["avoidance_offset"] = new_offset
        state["lateral_speed_mps"] = change / (self.ACTION_REPEAT * 0.01)
        person_left = np.array([-heading[1], heading[0]], dtype=np.float32)
        return nominal_xy + person_left * new_offset, change

    def _update_people(self) -> None:
        if not self.dynamic_people_enabled or self.active_people_count <= 0:
            self.dynamic_obstacle_data = []
            for index, entity in enumerate(self.person_entities):
                entity.set_pos(np.array([30.0 + index, 30.0, 0.0], dtype=np.float32))
            self._refresh_obstacle_data()
            return

        self.dynamic_obstacle_data = []
        for index, state in enumerate(self.person_states):
            if index >= self.active_people_count:
                self.person_entities[index].set_pos(
                    np.array([30.0 + index, 30.0, 0.0], dtype=np.float32)
                )
                state["current_xy"] = np.array([30.0 + index, 30.0], dtype=np.float32)
                state["velocity_xy"] = np.zeros(2, dtype=np.float32)
                continue
            start, end = state["start"], state["end"]
            vector = end - start
            length = float(np.linalg.norm(vector))
            direction = vector / max(length, 1e-6)
            travelled = float(
                state.get("route_progress", state["offset"])
            ) % (2.0 * length)
            if travelled <= length:
                xy = start + direction * travelled
                heading = direction
            else:
                xy = end - direction * (travelled - length)
                heading = -direction
            nominal_xy = xy.astype(np.float32)
            xy, lateral_change = self._update_reciprocal_offset(
                state, nominal_xy, heading.astype(np.float32)
            )
            active = bool(state.get("reciprocal_active", False))
            current_speed = float(state["speed"]) * (
                self.PERSON_YIELD_SPEED_SCALE if active else 1.0
            )
            state["current_speed"] = current_speed
            state["progress_at_pose"] = travelled
            state["state_time"] = float(self.sim_time)
            route_dt = max(
                0.0,
                float(self.sim_time)
                - float(state.get("last_progress_time", self.sim_time)),
            )
            state["route_progress"] = (
                travelled + current_speed * route_dt
            ) % (2.0 * length)
            state["last_progress_time"] = float(self.sim_time)
            yaw = math.atan2(float(heading[1]), float(heading[0]))
            self.person_entities[index].set_pos(
                np.array([float(xy[0]), float(xy[1]), 0.0], dtype=np.float32)
            )
            self.person_entities[index].set_quat(
                np.array(
                    [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
                    dtype=np.float32,
                )
            )
            footprint = float(state["footprint"])
            state["current_xy"] = xy.astype(np.float32)
            person_left = np.array([-heading[1], heading[0]], dtype=np.float32)
            lateral_speed = lateral_change / (self.ACTION_REPEAT * 0.01)
            state["velocity_xy"] = (
                heading * current_speed + person_left * lateral_speed
            ).astype(np.float32)
            self.dynamic_obstacle_data.append(
                [float(xy[0]), float(xy[1]), footprint, footprint, 1.75]
            )
        self._refresh_obstacle_data()

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _get_robot_position(self) -> np.ndarray:
        pos = self.robot.get_pos().cpu().numpy()
        return np.array([pos[0], pos[1]], dtype=np.float32)

    def _get_robot_yaw(self) -> float:
        w, x, y, z = self.robot.get_quat().cpu().numpy()
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _get_active_target(self) -> np.ndarray:
        return self.goal_position if self.package_picked else self.package_position

    def _get_target_information(self) -> tuple[float, float]:
        delta = self._get_active_target() - self._get_robot_position()
        distance = float(np.linalg.norm(delta))
        target_angle = math.atan2(float(delta[1]), float(delta[0]))
        return distance, self._normalize_angle(target_angle - self._get_robot_yaw())

    @staticmethod
    def _ray_rectangle_distance(origin, direction, cx, cy, sx, sy):
        min_x, max_x = cx - sx / 2.0, cx + sx / 2.0
        min_y, max_y = cy - sy / 2.0, cy + sy / 2.0
        ox, oy = float(origin[0]), float(origin[1])
        dx, dy = float(direction[0]), float(direction[1])
        distances = []
        if abs(dx) > 1e-8:
            for boundary in (min_x, max_x):
                t = (boundary - ox) / dx
                y_hit = oy + t * dy
                if t >= 0.0 and min_y <= y_hit <= max_y:
                    distances.append(t)
        if abs(dy) > 1e-8:
            for boundary in (min_y, max_y):
                t = (boundary - oy) / dy
                x_hit = ox + t * dx
                if t >= 0.0 and min_x <= x_hit <= max_x:
                    distances.append(t)
        return min(distances) if distances else None

    def _get_lidar(self) -> np.ndarray:
        origin = self._get_robot_position()
        yaw = self._get_robot_yaw()
        values = []
        for index in range(self.NUM_LIDAR_RAYS):
            angle = yaw + 2.0 * math.pi * index / self.NUM_LIDAR_RAYS
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            closest = self.LIDAR_MAX_DISTANCE
            for x, y, sx, sy, _ in self.obstacle_data:
                distance = self._ray_rectangle_distance(origin, direction, x, y, sx, sy)
                if distance is not None:
                    closest = min(closest, distance)
            values.append(
                float(np.clip(closest, self.LIDAR_MIN_DISTANCE, self.LIDAR_MAX_DISTANCE))
                / self.LIDAR_MAX_DISTANCE
            )
        return np.asarray(values, dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        distance, angle_error = self._get_target_information()
        lidar = self._get_lidar()
        if self.previous_lidar is None:
            lidar_delta = np.zeros_like(lidar)
        else:
            lidar_delta = np.clip((lidar - self.previous_lidar) * 4.0, -1.0, 1.0)
        self.previous_lidar = lidar.copy()
        extra = np.array(
            [
                distance,
                angle_error,
                self.linear_velocity,
                self.angular_velocity,
                1.0 if self.package_picked else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([lidar, lidar_delta, extra]).astype(np.float32)

    def predict_dynamic_risk(self, action=None, horizon: float = 2.0) -> dict:
        robot_xy = self._get_robot_position()
        yaw = self._get_robot_yaw()
        if action is None:
            linear_speed = float(self.linear_velocity)
        else:
            linear_speed = float(np.clip(action[0], -1.0, 1.0)) * self.MAX_LINEAR_SPEED
        robot_velocity = linear_speed * np.array(
            [math.cos(yaw), math.sin(yaw)], dtype=np.float32
        )
        best_clearance = float("inf")
        best_time = float("inf")
        nearest_type = None
        for index, state in enumerate(self.person_states[: self.active_people_count]):
            if "current_xy" not in state:
                continue
            relative_position = np.asarray(state["current_xy"]) - robot_xy
            relative_velocity = np.asarray(state["velocity_xy"]) - robot_velocity
            velocity_sq = float(np.dot(relative_velocity, relative_velocity))
            raw_cpa_time = (
                0.0
                if velocity_sq < 1e-8
                else -float(np.dot(relative_position, relative_velocity)) / velocity_sq
            )
            closest_time = float(np.clip(raw_cpa_time, 0.0, horizon))
            separation = float(
                np.linalg.norm(relative_position + relative_velocity * closest_time)
            )
            combined_radius = 0.72 + float(state["footprint"]) / 2.0
            clearance = separation - combined_radius
            entry_time = 99.0
            if velocity_sq > 1e-8:
                protected_radius = combined_radius + 0.12
                b = 2.0 * float(np.dot(relative_position, relative_velocity))
                c = float(np.dot(relative_position, relative_position)) - protected_radius**2
                discriminant = b * b - 4.0 * velocity_sq * c
                if discriminant >= 0.0:
                    first_root = (-b - math.sqrt(discriminant)) / (2.0 * velocity_sq)
                    second_root = (-b + math.sqrt(discriminant)) / (2.0 * velocity_sq)
                    if second_root >= 0.0 and raw_cpa_time > 0.0:
                        entry_time = max(0.0, first_root)
            if clearance < best_clearance:
                best_clearance = clearance
                best_time = entry_time
                nearest_type = state["type"]
        if not np.isfinite(best_clearance):
            return {"clearance": 99.0, "ttc": 99.0, "person": None, "danger": False}
        danger = bool(best_clearance < 0.12 and 0.0 <= best_time <= horizon)
        return {
            "clearance": float(best_clearance),
            "ttc": float(best_time if danger else 99.0),
            "person": nearest_type,
            "danger": danger,
        }

    def _apply_action(self, action) -> None:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.linear_velocity = float(action[0]) * self.MAX_LINEAR_SPEED
        self.angular_velocity = float(action[1]) * self.MAX_ANGULAR_SPEED
        left = (
            self.linear_velocity - self.angular_velocity * self.TRACK_WIDTH / 2.0
        ) / self.WHEEL_RADIUS
        right = (
            self.linear_velocity + self.angular_velocity * self.TRACK_WIDTH / 2.0
        ) / self.WHEEL_RADIUS
        self.robot.control_dofs_velocity(
            np.array([left, left, right, right], dtype=np.float32), self.wheel_dofs
        )

    def _check_collision_against(self, layout) -> bool:
        position = self._get_robot_position()
        yaw = self._get_robot_yaw()
        half_length = self.ROBOT_COLLISION_HALF_LENGTH + self.COLLISION_CONTACT_MARGIN
        half_width = self.ROBOT_COLLISION_HALF_WIDTH + self.COLLISION_CONTACT_MARGIN
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        left = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        for x, y, sx, sy, _ in layout:
            delta = np.array([x, y], dtype=np.float32) - position
            ohx, ohy = sx / 2.0, sy / 2.0
            robot_x = half_length * abs(float(forward[0])) + half_width * abs(float(left[0]))
            robot_y = half_length * abs(float(forward[1])) + half_width * abs(float(left[1]))
            if abs(float(delta[0])) > robot_x + ohx:
                continue
            if abs(float(delta[1])) > robot_y + ohy:
                continue
            obstacle_forward = ohx * abs(float(forward[0])) + ohy * abs(float(forward[1]))
            obstacle_left = ohx * abs(float(left[0])) + ohy * abs(float(left[1]))
            if abs(float(np.dot(delta, forward))) > half_length + obstacle_forward:
                continue
            if abs(float(np.dot(delta, left))) > half_width + obstacle_left:
                continue
            return True
        return False

    def _check_collision(self) -> bool:
        return self._check_collision_against(self.obstacle_data)

    def _classify_collision(self) -> str | None:
        if self._check_collision_against(self.dynamic_obstacle_data):
            return "dynamic"
        static_layout = (
            self.static_obstacle_data
            + self.table_obstacle_data
            + self.decor_obstacle_data
        )
        if self._check_collision_against(static_layout):
            return "static"
        return None

    def _check_out_of_bounds(self) -> bool:
        x, y = self._get_robot_position()
        return bool(
            x < self.WORLD_X_MIN
            or x > self.WORLD_X_MAX
            or y < self.WORLD_Y_MIN
            or y > self.WORLD_Y_MAX
        )

    def _calculate_reward(self, progress: float) -> float:
        reward = 15.0 * float(progress) - 0.02
        if self.just_picked_package:
            reward += 25.0
        if self.package_delivered:
            reward += 150.0
        if self.collision:
            reward -= 100.0
        if self.out_of_bounds:
            reward -= 100.0
        min_distance = float(np.min(self._get_lidar()) * self.LIDAR_MAX_DISTANCE)
        if min_distance < 1.10:
            reward -= 0.45 * ((1.10 - min_distance) / 1.10)
        dynamic_risk = self.predict_dynamic_risk()
        if dynamic_risk["clearance"] < 1.0:
            reward -= 1.8 * (1.0 - max(dynamic_risk["clearance"], 0.0))
        if dynamic_risk["danger"]:
            reward -= 2.5 * (2.0 - min(dynamic_risk["ttc"], 2.0))
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        requested_table = options.get("table_id", self.requested_table_id)
        self.requested_table_id = None if requested_table in (None, 0) else int(requested_table)
        self.requested_item = options.get("item", self.requested_item)
        requested_quantity = options.get("quantity", self.requested_quantity)
        self.requested_quantity = (
            None if requested_quantity is None else int(requested_quantity)
        )

        self.package_picked = False
        self.package_delivered = False
        self.just_picked_package = False
        self.collision = False
        self.collision_type = None
        self.out_of_bounds = False
        self.current_step = 0
        self.sim_time = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.social_yield_active = False
        self.social_yield_events = 0
        self.previous_lidar = None

        self.robot.set_pos(
            np.array([self.start_position[0], self.start_position[1], 0.0], dtype=np.float32)
        )
        self.robot.set_quat(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        self.robot.control_dofs_velocity(np.zeros(4, dtype=np.float32), self.wheel_dofs)
        self.package.set_pos(
            np.array([self.package_position[0], self.package_position[1], 1.05], dtype=np.float32)
        )

        self._randomize_tables_and_order()
        self._randomize_people()
        self._update_people()
        for _ in range(10):
            self.scene.step()
        return self._get_observation(), self._info(success=False)

    def begin_continuation_order(
        self,
        table_id: int,
        item: str | None = None,
        quantity: int | None = None,
    ):
        """Start the next queued delivery from the robot's current position.

        The first order uses :meth:`reset` and collects a package at the
        counter.  Once that delivery succeeds, later orders stay in the same
        restaurant scene and begin with a tray already on the robot.  This
        preserves the robot pose, table layout and moving-person timeline;
        only the delivery target and order record change.
        """
        table_id = int(table_id)
        if table_id not in range(1, self.TABLE_COUNT + 1):
            raise ValueError("table_id must be 1-8")

        robot_xy = self._get_robot_position().astype(np.float32)
        valid_options = []
        for candidate in self._table_service_candidates(table_id):
            if not self._candidate_inside_world(candidate):
                continue
            path = self.plan_path(robot_xy, candidate)
            if path is not None:
                valid_options.append((len(path), candidate))
        if not valid_options:
            raise RuntimeError(
                f"No valid direct route from current robot position to table {table_id}"
            )

        _, goal = min(valid_options, key=lambda item: item[0])
        self.requested_table_id = table_id
        self.requested_item = item
        self.requested_quantity = quantity
        self.selected_table_id = table_id
        self.goal_position = np.asarray(goal, dtype=np.float32)

        self.current_order = self.order_system.create_order(
            self.np_random,
            table_id=table_id,
            item=item,
            quantity=quantity,
        )
        self.order_system.mark_preparing()
        self.order_system.mark_on_robot()
        self.package_picked = True
        self.package_delivered = False
        self.just_picked_package = False
        self.collision = False
        self.collision_type = None
        self.out_of_bounds = False
        self.current_step = 0
        # Keep sim_time continuous so moving people do not jump backwards.
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.previous_lidar = None

        robot_3d = self.robot.get_pos().cpu().numpy()
        self.package.set_pos(
            np.array([robot_3d[0], robot_3d[1], robot_3d[2] + 1.08], dtype=np.float32)
        )
        self._update_people()
        for _ in range(3):
            self.scene.step()
        return self._get_observation(), self._info(success=False)

    def step(self, action):
        self.current_step += 1
        self.just_picked_package = False
        target_before = self._get_active_target().copy()
        previous_distance = float(np.linalg.norm(self._get_robot_position() - target_before))

        self.sim_time += self.ACTION_REPEAT * 0.01
        self._update_people()
        self._apply_action(action)
        for _ in range(self.ACTION_REPEAT):
            self.scene.step()

        robot_position = self._get_robot_position()
        progress = previous_distance - float(np.linalg.norm(robot_position - target_before))
        if not self.package_picked:
            if float(np.linalg.norm(robot_position - self.package_position)) < self.pickup_distance:
                self.package_picked = True
                self.just_picked_package = True
                self.order_system.mark_on_robot()

        if self.package_picked and not self.package_delivered:
            robot_3d = self.robot.get_pos().cpu().numpy()
            self.package.set_pos(
                np.array([robot_3d[0], robot_3d[1], robot_3d[2] + 1.08], dtype=np.float32)
            )
            if float(np.linalg.norm(robot_position - self.goal_position)) < self.delivery_distance:
                self.package_delivered = True
                self.order_system.mark_delivered()
                centre = self.table_centres[self.selected_table_id]
                self.package.set_pos(
                    np.array([centre[0], centre[1], 0.98], dtype=np.float32)
                )

        self.collision = self._check_collision()
        self.collision_type = self._classify_collision() if self.collision else None
        self.out_of_bounds = self._check_out_of_bounds()
        if (self.collision or self.out_of_bounds) and not self.package_delivered:
            self.order_system.mark_failed()
        reward = self._calculate_reward(progress)
        terminated = bool(self.package_delivered or self.collision or self.out_of_bounds)
        distance_to_goal = float(np.linalg.norm(robot_position - self.goal_position))
        final_approach = bool(
            self.package_picked
            and not self.package_delivered
            and distance_to_goal <= self.delivery_grace_distance
        )
        regular_timeout = self.current_step >= self.max_steps and not final_approach
        grace_timeout = self.current_step >= self.max_steps + self.delivery_grace_steps
        truncated = bool((regular_timeout or grace_timeout) and not terminated)
        info = self._info(success=self.package_delivered)
        info["progress"] = float(progress)
        return self._get_observation(), reward, terminated, truncated, info

    def _info(self, *, success: bool) -> dict:
        distance, _ = self._get_target_information()
        dynamic_risk = self.predict_dynamic_risk()
        return {
            "phase": "delivered"
            if self.package_delivered
            else ("to_table" if self.package_picked else "to_pickup"),
            "selected_table": self.selected_table_id,
            "order": self.current_order.to_dict() if self.current_order else None,
            "package_picked": self.package_picked,
            "package_delivered": self.package_delivered,
            "success": bool(success),
            "collision": self.collision,
            "collision_type": self.collision_type,
            "out_of_bounds": self.out_of_bounds,
            "distance_to_target": float(distance),
            "min_clearance": float(np.min(self._get_lidar()) * self.LIDAR_MAX_DISTANCE),
            "dynamic_clearance": dynamic_risk["clearance"],
            "dynamic_ttc": dynamic_risk["ttc"],
            "dynamic_person": dynamic_risk["person"],
            "dynamic_danger": dynamic_risk["danger"],
            "social_yield_active": any(
                bool(state.get("reciprocal_active", False))
                for state in self.person_states[: self.active_people_count]
            ),
            "social_yield_events": sum(
                int(state.get("reciprocal_events", 0))
                for state in self.person_states[: self.active_people_count]
            ),
            "reciprocal_active": any(
                bool(state.get("reciprocal_active", False))
                for state in self.person_states[: self.active_people_count]
            ),
            "reciprocal_events": sum(
                int(state.get("reciprocal_events", 0))
                for state in self.person_states[: self.active_people_count]
            ),
            "yielding_people": [
                state["type"]
                for state in self.person_states[: self.active_people_count]
                if state.get("reciprocal_active", False)
            ],
            "table_positions": {
                table_id: [float(pos[0]), float(pos[1])]
                for table_id, pos in self.table_centres.items()
            },
            "people": [
                {
                    "type": state["type"],
                    "speed_mps": float(state.get("current_speed", state["speed"])),
                    "initial_direction": state["initial_direction"],
                    "cooperative": bool(state.get("cooperative", True)),
                }
                for index, state in enumerate(
                    self.person_states[: self.active_people_count]
                )
            ],
            "step": self.current_step,
            "sim_time": float(self.sim_time),
        }

    def set_curriculum(self, stage: int) -> None:
        settings = {
            0: (0, 0.50),
            1: (1, 0.60),
            2: (2, 0.78),
            3: (4, 1.00),
        }
        if stage not in settings:
            raise ValueError("Curriculum stage must be 0, 1, 2, or 3")
        self.active_people_count, self.people_speed_scale = settings[stage]
        self.dynamic_people_enabled = self.active_people_count > 0

    def _path_world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - self.WORLD_X_MIN) / self.PATH_GRID_RESOLUTION)),
            int(round((y - self.WORLD_Y_MIN) / self.PATH_GRID_RESOLUTION)),
        )

    def _path_grid_to_world(self, node) -> np.ndarray:
        return np.array(
            [
                self.WORLD_X_MIN + node[0] * self.PATH_GRID_RESOLUTION,
                self.WORLD_Y_MIN + node[1] * self.PATH_GRID_RESOLUTION,
            ],
            dtype=np.float32,
        )

    def _build_path_occupancy_grid(self, obstacle_layout=None) -> np.ndarray:
        # Moving people are deliberately excluded from the global grid. A*
        # remains stable while predictive DWA handles their short-lived motion.
        layout = (
            self.static_obstacle_data
            + self.table_obstacle_data
            + self.decor_obstacle_data
            if obstacle_layout is None
            else obstacle_layout
        )
        nx = int(round((self.WORLD_X_MAX - self.WORLD_X_MIN) / self.PATH_GRID_RESOLUTION)) + 1
        ny = int(round((self.WORLD_Y_MAX - self.WORLD_Y_MIN) / self.PATH_GRID_RESOLUTION)) + 1
        occupancy = np.zeros((nx, ny), dtype=bool)
        for x, y, sx, sy, _ in layout:
            gx0, gy0 = self._path_world_to_grid(
                x - sx / 2.0 - self.PATH_INFLATION,
                y - sy / 2.0 - self.PATH_INFLATION,
            )
            gx1, gy1 = self._path_world_to_grid(
                x + sx / 2.0 + self.PATH_INFLATION,
                y + sy / 2.0 + self.PATH_INFLATION,
            )
            gx0, gy0 = max(gx0, 0), max(gy0, 0)
            gx1, gy1 = min(gx1, nx - 1), min(gy1, ny - 1)
            if gx0 <= gx1 and gy0 <= gy1:
                occupancy[gx0 : gx1 + 1, gy0 : gy1 + 1] = True
        return occupancy

    def plan_path(self, start_xy=None, goal_xy=None):
        start_xy = self._get_robot_position() if start_xy is None else np.asarray(start_xy)
        goal_xy = self._get_active_target() if goal_xy is None else np.asarray(goal_xy)
        occupancy = self._build_path_occupancy_grid()
        start = self._path_world_to_grid(float(start_xy[0]), float(start_xy[1]))
        goal = self._path_world_to_grid(float(goal_xy[0]), float(goal_xy[1]))
        nx, ny = occupancy.shape

        def valid(node):
            x, y = node
            return 0 <= x < nx and 0 <= y < ny and not occupancy[x, y]

        if not valid(start) or not valid(goal):
            return None
        neighbours = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        open_heap = [(math.dist(start, goal), 0.0, start)]
        came_from = {start: None}
        costs = {start: 0.0}
        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)
            if current == goal:
                break
            if current_cost > costs.get(current, float("inf")):
                continue
            cx, cy = current
            for dx, dy, step_cost in neighbours:
                nxt = (cx + dx, cy + dy)
                if not valid(nxt):
                    continue
                if dx and dy and (occupancy[cx + dx, cy] or occupancy[cx, cy + dy]):
                    continue
                new_cost = current_cost + step_cost
                if new_cost < costs.get(nxt, float("inf")):
                    costs[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(
                        open_heap,
                        (new_cost + math.dist(nxt, goal), new_cost, nxt),
                    )
        if goal not in came_from:
            return None
        nodes = []
        current = goal
        while current is not None:
            nodes.append(current)
            current = came_from[current]
        nodes.reverse()
        return [self._path_grid_to_world(node) for node in nodes]

    def close(self):
        pass
