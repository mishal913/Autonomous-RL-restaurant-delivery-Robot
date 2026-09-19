import genesis as gs
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import math
import heapq


class DeliveryRobotEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render=True, task_mode="full"):

        super().__init__()

        self.render_enabled = render

        # ========================================================
        # TASK MODE
        # ========================================================
        #
        # "full":
        #     Normal delivery task:
        #     START -> PACKAGE -> GOAL with obstacles.
        #
        # "goal_only":
        #     Stage-0 learning task.
        #     The robot starts logically carrying the package,
        #     obstacles are disabled, and SAC only has to learn:
        #
        #         START -> GOAL
        #
        # "package_goal":
        #     Stage-1 learning task.
        #     The package is present at its normal location,
        #     obstacles are disabled, and the robot must learn:
        #
        #         START -> PACKAGE -> GOAL
        #
        # "one_fixed":
        #     Hard Stage-2 task.
        #     ONE fixed obstacle is placed directly across the
        #     package->goal route.
        #
        # "one_offset":
        #     Easier Stage-2A task.
        #     ONE obstacle is placed near, but not directly across,
        #     the nominal package->goal route.
        #
        # "one_mid":
        #     Stage-2B task.
        #     One fixed obstacle at (3.8, 3.5).
        #
        # "one_random_narrow":
        #     Stage-2C task.
        #     One obstacle is randomized every episode:
        #         x = 3.8
        #         y ~ Uniform(3.2, 3.5)
        #
        # "one_random_wide":
        #     Stage-2D task.
        #     One obstacle:
        #         x = 3.8
        #         y ~ Uniform(2.8, 3.5)
        #
        # "one_random_wider":
        #     Stage-2E task.
        #     One obstacle:
        #         x = 3.8
        #         y ~ Uniform(2.4, 3.5)
        #
        # "two_random":
        #     Hard Stage-3 task with TWO strongly randomized
        #     obstacles.
        #
        # "two_random_easy":
        #     Stage-3A bridge task with two obstacles.
        #
        # "one_random_routewide":
        #     Generalization-training task.
        #
        #     Exactly ONE obstacle is active, but unlike Stage 2
        #     its x-position is NOT fixed at 3.8.
        #
        #     Most episodes sample an obstacle around different
        #     portions of the package->goal corridor.
        #
        #     Some episodes deliberately sample the later/lower
        #     region where the O2-only diagnostic exposed weaker
        #     generalization.
        #
        #     Reward, observations, LiDAR and actions are unchanged.
        #
        # "full":
        #     Normal delivery task with obstacle curriculum.
        #
        # All modes use the SAME observation/action interface.
        # ========================================================

        if task_mode not in [
            "full",
            "goal_only",
            "package_goal",
            "one_fixed",
            "one_offset",
            "one_mid",
            "one_random_narrow",
            "one_random_wide",
            "one_random_wider",
            "two_random",
            "two_random_easy",
            "one_random_routewide",
            "multi_random_routewide",
            "three_random_routewide",
            "four_random_routewide",
            "showcase_six_random",
            "showcase_dual_delivery"
        ]:

            raise ValueError(
                "task_mode must be "
                "'full', 'goal_only', 'package_goal', "
                "'one_fixed', 'one_offset', 'one_mid', "
                "'one_random_narrow', 'one_random_wide', "
                "'one_random_wider', 'two_random', "
                "'two_random_easy', 'one_random_routewide', "
                "'multi_random_routewide', 'three_random_routewide', "
                "'four_random_routewide', 'showcase_six_random', "
                "or 'showcase_dual_delivery'"
            )

        self.task_mode = task_mode

        # ========================================================
        # STAGE-2 FIXED OBSTACLE
        # ========================================================
        #
        # The straight segment from package (1.2, 0.0) to goal
        # (7.0, 5.0) passes approximately through (3.8, 2.2).
        #
        # Placing a 0.8 x 0.8 obstacle here forces a detour while
        # still leaving plenty of free space around it.
        # ========================================================

        self.stage2_obstacle_position = np.array(
            [
                3.8,
                2.2
            ],
            dtype=np.float32
        )

        # Stage 2A: easier offset obstacle.
        #
        # This position is close enough to the nominal route to
        # appear in LiDAR / proximity reward, but far enough that
        # the Stage-1 policy is not forced into collision on every
        # episode.
        self.stage2a_obstacle_position = np.array(
            [
                3.8,
                4.2
            ],
            dtype=np.float32
        )

        # Stage 2B:
        # chosen directly from the generalization probe boundary.
        self.stage2b_obstacle_position = np.array(
            [
                3.8,
                3.5
            ],
            dtype=np.float32
        )

        # Stage 2C narrow one-obstacle randomization.
        self.stage2c_obstacle_x = 3.8
        self.stage2c_y_min = 3.2
        self.stage2c_y_max = 3.5

        # Stage 2D: wider one-obstacle randomization.
        self.stage2d_obstacle_x = 3.8
        self.stage2d_y_min = 2.8
        self.stage2d_y_max = 3.5

        # Stage 2E: wider one-obstacle randomization.
        self.stage2e_obstacle_x = 3.8
        self.stage2e_y_min = 2.4
        self.stage2e_y_max = 3.5

        # ========================================================
        # STAGE 3: TWO RANDOMIZED OBSTACLES
        # ========================================================
        #
        # These are deliberately constrained bands rather than
        # full-map randomization. Stage 3 should teach the policy
        # to deal with TWO different obstacle encounters before
        # we move to four obstacles / full randomization.
        #
        # Obstacle 1: earlier portion of package -> goal route
        # Obstacle 2: later portion of package -> goal route
        # ========================================================

        self.stage3_obs1_x_min = 2.7
        self.stage3_obs1_x_max = 4.1
        self.stage3_obs1_y_min = 1.7
        self.stage3_obs1_y_max = 3.6

        self.stage3_obs2_x_min = 4.4
        self.stage3_obs2_x_max = 6.1
        self.stage3_obs2_y_min = 2.8
        self.stage3_obs2_y_max = 4.8

        self.stage3_min_obstacle_separation = 2.0
        self.stage3_max_layout_attempts = 250

        # ========================================================
        # STAGE 3A: EASY SECOND OBSTACLE
        # ========================================================

        # Obstacle 1: exactly the mastered Stage-2 style.
        self.stage3a_obs1_x = 3.8
        self.stage3a_obs1_y_min = 2.2
        self.stage3a_obs1_y_max = 3.5

        # Obstacle 2: lower/off-route but still within LiDAR range
        # during much of the package->goal traversal.
        self.stage3a_obs2_x_min = 4.8
        self.stage3a_obs2_x_max = 6.0

        # Corrected Stage-3A bridge range.
        #
        # Frozen Stage-2D diagnostic:
        #   y=-1.7..-1.0 -> 100% success
        #   y=-0.9..-0.2 -> 80% success
        #
        # Combined bridge range:
        #   y=-1.7..-0.2 -> 80% success / 20% collision
        #
        # This is intentionally learnable but non-trivial.
        self.stage3a_obs2_y_min = -1.7
        self.stage3a_obs2_y_max = -0.2

        self.stage3a_min_obstacle_separation = 2.0
        self.stage3a_max_layout_attempts = 200


        # ========================================================
        # ONE-OBSTACLE ROUTE-WIDE GENERALIZATION
        # ========================================================
        #
        # The obstacle is no longer tied to x=3.8.
        #
        # Distribution A (70%):
        #   sample different x positions along package -> goal,
        #   then place obstacle around the nominal corridor.
        #
        # Distribution B (30%):
        #   later/lower region identified by the O2 diagnostic.
        #
        # Exactly one physical obstacle remains active.
        # ========================================================

        self.routewide_x_min = 2.2
        self.routewide_x_max = 6.2

        self.routewide_corridor_offset_min = -1.4
        self.routewide_corridor_offset_max = 1.4

        self.routewide_lower_x_min = 4.5
        self.routewide_lower_x_max = 6.0

        self.routewide_lower_y_min = -1.7
        self.routewide_lower_y_max = 0.5

        self.routewide_corridor_probability = 0.70
        self.routewide_max_layout_attempts = 250

        # ========================================================
        # FINAL MULTI-RANDOMIZED OBSTACLE TASK
        # ========================================================
        #
        # Training mode "multi_random_routewide":
        #   15% of episodes -> 1 obstacle
        #   35% of episodes -> 2 obstacles
        #   50% of episodes -> 3 obstacles
        #
        # This is NOT a stage curriculum. Obstacle count and
        # positions are randomized from the beginning.
        #
        # Final evaluation/demo mode "three_random_routewide":
        #   exactly 3 randomized obstacles every episode.
        #
        # Obstacles are sampled in early/middle/late route bands.
        # This makes the robot encounter repeated local avoidance
        # situations while still varying x/y continuously.
        #
        # A* is used ONLY to reject impossible layouts. The SAC
        # policy never receives a path or planner output.
        # ========================================================

        self.multi_sections = [
            # x_min, x_max, corridor_offset_min, corridor_offset_max
            # Four broad route sections keep obstacles visually spread
            # across the whole delivery path.
            (2.15, 3.10, -1.30, 1.30),
            (3.35, 4.35, -1.30, 1.30),
            (4.60, 5.65, -1.30, 1.30),
            (5.95, 7.05, -1.30, 1.30),
        ]

        self.multi_min_obstacle_separation = 1.55
        self.multi_max_layout_attempts = 500

        # ========================================================
        # ROBOT
        # ========================================================

        self.WHEEL_RADIUS = 0.18
        self.TRACK_WIDTH = 0.92

        self.MAX_LINEAR_SPEED = 1.5
        self.MAX_ANGULAR_SPEED = 2.0

        # Keep original conservative radius for A* path safety.
        # It is no longer used for final collision declaration.
        self.ROBOT_RADIUS = 0.80

        # ========================================================
        # PRECISE COLLISION FOOTPRINT
        # ========================================================
        #
        # delivery_robot_showcase.urdf footprint is approximately:
        #   length = 1.20 m
        #   width  = 1.04 m including wheels
        #
        # Old collision used a 0.80 m CIRCLE around the robot.
        # That created false collisions near sides/corners.
        #
        # New collision uses a ROTATED RECTANGLE matching the robot.
        # ========================================================

        self.ROBOT_COLLISION_HALF_LENGTH = 0.60
        self.ROBOT_COLLISION_HALF_WIDTH = 0.52

        # Very small tolerance only for numerical contact.
        self.COLLISION_CONTACT_MARGIN = 0.015

        # ========================================================
        # TASK LOCATIONS
        # ========================================================

        self.start_position = np.array(
            [0.0, 0.0],
            dtype=np.float32
        )

        # Both delivery parcels are picked from this ONE yellow
        # pickup zone.
        self.package_position = np.array(
            [1.2, 0.0],
            dtype=np.float32
        )

        # --------------------------------------------------------
        # TWO CONTINUOUS DELIVERY STOPS
        # --------------------------------------------------------
        #
        # Delivery A preserves the existing showcase destination.
        #
        # Delivery B is farther ahead on the same upper corridor,
        # so after A the robot naturally continues forward instead
        # of resetting or returning to START.
        # --------------------------------------------------------

        self.goal_a_position = np.array(
            [8.0, 5.5],
            dtype=np.float32
        )

        # Delivery B is one full START->A route-length farther
        # along the same direction.
        #
        # START -> A = [8.0, 5.5]
        # A     -> B = [8.0, 5.5]
        #
        # Therefore both route sections have the same scale.
        self.goal_b_position = np.array(
            [16.0, 11.0],
            dtype=np.float32
        )

        # goal_position always means the CURRENT delivery target.
        # It starts at A and switches to B automatically after A.
        self.goal_position = (
            self.goal_a_position.copy()
        )

        self.pickup_distance = 0.8
        self.delivery_distance = 0.9

        # ========================================================
        # EPISODE
        # ========================================================

        self.max_steps = 1000
        self.current_step = 0

        self.ACTION_REPEAT = 4

        self.WORLD_LIMIT = 21.0

        # ========================================================
        # LIDAR
        # ========================================================

        self.NUM_LIDAR_RAYS = 16
        self.LIDAR_MAX_DISTANCE = 5.0
        self.LIDAR_MIN_DISTANCE = 0.05

        # ========================================================
        # ORIGINAL OBSTACLE POSITIONS
        # ========================================================

        # Eight placeholder positions.
        # In showcase_dual_delivery they are replaced by TRUE
        # randomized 4+4 leg layouts at every reset.
        self.base_obstacle_positions = [
            (2.0,  1.8),
            (3.2, -0.8),
            (5.0,  3.5),
            (6.4,  1.4),
            (9.8,  7.2),
            (11.3, 4.9),
            (13.2, 9.7),
            (14.6, 7.0),
        ]

        # ========================================================
        # TRUE FULL-MAP RANDOMIZATION
        # ========================================================
        #
        # Every obstacle can appear anywhere inside this usable
        # rectangle on every reset. There are NO fixed obstacle
        # zones anymore.
        #
        # Example:
        #   run 1: pedestrian may be on LEFT
        #   run 2: same pedestrian may be on RIGHT
        #   run 3: same pedestrian may be near CENTER
        #
        # Safety constraints still prevent impossible/congested
        # layouts around START, PICKUP and GOAL.
        # ========================================================

        self.showcase_random_x_min = 1.20
        self.showcase_random_x_max = 6.75

        self.showcase_random_y_min = -1.20
        self.showcase_random_y_max = 5.80

        self.showcase_min_separation = 1.65
        self.showcase_max_layout_attempts = 180
        self.showcase_per_obstacle_attempts = 120

        # ========================================================
        # FINAL TWO-LEG RANDOMIZATION
        # ========================================================
        #
        # Exactly three random obstacles are generated in:
        #   START/PICKUP -> DELIVERY A
        #
        # and exactly three in:
        #   DELIVERY A -> DELIVERY B
        #
        # The six resulting positions are SHUFFLED before being
        # assigned to URDF types. Therefore a pedestrian/car that
        # appears in leg 1 on one run may appear in leg 2 on the
        # next run.
        #
        # Within each leg, Y is fully randomized — left/right/up/
        # down placement is not fixed.
        # ========================================================

        # ========================================================
        # OPEN RANDOM OBSTACLE CORRIDORS
        # ========================================================
        #
        # Instead of sampling anywhere in a huge rectangle, each
        # obstacle is sampled somewhere along the appropriate route
        # segment with a randomized lateral offset.
        #
        # This gives:
        #   - real random left/right placement;
        #   - open visual spacing;
        #   - no objects directly on the gates;
        #   - no congested pile-up.
        # ========================================================

        self.dual_obstacles_per_leg = 4

        # Fraction along each route where obstacles may appear.
        # 0.0 = beginning of route, 1.0 = gate.
        self.dual_t_min = 0.20
        self.dual_t_max = 0.80

        # Random perpendicular offset from route centre line.
        self.dual_lateral_min = -3.0
        self.dual_lateral_max = 3.0

        self.dual_min_obstacle_separation = 2.15
        self.dual_landmark_clearance = 1.90

        self.dual_layout_attempts = 260
        self.dual_position_attempts = 180

        # ========================================================
        # REAL URDF OBSTACLE ASSETS
        # ========================================================
        #
        # Randomization, task logic, pickup/drop and controller are
        # unchanged. These four assets simply replace the solid
        # white Box obstacles used in the original showcase.
        # ========================================================

        # Two complete visual sets:
        #   pedestrian + car + barrier + parcels
        # repeated once for the A->B route.
        self.obstacle_sizes = [
            (0.55, 0.48, 1.40),   # pedestrian
            (1.55, 0.95, 0.76),   # delivery vehicle
            (0.38, 1.48, 0.92),   # road barrier
            (1.15, 0.88, 0.90),   # parcel stack

            (0.55, 0.48, 1.40),   # pedestrian
            (1.55, 0.95, 0.76),   # delivery vehicle
            (0.38, 1.48, 0.92),   # road barrier
            (1.15, 0.88, 0.90),   # parcel stack
        ]

        self.obstacle_asset_files = [
            "pedestrian_obstacle.urdf",
            "delivery_vehicle_obstacle.urdf",
            "road_barrier_obstacle.urdf",
            "parcel_stack_obstacle.urdf",

            "pedestrian_obstacle.urdf",
            "delivery_vehicle_obstacle.urdf",
            "road_barrier_obstacle.urdf",
            "parcel_stack_obstacle.urdf",
        ]

        self.obstacle_data = []

        # ========================================================
        # CURRICULUM
        # ========================================================

        self.curriculum_level = "mild"

        # mild:
        # obstacle moves +/- 0.5 m around original position

        self.mild_range = 0.5

        # medium:
        # obstacle moves +/- 1.2 m around original position

        self.medium_range = 1.2

        # full randomization bounds

        self.FULL_X_MIN = 1.8
        self.FULL_X_MAX = 6.7

        self.FULL_Y_MIN = -0.5
        self.FULL_Y_MAX = 5.3

        # clearance

        self.START_CLEARANCE = 1.5
        self.PACKAGE_CLEARANCE = 1.3
        self.GOAL_CLEARANCE = 1.3
        self.OBSTACLE_CLEARANCE = 1.4

        # ========================================================
        # SOLVABILITY FILTER
        # ========================================================
        #
        # A* is used ONLY when generating a random obstacle layout.
        # It checks whether a collision-free route exists:
        #
        #   START -> PACKAGE
        #   PACKAGE -> GOAL
        #
        # SAC never sees the A* route and never uses A* for control.
        # The robot still navigates only from its observations.
        # ========================================================

        self.PATH_GRID_RESOLUTION = 0.10

        self.PATH_EXTRA_SAFETY_MARGIN = 0.05

        # Maximum number of complete random layouts to try
        # before falling back to the known base layout.
        self.MAX_LAYOUT_GENERATION_ATTEMPTS = 200

        # Number of placement attempts for each obstacle
        # while building one candidate layout.
        self.MAX_OBSTACLE_PLACEMENT_ATTEMPTS = 1000

        # Diagnostic value: how many whole-layout attempts
        # were required on the most recent reset.
        self.last_layout_generation_attempts = 0

        # ========================================================
        # ACTION SPACE
        # ========================================================

        self.action_space = spaces.Box(

            low=np.array(
                [-1.0, -1.0],
                dtype=np.float32
            ),

            high=np.array(
                [1.0, 1.0],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        # ========================================================
        # OBSERVATION SPACE
        # ========================================================

        lidar_low = np.zeros(
            self.NUM_LIDAR_RAYS,
            dtype=np.float32
        )

        lidar_high = np.ones(
            self.NUM_LIDAR_RAYS,
            dtype=np.float32
        )

        extra_low = np.array(
            [
                0.0,
                -np.pi,
                -self.MAX_LINEAR_SPEED,
                -self.MAX_ANGULAR_SPEED,
                0.0
            ],
            dtype=np.float32
        )

        extra_high = np.array(
            [
                20.0,
                np.pi,
                self.MAX_LINEAR_SPEED,
                self.MAX_ANGULAR_SPEED,
                1.0
            ],
            dtype=np.float32
        )

        self.observation_space = spaces.Box(

            low=np.concatenate(
                [lidar_low, extra_low]
            ),

            high=np.concatenate(
                [lidar_high, extra_high]
            ),

            dtype=np.float32
        )

        # ========================================================
        # STATE
        # ========================================================

        self.package_picked = False

        # package_delivered is TRUE only after DELIVERY B.
        self.package_delivered = False

        self.just_picked_package = False

        self.delivery_a_complete = False
        self.delivery_b_complete = False
        self.just_delivered_a = False

        self.collision = False
        self.out_of_bounds = False

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.pickup_count = 0
        self.delivery_count = 0

        self._create_scene()

    # ============================================================
    # CURRICULUM SETTER
    # ============================================================

    def set_curriculum_level(self, level):

        if level not in [
            "mild",
            "medium",
            "full"
        ]:

            raise ValueError(
                "Curriculum level must be "
                "'mild', 'medium', or 'full'"
            )

        if level != self.curriculum_level:

            self.curriculum_level = level

            print()
            print("=" * 70)
            print(
                "CURRICULUM CHANGED TO:",
                level.upper()
            )
            print("=" * 70)

    # ============================================================
    # SCENE
    # ============================================================

    def _create_scene(self):

        self.scene = gs.Scene(

            show_viewer=self.render_enabled,

            sim_options=gs.options.SimOptions(
                dt=0.01
            ),

            viewer_options=gs.options.ViewerOptions(

                res=(1000, 700),

                # Closer initial camera.
                # You can still freely orbit / pan / zoom with the
                # mouse while the robot is running.
                camera_pos=(
                    18.0,
                    -17.5,
                    13.0
                ),

                camera_lookat=(
                    8.0,
                    5.5,
                    0.4
                ),

                camera_fov=45,

                # Keep Genesis interactive camera controls enabled.
                enable_help_text=True,
                enable_default_keybinds=True,

                refresh_rate=60
            )
        )

        self.scene.add_entity(
            gs.morphs.Plane()
        )

        self.robot = self.scene.add_entity(

            gs.morphs.URDF(

                file="delivery_robot_showcase.urdf",

                pos=(
                    self.start_position[0],
                    self.start_position[1],
                    0.0
                )
            )
        )

        # HOME is intentionally not created as a physical box.
        # The start position is represented logically by self.start_position.

        # ========================================================
        # TWO DELIVERY PARCELS AT ONE PICKUP POINT
        # ========================================================

        self.package = self.scene.add_entity(
            gs.morphs.Box(
                size=(
                    0.32,
                    0.32,
                    0.32
                ),
                pos=(
                    self.package_position[0],
                    self.package_position[1] - 0.22,
                    0.18
                )
            )
        )

        self.package_b = self.scene.add_entity(
            gs.morphs.Box(
                size=(
                    0.32,
                    0.32,
                    0.32
                ),
                pos=(
                    self.package_position[0],
                    self.package_position[1] + 0.22,
                    0.18
                )
            )
        )

        # ========================================================
        # MODERN SHOWCASE INTERFACE
        # ========================================================
        #
        # Visual-only START/GOAL arches and floor pads. They have
        # NO collision geometry, so they cannot disturb the working
        # SAC+A* navigation.
        # ========================================================

        self.start_gate = self.scene.add_entity(
            gs.morphs.URDF(
                file="modern_start_gate.urdf",
                pos=(-0.72, 0.0, 0.0),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        self.start_pad = self.scene.add_entity(
            gs.morphs.URDF(
                file="showcase_start_pad.urdf",
                pos=(
                    float(self.start_position[0]),
                    float(self.start_position[1]),
                    0.0
                ),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        # Yellow pickup-zone outline below the physical package.
        self.pickup_pad = self.scene.add_entity(
            gs.morphs.URDF(
                file="showcase_pickup_pad.urdf",
                pos=(
                    float(self.package_position[0]),
                    float(self.package_position[1]),
                    0.0
                ),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        goal_x = float(self.goal_position[0])
        goal_y = float(self.goal_position[1])

        # Keep decorative goal arch slightly beyond the delivery
        # point and completely non-colliding.
        goal_gate_x = min(
            goal_x + 0.48,
            self.WORLD_LIMIT - 0.20
        )

        self.goal_gate = self.scene.add_entity(
            gs.morphs.URDF(
                file="modern_goal_gate.urdf",
                pos=(
                    goal_gate_x,
                    goal_y,
                    0.0
                ),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        # DELIVERY A pad/gate.
        self.goal_pad = self.scene.add_entity(
            gs.morphs.URDF(
                file="showcase_goal_pad.urdf",
                pos=(
                    goal_x,
                    goal_y,
                    0.0
                ),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        # --------------------------------------------------------
        # DELIVERY B — EXACT SAME GATE/PAD STYLE
        # --------------------------------------------------------

        goal_b_x = float(
            self.goal_b_position[0]
        )

        goal_b_y = float(
            self.goal_b_position[1]
        )

        goal_b_gate_x = min(
            goal_b_x + 0.48,
            self.WORLD_LIMIT - 0.20
        )

        self.goal_b_gate = self.scene.add_entity(
            gs.morphs.URDF(
                file="modern_goal_gate.urdf",
                pos=(
                    goal_b_gate_x,
                    goal_b_y,
                    0.0
                ),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        self.goal_b_pad = self.scene.add_entity(
            gs.morphs.URDF(
                file="showcase_goal_pad.urdf",
                pos=(
                    goal_b_x,
                    goal_b_y,
                    0.0
                ),
                fixed=True,
                prioritize_urdf_material=True
            )
        )

        self.obstacles = []

        for i, asset_file in enumerate(
            self.obstacle_asset_files
        ):

            x, y = (
                self.base_obstacle_positions[i]
            )

            obstacle = self.scene.add_entity(
                gs.morphs.URDF(
                    file=asset_file,
                    pos=(
                        float(x),
                        float(y),
                        0.0
                    ),
                    fixed=True,
                    prioritize_urdf_material=True
                )
            )

            self.obstacles.append(
                obstacle
            )

            sx, sy, sz = (
                self.obstacle_sizes[i]
            )

            self.obstacle_data.append(
                [
                    float(x),
                    float(y),
                    float(sx),
                    float(sy),
                    float(sz)
                ]
            )

        self.scene.build()

        fl = self.robot.get_joint(
            "front_left_wheel_joint"
        )

        fr = self.robot.get_joint(
            "front_right_wheel_joint"
        )

        rl = self.robot.get_joint(
            "rear_left_wheel_joint"
        )

        rr = self.robot.get_joint(
            "rear_right_wheel_joint"
        )

        self.wheel_dofs = [
            fl.dofs_idx_local[0],
            rl.dofs_idx_local[0],
            fr.dofs_idx_local[0],
            rr.dofs_idx_local[0]
        ]

        print(
            "Wheel DOFs:",
            self.wheel_dofs
        )

    # ============================================================
    # STAGE-0 / STAGE-1 OBSTACLE DISABLE
    # ============================================================

    def _disable_obstacles_for_goal_only(self):
        """
        Move all obstacle entities far outside the training world.

        Used by goal_only and package_goal modes.

        The obstacle records are also updated so LiDAR and the
        custom collision checker see no nearby obstacles.
        """

        far_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0),
            (33.0, 33.0)
        ]

        disabled_layout = []

        for i, (
            sx,
            sy,
            sz
        ) in enumerate(
            self.obstacle_sizes
        ):

            x, y = (
                far_positions[i]
            )

            self.obstacles[i].set_pos(
                np.array(
                    [
                        x,
                        y,
                        0.0
                    ],
                    dtype=np.float32
                )
            )

            disabled_layout.append(
                [
                    float(x),
                    float(y),
                    float(sx),
                    float(sy),
                    float(sz)
                ]
            )

        self.obstacle_data = (
            disabled_layout
        )

        self.last_layout_generation_attempts = 0


    # ============================================================
    # STAGE-2 ONE FIXED OBSTACLE
    # ============================================================

    def _set_stage2_one_fixed_obstacle(self):
        """
        Keep only obstacle 1 inside the world.

        It is placed directly across the nominal package->goal
        straight-line route. All remaining obstacles are moved
        far outside the world.

        SAC is NOT told where the obstacle is. It only perceives
        it through LiDAR.
        """

        fixed_x = float(
            self.stage2_obstacle_position[0]
        )

        fixed_y = float(
            self.stage2_obstacle_position[1]
        )

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        layout = []

        # --------------------------------------------------------
        # ACTIVE OBSTACLE
        # --------------------------------------------------------

        sx, sy, sz = (
            self.obstacle_sizes[0]
        )

        self.obstacles[0].set_pos(
            np.array(
                [
                    fixed_x,
                    fixed_y,
                    0.0
                ],
                dtype=np.float32
            )
        )

        layout.append(
            [
                fixed_x,
                fixed_y,
                float(sx),
                float(sy),
                float(sz)
            ]
        )

        # --------------------------------------------------------
        # DISABLE REMAINING 3
        # --------------------------------------------------------

        for i in range(
            1,
            len(self.obstacles)
        ):

            x, y = (
                disabled_positions[i - 1]
            )

            osx, osy, osz = (
                self.obstacle_sizes[i]
            )

            self.obstacles[i].set_pos(
                np.array(
                    [
                        x,
                        y,
                        0.0
                    ],
                    dtype=np.float32
                )
            )

            layout.append(
                [
                    float(x),
                    float(y),
                    float(osx),
                    float(osy),
                    float(osz)
                ]
            )

        self.obstacle_data = layout

        self.last_layout_generation_attempts = 0


    # ============================================================
    # STAGE-2A ONE OFFSET OBSTACLE
    # ============================================================

    def _set_stage2a_one_offset_obstacle(self):
        """
        Keep one obstacle inside the world, offset from the
        package->goal straight route.

        The obstacle remains close enough to influence LiDAR and
        the proximity reward, but does not completely block the
        inherited Stage-1 path.

        SAC is never given the obstacle coordinates.
        """

        fixed_x = float(
            self.stage2a_obstacle_position[0]
        )

        fixed_y = float(
            self.stage2a_obstacle_position[1]
        )

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        layout = []

        sx, sy, sz = (
            self.obstacle_sizes[0]
        )

        self.obstacles[0].set_pos(
            np.array(
                [
                    fixed_x,
                    fixed_y,
                    0.0
                ],
                dtype=np.float32
            )
        )

        layout.append(
            [
                fixed_x,
                fixed_y,
                float(sx),
                float(sy),
                float(sz)
            ]
        )

        for i in range(
            1,
            len(self.obstacles)
        ):

            x, y = (
                disabled_positions[i - 1]
            )

            osx, osy, osz = (
                self.obstacle_sizes[i]
            )

            self.obstacles[i].set_pos(
                np.array(
                    [
                        x,
                        y,
                        0.0
                    ],
                    dtype=np.float32
                )
            )

            layout.append(
                [
                    float(x),
                    float(y),
                    float(osx),
                    float(osy),
                    float(osz)
                ]
            )

        self.obstacle_data = layout

        self.last_layout_generation_attempts = 0


    # ============================================================
    # STAGE-2B ONE MID-DIFFICULTY OBSTACLE
    # ============================================================

    def _set_stage2b_one_mid_obstacle(self):
        """
        Keep one obstacle inside the world at (3.8, 3.5).

        This position was chosen from the Stage-2A frozen-policy
        probe:
            y = 3.6 -> success
            y = 3.4 -> collision

        The other three obstacles remain disabled.
        """

        fixed_x = float(
            self.stage2b_obstacle_position[0]
        )

        fixed_y = float(
            self.stage2b_obstacle_position[1]
        )

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        layout = []

        sx, sy, sz = (
            self.obstacle_sizes[0]
        )

        self.obstacles[0].set_pos(
            np.array(
                [
                    fixed_x,
                    fixed_y,
                    0.0
                ],
                dtype=np.float32
            )
        )

        layout.append(
            [
                fixed_x,
                fixed_y,
                float(sx),
                float(sy),
                float(sz)
            ]
        )

        for i in range(
            1,
            len(self.obstacles)
        ):

            x, y = (
                disabled_positions[i - 1]
            )

            osx, osy, osz = (
                self.obstacle_sizes[i]
            )

            self.obstacles[i].set_pos(
                np.array(
                    [
                        x,
                        y,
                        0.0
                    ],
                    dtype=np.float32
                )
            )

            layout.append(
                [
                    float(x),
                    float(y),
                    float(osx),
                    float(osy),
                    float(osz)
                ]
            )

        self.obstacle_data = layout

        self.last_layout_generation_attempts = 0


    # ============================================================
    # STAGE-2C ONE NARROW-RANDOM OBSTACLE
    # ============================================================

    def _set_stage2c_one_random_narrow_obstacle(self):
        """
        Randomize one active obstacle every reset.

        x is fixed at 3.8.
        y is sampled uniformly from [3.2, 3.5].

        The other three obstacles are moved outside the world.
        Every candidate is checked for route solvability.
        """

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        accepted_layout = None

        for attempt in range(1, 101):

            fixed_x = float(
                self.stage2c_obstacle_x
            )

            fixed_y = float(
                self.np_random.uniform(
                    self.stage2c_y_min,
                    self.stage2c_y_max
                )
            )

            layout = []

            sx, sy, sz = (
                self.obstacle_sizes[0]
            )

            layout.append(
                [
                    fixed_x,
                    fixed_y,
                    float(sx),
                    float(sy),
                    float(sz)
                ]
            )

            for i in range(
                1,
                len(self.obstacles)
            ):

                x, y = disabled_positions[
                    i - 1
                ]

                osx, osy, osz = (
                    self.obstacle_sizes[i]
                )

                layout.append(
                    [
                        float(x),
                        float(y),
                        float(osx),
                        float(osy),
                        float(osz)
                    ]
                )

            if self._layout_is_solvable(
                layout
            ):

                accepted_layout = layout
                self.last_layout_generation_attempts = attempt
                break

        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable "
                "Stage-2C layout."
            )

        self._apply_obstacle_layout(
            accepted_layout
        )


    # ============================================================
    # STAGE-2D ONE WIDE-RANDOM OBSTACLE
    # ============================================================

    def _set_stage2d_one_random_wide_obstacle(self):
        """
        Randomize one active obstacle every reset.

        x is fixed at 3.8.
        y is sampled uniformly from [2.8, 3.5].

        The other three obstacles are outside the world.
        Every sampled layout is checked for solvability.
        """

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        accepted_layout = None

        for attempt in range(
            1,
            101
        ):

            fixed_x = float(
                self.stage2d_obstacle_x
            )

            fixed_y = float(
                self.np_random.uniform(
                    self.stage2d_y_min,
                    self.stage2d_y_max
                )
            )

            layout = []

            sx, sy, sz = (
                self.obstacle_sizes[0]
            )

            layout.append(
                [
                    fixed_x,
                    fixed_y,
                    float(sx),
                    float(sy),
                    float(sz)
                ]
            )

            for i in range(
                1,
                len(self.obstacles)
            ):

                x, y = (
                    disabled_positions[i - 1]
                )

                osx, osy, osz = (
                    self.obstacle_sizes[i]
                )

                layout.append(
                    [
                        float(x),
                        float(y),
                        float(osx),
                        float(osy),
                        float(osz)
                    ]
                )

            if self._layout_is_solvable(
                layout
            ):

                accepted_layout = layout
                self.last_layout_generation_attempts = attempt
                break

        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable "
                "Stage-2D layout."
            )

        self._apply_obstacle_layout(
            accepted_layout
        )


    # ============================================================
    # STAGE-2E ONE WIDER-RANDOM OBSTACLE
    # ============================================================

    def _set_stage2e_one_random_wider_obstacle(self):
        """
        Randomize one active obstacle every reset.

        x is fixed at 3.8.
        y is sampled uniformly from [2.4, 3.5].

        The other three obstacles remain outside the world.
        Every sampled candidate is checked for route solvability.
        """

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        accepted_layout = None

        for attempt in range(
            1,
            101
        ):

            fixed_x = float(
                self.stage2e_obstacle_x
            )

            fixed_y = float(
                self.np_random.uniform(
                    self.stage2e_y_min,
                    self.stage2e_y_max
                )
            )

            layout = []

            sx, sy, sz = (
                self.obstacle_sizes[0]
            )

            layout.append(
                [
                    fixed_x,
                    fixed_y,
                    float(sx),
                    float(sy),
                    float(sz)
                ]
            )

            for i in range(
                1,
                len(self.obstacles)
            ):

                x, y = (
                    disabled_positions[i - 1]
                )

                osx, osy, osz = (
                    self.obstacle_sizes[i]
                )

                layout.append(
                    [
                        float(x),
                        float(y),
                        float(osx),
                        float(osy),
                        float(osz)
                    ]
                )

            if self._layout_is_solvable(
                layout
            ):

                accepted_layout = layout
                self.last_layout_generation_attempts = attempt
                break

        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable "
                "Stage-2E layout."
            )

        self._apply_obstacle_layout(
            accepted_layout
        )


    # ============================================================
    # STAGE-3 TWO RANDOMIZED OBSTACLES
    # ============================================================

    def _set_stage3_two_random_obstacles(self):
        """
        Randomize TWO active obstacles every reset.

        Obstacle 1 is sampled in an earlier package->goal band.
        Obstacle 2 is sampled in a later package->goal band.

        The remaining two obstacles are disabled outside the world.

        Every complete layout must:
          - keep obstacle centres sufficiently separated;
          - remain clear of package and goal;
          - be physically solvable according to the existing
            inflated A* route checker.

        A* is never exposed to SAC.
        """

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0)
        ]

        accepted_layout = None


        for attempt in range(
            1,
            self.stage3_max_layout_attempts + 1
        ):

            # ----------------------------------------------------
            # SAMPLE ACTIVE OBSTACLE 1
            # ----------------------------------------------------

            x1 = float(
                self.np_random.uniform(
                    self.stage3_obs1_x_min,
                    self.stage3_obs1_x_max
                )
            )

            y1 = float(
                self.np_random.uniform(
                    self.stage3_obs1_y_min,
                    self.stage3_obs1_y_max
                )
            )


            # ----------------------------------------------------
            # SAMPLE ACTIVE OBSTACLE 2
            # ----------------------------------------------------

            x2 = float(
                self.np_random.uniform(
                    self.stage3_obs2_x_min,
                    self.stage3_obs2_x_max
                )
            )

            y2 = float(
                self.np_random.uniform(
                    self.stage3_obs2_y_min,
                    self.stage3_obs2_y_max
                )
            )


            p1 = np.array(
                [x1, y1],
                dtype=np.float32
            )

            p2 = np.array(
                [x2, y2],
                dtype=np.float32
            )


            # ----------------------------------------------------
            # KEEP ACTIVE OBSTACLES SEPARATED
            # ----------------------------------------------------

            if (
                np.linalg.norm(
                    p1 - p2
                )
                <
                self.stage3_min_obstacle_separation
            ):

                continue


            # ----------------------------------------------------
            # KEEP CLEAR OF PACKAGE / GOAL
            # ----------------------------------------------------

            locally_valid = True

            for candidate in [
                p1,
                p2
            ]:

                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.package_position
                    )
                    <
                    self.PACKAGE_CLEARANCE
                ):

                    locally_valid = False
                    break

                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.goal_position
                    )
                    <
                    self.GOAL_CLEARANCE
                ):

                    locally_valid = False
                    break


            if not locally_valid:

                continue


            # ----------------------------------------------------
            # BUILD COMPLETE 4-OBSTACLE RECORD
            # ----------------------------------------------------

            layout = []

            sx1, sy1, sz1 = (
                self.obstacle_sizes[0]
            )

            layout.append(
                [
                    x1,
                    y1,
                    float(sx1),
                    float(sy1),
                    float(sz1)
                ]
            )


            sx2, sy2, sz2 = (
                self.obstacle_sizes[1]
            )

            layout.append(
                [
                    x2,
                    y2,
                    float(sx2),
                    float(sy2),
                    float(sz2)
                ]
            )


            # Disable obstacles 3 and 4.
            for local_index, obstacle_index in enumerate(
                [2, 3]
            ):

                x, y = (
                    disabled_positions[
                        local_index
                    ]
                )

                sx, sy, sz = (
                    self.obstacle_sizes[
                        obstacle_index
                    ]
                )

                layout.append(
                    [
                        float(x),
                        float(y),
                        float(sx),
                        float(sy),
                        float(sz)
                    ]
                )


            # ----------------------------------------------------
            # GLOBAL SOLVABILITY FILTER
            # ----------------------------------------------------

            if not self._layout_is_solvable(
                layout
            ):

                continue


            accepted_layout = layout
            self.last_layout_generation_attempts = attempt
            break


        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable Stage-3 "
                "two-obstacle layout."
            )


        self._apply_obstacle_layout(
            accepted_layout
        )


    # ============================================================
    # STAGE-3A TWO OBSTACLES, SECOND ONE EASY/OFF-ROUTE
    # ============================================================

    def _set_stage3a_two_random_easy_obstacles(self):
        """
        Stage-3A bridge curriculum.

        Active obstacle 1:
            x = 3.8
            y ~ U(2.2, 3.5)

        Active obstacle 2:
            x ~ U(4.8, 6.0)
            y ~ U(0.8, 1.6)

        Obstacles 3 and 4 are disabled.

        Every layout is checked for:
          - package/goal clearance,
          - obstacle separation,
          - global solvability.

        SAC never receives obstacle coordinates directly.
        """

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0)
        ]

        accepted_layout = None


        for attempt in range(
            1,
            self.stage3a_max_layout_attempts + 1
        ):

            # ----------------------------------------------------
            # OBSTACLE 1: MASTERED SINGLE-OBSTACLE DISTRIBUTION
            # ----------------------------------------------------

            x1 = float(
                self.stage3a_obs1_x
            )

            y1 = float(
                self.np_random.uniform(
                    self.stage3a_obs1_y_min,
                    self.stage3a_obs1_y_max
                )
            )


            # ----------------------------------------------------
            # OBSTACLE 2: EASY SECOND OBSTACLE
            # ----------------------------------------------------

            x2 = float(
                self.np_random.uniform(
                    self.stage3a_obs2_x_min,
                    self.stage3a_obs2_x_max
                )
            )

            y2 = float(
                self.np_random.uniform(
                    self.stage3a_obs2_y_min,
                    self.stage3a_obs2_y_max
                )
            )


            p1 = np.array(
                [x1, y1],
                dtype=np.float32
            )

            p2 = np.array(
                [x2, y2],
                dtype=np.float32
            )


            # ----------------------------------------------------
            # SEPARATION
            # ----------------------------------------------------

            if (
                np.linalg.norm(
                    p1 - p2
                )
                <
                self.stage3a_min_obstacle_separation
            ):

                continue


            # ----------------------------------------------------
            # PACKAGE / GOAL CLEARANCE
            # ----------------------------------------------------

            locally_valid = True

            for candidate in [
                p1,
                p2
            ]:

                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.package_position
                    )
                    <
                    self.PACKAGE_CLEARANCE
                ):

                    locally_valid = False
                    break

                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.goal_position
                    )
                    <
                    self.GOAL_CLEARANCE
                ):

                    locally_valid = False
                    break


            if not locally_valid:

                continue


            # ----------------------------------------------------
            # COMPLETE LAYOUT
            # ----------------------------------------------------

            layout = []


            sx1, sy1, sz1 = (
                self.obstacle_sizes[0]
            )

            layout.append(
                [
                    x1,
                    y1,
                    float(sx1),
                    float(sy1),
                    float(sz1)
                ]
            )


            sx2, sy2, sz2 = (
                self.obstacle_sizes[1]
            )

            layout.append(
                [
                    x2,
                    y2,
                    float(sx2),
                    float(sy2),
                    float(sz2)
                ]
            )


            for local_index, obstacle_index in enumerate(
                [2, 3]
            ):

                x, y = (
                    disabled_positions[
                        local_index
                    ]
                )

                sx, sy, sz = (
                    self.obstacle_sizes[
                        obstacle_index
                    ]
                )

                layout.append(
                    [
                        float(x),
                        float(y),
                        float(sx),
                        float(sy),
                        float(sz)
                    ]
                )


            # ----------------------------------------------------
            # SOLVABILITY
            # ----------------------------------------------------

            if not self._layout_is_solvable(
                layout
            ):

                continue


            accepted_layout = layout
            self.last_layout_generation_attempts = attempt
            break


        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable Stage-3A layout."
            )


        self._apply_obstacle_layout(
            accepted_layout
        )


    # ============================================================
    # ONE RANDOM OBSTACLE ACROSS THE ROUTE
    # ============================================================

    def _set_one_random_routewide_obstacle(self):
        """
        Exactly ONE active obstacle per episode.

        This task is designed to test/teach location generalization,
        not multi-obstacle navigation.

        70% of samples:
            obstacle x varies across the route and y is sampled
            around the nominal package->goal corridor.

        30% of samples:
            obstacle is sampled in the later/lower band where the
            O2-only diagnostic showed weaker generalization.

        All layouts must satisfy the existing local clearances and
        the existing A* solvability check.

        A* is used only for validation and is never observed by SAC.
        """

        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0)
        ]

        accepted_layout = None

        package_xy = np.asarray(
            self.package_position,
            dtype=np.float32
        )

        goal_xy = np.asarray(
            self.goal_position,
            dtype=np.float32
        )

        route_dx = float(
            goal_xy[0] - package_xy[0]
        )

        route_dy = float(
            goal_xy[1] - package_xy[1]
        )


        for attempt in range(
            1,
            self.routewide_max_layout_attempts + 1
        ):

            use_corridor = bool(
                self.np_random.random()
                <
                self.routewide_corridor_probability
            )


            # ----------------------------------------------------
            # A. BROAD PACKAGE->GOAL CORRIDOR
            # ----------------------------------------------------

            if use_corridor:

                x = float(
                    self.np_random.uniform(
                        self.routewide_x_min,
                        self.routewide_x_max
                    )
                )

                route_fraction = (
                    x - float(package_xy[0])
                ) / route_dx

                nominal_y = (
                    float(package_xy[1])
                    +
                    route_fraction
                    *
                    route_dy
                )

                lateral_offset = float(
                    self.np_random.uniform(
                        self.routewide_corridor_offset_min,
                        self.routewide_corridor_offset_max
                    )
                )

                y = (
                    nominal_y
                    +
                    lateral_offset
                )


            # ----------------------------------------------------
            # B. LATER / LOWER GENERALIZATION BAND
            # ----------------------------------------------------

            else:

                x = float(
                    self.np_random.uniform(
                        self.routewide_lower_x_min,
                        self.routewide_lower_x_max
                    )
                )

                y = float(
                    self.np_random.uniform(
                        self.routewide_lower_y_min,
                        self.routewide_lower_y_max
                    )
                )


            candidate = np.array(
                [x, y],
                dtype=np.float32
            )


            # ----------------------------------------------------
            # LOCAL CLEARANCE
            # ----------------------------------------------------

            if (
                np.linalg.norm(
                    candidate - self.start_position
                )
                <
                self.START_CLEARANCE
            ):
                continue


            if (
                np.linalg.norm(
                    candidate - self.package_position
                )
                <
                self.PACKAGE_CLEARANCE
            ):
                continue


            if (
                np.linalg.norm(
                    candidate - self.goal_position
                )
                <
                self.GOAL_CLEARANCE
            ):
                continue


            # ----------------------------------------------------
            # BUILD FULL FOUR-OBSTACLE RECORD
            # ----------------------------------------------------

            sx, sy, sz = self.obstacle_sizes[0]

            layout = [
                [
                    x,
                    y,
                    float(sx),
                    float(sy),
                    float(sz)
                ]
            ]


            for local_index, obstacle_index in enumerate(
                [1, 2, 3]
            ):

                dx, dy = disabled_positions[
                    local_index
                ]

                osx, osy, osz = self.obstacle_sizes[
                    obstacle_index
                ]

                layout.append(
                    [
                        float(dx),
                        float(dy),
                        float(osx),
                        float(osy),
                        float(osz)
                    ]
                )


            # ----------------------------------------------------
            # WHOLE-TASK SOLVABILITY
            # ----------------------------------------------------

            if not self._layout_is_solvable(
                layout
            ):
                continue


            accepted_layout = layout
            self.last_layout_generation_attempts = attempt

            # Useful diagnostic label only; not part of observation.
            self.routewide_last_sample_type = (
                "corridor"
                if use_corridor
                else "lower"
            )

            break


        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable "
                "one_random_routewide layout."
            )


        self._apply_obstacle_layout(
            accepted_layout
        )



    # ============================================================
    # MULTI RANDOM OBSTACLES ACROSS THE ROUTE
    # ============================================================

    def _set_multi_random_routewide_obstacles(
        self,
        force_count=None
    ):
        """
        Generate 1, 2 or 3 randomized obstacles distributed across
        early/middle/late portions of PACKAGE -> GOAL.

        multi_random_routewide:
            randomly chooses 1, 2 or 3 obstacles every episode.

        three_random_routewide:
            exactly 3 obstacles every episode.

        four_random_routewide:
            exactly 4 obstacles every episode.

        All active obstacle x/y positions are continuously randomized.
        Layouts must pass local-clearance, inter-obstacle-separation,
        and whole-task A* solvability checks.

        The A* result is discarded; SAC receives only its normal
        LiDAR + target observations.
        """

        if force_count is None:

            r = float(
                self.np_random.random()
            )

            if r < 0.15:
                active_count = 1

            elif r < 0.50:
                active_count = 2

            else:
                active_count = 3

        else:
            active_count = int(
                force_count
            )


        if active_count not in [
            1,
            2,
            3,
            4
        ]:
            raise ValueError(
                "active_count must be 1, 2, 3, or 4"
            )


        package_xy = np.asarray(
            self.package_position,
            dtype=np.float32
        )

        goal_xy = np.asarray(
            self.goal_position,
            dtype=np.float32
        )

        route_dx = float(
            goal_xy[0] - package_xy[0]
        )

        route_dy = float(
            goal_xy[1] - package_xy[1]
        )


        disabled_positions = [
            (30.0, 30.0),
            (33.0, 30.0),
            (30.0, 33.0),
            (33.0, 33.0),
        ]


        accepted_layout = None


        for attempt in range(
            1,
            self.multi_max_layout_attempts + 1
        ):

            # ----------------------------------------------------
            # CHOOSE ROUTE SECTIONS
            # ----------------------------------------------------
            #
            # With three obstacles, use all three route sections.
            # With one/two, sample sections randomly so the policy
            # does not associate obstacle index with a fixed place.
            # ----------------------------------------------------

            if active_count == len(
                self.multi_sections
            ):

                section_indices = list(
                    range(
                        len(
                            self.multi_sections
                        )
                    )
                )

            else:

                section_indices = (
                    self.np_random.choice(
                        len(
                            self.multi_sections
                        ),
                        size=active_count,
                        replace=False
                    )
                    .tolist()
                )

                section_indices.sort()


            active_records = []
            failed_local = False


            for local_i, section_index in enumerate(
                section_indices
            ):

                (
                    x_min,
                    x_max,
                    offset_min,
                    offset_max
                ) = self.multi_sections[
                    section_index
                ]


                x = float(
                    self.np_random.uniform(
                        x_min,
                        x_max
                    )
                )


                route_fraction = (
                    x
                    -
                    float(package_xy[0])
                ) / route_dx


                nominal_y = (
                    float(package_xy[1])
                    +
                    route_fraction
                    *
                    route_dy
                )


                # ------------------------------------------------
                # FINAL SHOWCASE ZIG-ZAG PLACEMENT
                # ------------------------------------------------
                #
                # With exactly 4 obstacles, deliberately alternate
                # left/right around the nominal PACKAGE -> GOAL
                # corridor. X is still randomized inside four
                # separated route bands, so obstacles are also
                # staggered forward/back along the journey.
                #
                # Example:
                #
                #      LEFT      RIGHT      LEFT      RIGHT
                #        O          O         O          O
                #
                # The whole pattern flips randomly each episode.
                # ------------------------------------------------

                if active_count == 4:

                    if local_i == 0:
                        self._showcase_side_sign = (
                            1.0
                            if float(self.np_random.random()) < 0.5
                            else -1.0
                        )

                    side_sign = (
                        self._showcase_side_sign
                        if local_i % 2 == 0
                        else -self._showcase_side_sign
                    )

                    lateral_offset = (
                        side_sign
                        *
                        float(
                            self.np_random.uniform(
                                0.55,
                                1.00
                            )
                        )
                    )

                else:

                    lateral_offset = float(
                        self.np_random.uniform(
                            offset_min,
                            offset_max
                        )
                    )


                y = (
                    nominal_y
                    +
                    lateral_offset
                )


                candidate = np.array(
                    [
                        x,
                        y
                    ],
                    dtype=np.float32
                )


                # ------------------------------------------------
                # START / PACKAGE / GOAL CLEARANCE
                # ------------------------------------------------

                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.start_position
                    )
                    <
                    self.START_CLEARANCE
                ):
                    failed_local = True
                    break


                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.package_position
                    )
                    <
                    self.PACKAGE_CLEARANCE
                ):
                    failed_local = True
                    break


                if (
                    np.linalg.norm(
                        candidate
                        -
                        self.goal_position
                    )
                    <
                    self.GOAL_CLEARANCE
                ):
                    failed_local = True
                    break


                # ------------------------------------------------
                # INTER-OBSTACLE SEPARATION
                # ------------------------------------------------

                for existing in active_records:

                    existing_xy = np.array(
                        [
                            existing[0],
                            existing[1]
                        ],
                        dtype=np.float32
                    )

                    if (
                        np.linalg.norm(
                            candidate
                            -
                            existing_xy
                        )
                        <
                        self.multi_min_obstacle_separation
                    ):
                        failed_local = True
                        break


                if failed_local:
                    break


                sx, sy, sz = (
                    self.obstacle_sizes[
                        local_i
                    ]
                )


                active_records.append(
                    [
                        float(x),
                        float(y),
                        float(sx),
                        float(sy),
                        float(sz)
                    ]
                )


            if failed_local:
                continue


            # ----------------------------------------------------
            # BUILD FULL FOUR-PHYSICAL-OBSTACLE LAYOUT
            # ----------------------------------------------------

            layout = [
                record.copy()
                for record in active_records
            ]


            while len(layout) < 4:

                obstacle_index = len(layout)

                dx, dy = disabled_positions[
                    obstacle_index
                ]

                sx, sy, sz = (
                    self.obstacle_sizes[
                        obstacle_index
                    ]
                )

                layout.append(
                    [
                        float(dx),
                        float(dy),
                        float(sx),
                        float(sy),
                        float(sz)
                    ]
                )


            # ----------------------------------------------------
            # WHOLE TASK MUST BE SOLVABLE
            # ----------------------------------------------------

            if not self._layout_is_solvable(
                layout
            ):
                continue


            accepted_layout = layout

            self.last_layout_generation_attempts = (
                attempt
            )

            self.multi_last_active_count = (
                active_count
            )

            break


        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate a solvable "
                "multi-randomized obstacle layout."
            )


        self._apply_obstacle_layout(
            accepted_layout
        )



    # ============================================================
    # TRUE FULL-MAP SIX-OBSTACLE RANDOMIZATION
    # ============================================================

    def _set_showcase_six_random_obstacles(self):
        """
        Six REAL URDF obstacles are independently randomized over
        the full usable map every episode:

            2 pedestrians
            2 delivery vehicles
            1 road barrier
            1 parcel stack

        There are NO fixed left/right zones.

        Therefore an obstacle that was on the left in one run can
        appear on the right, centre, upper or lower part of the map
        in the next run.

        A layout is accepted only if:
          - START remains clear;
          - PICKUP remains clear;
          - GOAL remains clear;
          - obstacles are not congested;
          - START -> PACKAGE -> GOAL remains A* solvable.
        """

        accepted_layout = None

        for layout_attempt in range(
            1,
            self.showcase_max_layout_attempts + 1
        ):

            layout = []
            failed = False

            for i in range(
                len(
                    self.obstacle_sizes
                )
            ):

                placed = False

                for _ in range(
                    self.showcase_per_obstacle_attempts
                ):

                    x = float(
                        self.np_random.uniform(
                            self.showcase_random_x_min,
                            self.showcase_random_x_max
                        )
                    )

                    y = float(
                        self.np_random.uniform(
                            self.showcase_random_y_min,
                            self.showcase_random_y_max
                        )
                    )

                    candidate = np.array(
                        [
                            x,
                            y
                        ],
                        dtype=np.float32
                    )

                    # --------------------------------------------
                    # Keep task landmarks open.
                    # --------------------------------------------

                    if (
                        np.linalg.norm(
                            candidate
                            -
                            self.start_position
                        )
                        <
                        self.START_CLEARANCE
                    ):
                        continue

                    if (
                        np.linalg.norm(
                            candidate
                            -
                            self.package_position
                        )
                        <
                        self.PACKAGE_CLEARANCE
                    ):
                        continue

                    if (
                        np.linalg.norm(
                            candidate
                            -
                            self.goal_position
                        )
                        <
                        self.GOAL_CLEARANCE
                    ):
                        continue


                    # --------------------------------------------
                    # Keep obstacles separated from one another.
                    # --------------------------------------------

                    too_close = False

                    for existing in layout:

                        existing_xy = np.array(
                            [
                                existing[0],
                                existing[1]
                            ],
                            dtype=np.float32
                        )

                        if (
                            np.linalg.norm(
                                candidate
                                -
                                existing_xy
                            )
                            <
                            self.showcase_min_separation
                        ):
                            too_close = True
                            break

                    if too_close:
                        continue


                    # --------------------------------------------
                    # Accept this obstacle position.
                    # --------------------------------------------

                    sx, sy, sz = (
                        self.obstacle_sizes[i]
                    )

                    layout.append(
                        [
                            x,
                            y,
                            float(sx),
                            float(sy),
                            float(sz)
                        ]
                    )

                    placed = True
                    break


                if not placed:
                    failed = True
                    break


            if failed:
                continue


            # --------------------------------------------
            # Whole mission must remain navigable.
            # --------------------------------------------

            if not self._layout_is_solvable(
                layout
            ):
                continue


            accepted_layout = layout

            self.last_layout_generation_attempts = (
                layout_attempt
            )

            break


        if accepted_layout is None:
            raise RuntimeError(
                "Could not generate a solvable TRUE-random "
                "six-obstacle showcase layout."
            )


        self._apply_obstacle_layout(
            accepted_layout
        )



    # ============================================================
    # FINAL DUAL-DELIVERY SOLVABILITY
    # ============================================================

    def _dual_layout_is_solvable(
        self,
        obstacle_layout
    ):

        occupancy = (
            self._build_path_occupancy_grid(
                obstacle_layout
            )
        )

        required_legs = [
            (
                self.start_position,
                self.package_position
            ),
            (
                self.package_position,
                self.goal_a_position
            ),
            (
                self.goal_a_position,
                self.goal_b_position
            )
        ]

        for start_xy, goal_xy in required_legs:

            if not self._astar_path_exists(
                occupancy,
                start_xy,
                goal_xy
            ):
                return False

        return True


    # ============================================================
    # FINAL TWO-LEG RANDOM OBSTACLE CYCLE
    # ============================================================

    def _sample_open_leg_obstacle_position(
        self,
        leg_start,
        leg_end,
        already_sampled
    ):
        """
        Sample one obstacle around a route segment.

        Longitudinal position and left/right lateral offset are both
        randomized. This means an obstacle that was on the left in
        one run can appear on the right in another run.
        """

        leg_start = np.asarray(
            leg_start,
            dtype=np.float32
        )

        leg_end = np.asarray(
            leg_end,
            dtype=np.float32
        )

        route = (
            leg_end
            -
            leg_start
        )

        route_length = float(
            np.linalg.norm(
                route
            )
        )

        if route_length < 1e-6:
            return None


        direction = (
            route
            /
            route_length
        )

        perpendicular = np.array(
            [
                -direction[1],
                direction[0]
            ],
            dtype=np.float32
        )


        mission_points = [
            self.start_position,
            self.package_position,
            self.goal_a_position,
            self.goal_b_position
        ]


        for _ in range(
            self.dual_position_attempts
        ):

            t = float(
                self.np_random.uniform(
                    self.dual_t_min,
                    self.dual_t_max
                )
            )

            lateral = float(
                self.np_random.uniform(
                    self.dual_lateral_min,
                    self.dual_lateral_max
                )
            )


            candidate = (
                leg_start
                +
                t * route
                +
                lateral * perpendicular
            ).astype(
                np.float32
            )


            # --------------------------------------------
            # Stay inside useful world area.
            # --------------------------------------------

            if (
                abs(
                    float(
                        candidate[0]
                    )
                )
                >
                self.WORLD_LIMIT - 1.0
                or
                abs(
                    float(
                        candidate[1]
                    )
                )
                >
                self.WORLD_LIMIT - 1.0
            ):
                continue


            # --------------------------------------------
            # Keep START/PICKUP/A/B open.
            # --------------------------------------------

            too_close_landmark = any(
                np.linalg.norm(
                    candidate
                    -
                    np.asarray(
                        point,
                        dtype=np.float32
                    )
                )
                <
                self.dual_landmark_clearance
                for point in mission_points
            )

            if too_close_landmark:
                continue


            # --------------------------------------------
            # Keep all obstacles visually open.
            # --------------------------------------------

            too_close_obstacle = any(
                np.linalg.norm(
                    candidate
                    -
                    previous
                )
                <
                self.dual_min_obstacle_separation
                for previous in already_sampled
            )

            if too_close_obstacle:
                continue


            return candidate


        return None


    def _set_dual_delivery_random_obstacles(self):
        """
        FINAL layout:

            START/PICKUP -> DELIVERY A:
                4 widely randomized obstacles

            DELIVERY A -> DELIVERY B:
                4 widely randomized obstacles

        A->B has the same route displacement as START->A.

        Asset types are shuffled between all eight sampled positions,
        so the pedestrian/car/barrier/parcel identities are NOT tied
        to a fixed section or side.
        """

        accepted_layout = None


        for layout_attempt in range(
            1,
            self.dual_layout_attempts + 1
        ):

            sampled_positions = []
            failed = False


            # --------------------------------------------
            # LEG 1: START/PICKUP -> DELIVERY A
            # --------------------------------------------

            for _ in range(
                self.dual_obstacles_per_leg
            ):

                position = (
                    self._sample_open_leg_obstacle_position(
                        self.package_position,
                        self.goal_a_position,
                        sampled_positions
                    )
                )

                if position is None:
                    failed = True
                    break

                sampled_positions.append(
                    position
                )


            if failed:
                continue


            # --------------------------------------------
            # LEG 2: DELIVERY A -> DELIVERY B
            # --------------------------------------------

            for _ in range(
                self.dual_obstacles_per_leg
            ):

                position = (
                    self._sample_open_leg_obstacle_position(
                        self.goal_a_position,
                        self.goal_b_position,
                        sampled_positions
                    )
                )

                if position is None:
                    failed = True
                    break

                sampled_positions.append(
                    position
                )


            if failed:
                continue


            # --------------------------------------------
            # RANDOM ASSET-TO-POSITION ASSIGNMENT
            # --------------------------------------------

            position_order = list(
                range(
                    len(
                        sampled_positions
                    )
                )
            )

            self.np_random.shuffle(
                position_order
            )


            layout = []

            for asset_index, position_index in enumerate(
                position_order
            ):

                candidate = (
                    sampled_positions[
                        position_index
                    ]
                )

                sx, sy, sz = (
                    self.obstacle_sizes[
                        asset_index
                    ]
                )

                layout.append(
                    [
                        float(
                            candidate[0]
                        ),
                        float(
                            candidate[1]
                        ),
                        float(sx),
                        float(sy),
                        float(sz)
                    ]
                )


            # --------------------------------------------
            # All three route legs must remain solvable.
            # --------------------------------------------

            if not self._dual_layout_is_solvable(
                layout
            ):
                continue


            accepted_layout = layout

            self.last_layout_generation_attempts = (
                layout_attempt
            )

            break


        if accepted_layout is None:

            raise RuntimeError(
                "Could not generate an open solvable 4+4 "
                "START -> A -> B obstacle layout."
            )


        self._apply_obstacle_layout(
            accepted_layout
        )


    # ============================================================
    # RANDOM OBSTACLE LAYOUT GENERATION
    # ============================================================

    def _sample_single_obstacle_position(
        self,
        obstacle_index,
        existing_obstacles
    ):
        """
        Sample one obstacle position for the active curriculum.

        Returns
        -------
        list | None
            [x, y, sx, sy, sz] when a valid local placement
            is found, otherwise None.

        This function checks only local placement constraints.
        Whole-map solvability is checked later.
        """

        sx, sy, sz = (
            self.obstacle_sizes[
                obstacle_index
            ]
        )

        base_x, base_y = (
            self.base_obstacle_positions[
                obstacle_index
            ]
        )


        for _ in range(
            self.MAX_OBSTACLE_PLACEMENT_ATTEMPTS
        ):

            # ====================================================
            # MILD CURRICULUM
            # ====================================================

            if (
                self.curriculum_level
                ==
                "mild"
            ):

                x = float(
                    self.np_random.uniform(
                        base_x
                        -
                        self.mild_range,
                        base_x
                        +
                        self.mild_range
                    )
                )

                y = float(
                    self.np_random.uniform(
                        base_y
                        -
                        self.mild_range,
                        base_y
                        +
                        self.mild_range
                    )
                )


            # ====================================================
            # MEDIUM CURRICULUM
            # ====================================================

            elif (
                self.curriculum_level
                ==
                "medium"
            ):

                x = float(
                    self.np_random.uniform(
                        base_x
                        -
                        self.medium_range,
                        base_x
                        +
                        self.medium_range
                    )
                )

                y = float(
                    self.np_random.uniform(
                        base_y
                        -
                        self.medium_range,
                        base_y
                        +
                        self.medium_range
                    )
                )


            # ====================================================
            # FULL CURRICULUM
            # ====================================================

            else:

                x = float(
                    self.np_random.uniform(
                        self.FULL_X_MIN,
                        self.FULL_X_MAX
                    )
                )

                y = float(
                    self.np_random.uniform(
                        self.FULL_Y_MIN,
                        self.FULL_Y_MAX
                    )
                )


            candidate = np.array(
                [
                    x,
                    y
                ],
                dtype=np.float32
            )


            # ====================================================
            # KEEP CLEAR OF START
            # ====================================================

            if (
                np.linalg.norm(
                    candidate
                    -
                    self.start_position
                )
                <
                self.START_CLEARANCE
            ):

                continue


            # ====================================================
            # KEEP CLEAR OF PACKAGE
            # ====================================================

            if (
                np.linalg.norm(
                    candidate
                    -
                    self.package_position
                )
                <
                self.PACKAGE_CLEARANCE
            ):

                continue


            # ====================================================
            # KEEP CLEAR OF GOAL
            # ====================================================

            if (
                np.linalg.norm(
                    candidate
                    -
                    self.goal_position
                )
                <
                self.GOAL_CLEARANCE
            ):

                continue


            # ====================================================
            # KEEP OBSTACLES SEPARATED
            # ====================================================

            too_close = False

            for existing in existing_obstacles:

                existing_xy = np.array(
                    [
                        existing[0],
                        existing[1]
                    ],
                    dtype=np.float32
                )

                if (
                    np.linalg.norm(
                        candidate
                        -
                        existing_xy
                    )
                    <
                    self.OBSTACLE_CLEARANCE
                ):

                    too_close = True
                    break


            if too_close:

                continue


            return [
                x,
                y,
                float(sx),
                float(sy),
                float(sz)
            ]


        return None


    def _sample_candidate_layout(self):
        """
        Generate one complete candidate obstacle layout.

        Returns
        -------
        list | None
            List of obstacle records, or None if local placement
            failed before all obstacles could be placed.
        """

        candidate_layout = []


        for obstacle_index in range(
            len(
                self.obstacle_sizes
            )
        ):

            obstacle = (
                self._sample_single_obstacle_position(
                    obstacle_index,
                    candidate_layout
                )
            )


            if obstacle is None:

                return None


            candidate_layout.append(
                obstacle
            )


        return candidate_layout


    # ============================================================
    # SOLVABILITY GRID
    # ============================================================

    def _path_world_to_grid(
        self,
        x,
        y
    ):

        world_min = (
            -float(
                self.WORLD_LIMIT
            )
        )

        gx = int(
            round(
                (
                    float(x)
                    -
                    world_min
                )
                /
                self.PATH_GRID_RESOLUTION
            )
        )

        gy = int(
            round(
                (
                    float(y)
                    -
                    world_min
                )
                /
                self.PATH_GRID_RESOLUTION
            )
        )


        return (
            gx,
            gy
        )


    def _build_path_occupancy_grid(
        self,
        obstacle_layout
    ):
        """
        Build a 2D occupancy grid for route-existence testing.

        Obstacles and world boundaries are inflated by the robot
        safety radius plus a small extra margin. This lets A*
        treat the finite-size robot as a point.
        """

        world_min = (
            -float(
                self.WORLD_LIMIT
            )
        )

        world_max = float(
            self.WORLD_LIMIT
        )

        resolution = float(
            self.PATH_GRID_RESOLUTION
        )

        total_inflation = (
            float(
                self.ROBOT_RADIUS
            )
            +
            float(
                self.PATH_EXTRA_SAFETY_MARGIN
            )
        )


        grid_size = int(
            round(
                (
                    world_max
                    -
                    world_min
                )
                /
                resolution
            )
        ) + 1


        occupancy = np.zeros(
            (
                grid_size,
                grid_size
            ),
            dtype=bool
        )


        # ========================================================
        # INFLATE WORLD BOUNDARY
        # ========================================================

        boundary_cells = int(
            math.ceil(
                total_inflation
                /
                resolution
            )
        )


        if boundary_cells > 0:

            occupancy[
                :boundary_cells,
                :
            ] = True

            occupancy[
                -boundary_cells:,
                :
            ] = True

            occupancy[
                :,
                :boundary_cells
            ] = True

            occupancy[
                :,
                -boundary_cells:
            ] = True


        # ========================================================
        # INFLATE OBSTACLES
        # ========================================================

        for obstacle in obstacle_layout:

            x = float(
                obstacle[0]
            )

            y = float(
                obstacle[1]
            )

            sx = float(
                obstacle[2]
            )

            sy = float(
                obstacle[3]
            )


            min_x = (
                x
                -
                sx / 2.0
                -
                total_inflation
            )

            max_x = (
                x
                +
                sx / 2.0
                +
                total_inflation
            )

            min_y = (
                y
                -
                sy / 2.0
                -
                total_inflation
            )

            max_y = (
                y
                +
                sy / 2.0
                +
                total_inflation
            )


            min_gx, min_gy = (
                self._path_world_to_grid(
                    min_x,
                    min_y
                )
            )

            max_gx, max_gy = (
                self._path_world_to_grid(
                    max_x,
                    max_y
                )
            )


            min_gx = max(
                0,
                min_gx
            )

            min_gy = max(
                0,
                min_gy
            )

            max_gx = min(
                grid_size - 1,
                max_gx
            )

            max_gy = min(
                grid_size - 1,
                max_gy
            )


            occupancy[
                min_gx:
                max_gx + 1,
                min_gy:
                max_gy + 1
            ] = True


        return occupancy


    # ============================================================
    # A* ROUTE-EXISTENCE CHECK
    # ============================================================

    def _path_heuristic(
        self,
        node,
        goal
    ):

        dx = (
            node[0]
            -
            goal[0]
        )

        dy = (
            node[1]
            -
            goal[1]
        )


        return math.sqrt(
            dx * dx
            +
            dy * dy
        )


    def _astar_path_exists(
        self,
        occupancy,
        start_xy,
        goal_xy
    ):
        """
        Return True when a collision-free route exists.

        A* is ONLY a layout validator. Its path is never returned
        to SAC and never used to drive the robot.
        """

        start = (
            self._path_world_to_grid(
                start_xy[0],
                start_xy[1]
            )
        )

        goal = (
            self._path_world_to_grid(
                goal_xy[0],
                goal_xy[1]
            )
        )


        grid_x_size = (
            occupancy.shape[0]
        )

        grid_y_size = (
            occupancy.shape[1]
        )


        # ========================================================
        # START / GOAL VALIDITY
        # ========================================================

        for node in [
            start,
            goal
        ]:

            gx, gy = node


            if (
                gx < 0
                or
                gx >= grid_x_size
                or
                gy < 0
                or
                gy >= grid_y_size
            ):

                return False


        if occupancy[
            start[0],
            start[1]
        ]:

            return False


        if occupancy[
            goal[0],
            goal[1]
        ]:

            return False


        # ========================================================
        # 8-CONNECTED GRID
        # ========================================================

        neighbours = [
            (
                -1,
                0,
                1.0
            ),
            (
                1,
                0,
                1.0
            ),
            (
                0,
                -1,
                1.0
            ),
            (
                0,
                1,
                1.0
            ),
            (
                -1,
                -1,
                math.sqrt(2.0)
            ),
            (
                -1,
                1,
                math.sqrt(2.0)
            ),
            (
                1,
                -1,
                math.sqrt(2.0)
            ),
            (
                1,
                1,
                math.sqrt(2.0)
            )
        ]


        open_heap = []

        heapq.heappush(
            open_heap,
            (
                self._path_heuristic(
                    start,
                    goal
                ),
                0.0,
                start
            )
        )


        best_cost = {
            start: 0.0
        }


        while open_heap:

            (
                _,
                current_cost,
                current
            ) = heapq.heappop(
                open_heap
            )


            if current == goal:

                return True


            if (
                current_cost
                >
                best_cost.get(
                    current,
                    float("inf")
                )
            ):

                continue


            cx, cy = current


            for (
                dx,
                dy,
                step_cost
            ) in neighbours:

                nx = (
                    cx
                    +
                    dx
                )

                ny = (
                    cy
                    +
                    dy
                )


                if (
                    nx < 0
                    or
                    nx >= grid_x_size
                    or
                    ny < 0
                    or
                    ny >= grid_y_size
                ):

                    continue


                if occupancy[
                    nx,
                    ny
                ]:

                    continue


                # =================================================
                # PREVENT DIAGONAL CORNER CUTTING
                # =================================================

                if (
                    dx != 0
                    and
                    dy != 0
                ):

                    if (
                        occupancy[
                            cx + dx,
                            cy
                        ]
                        or
                        occupancy[
                            cx,
                            cy + dy
                        ]
                    ):

                        continue


                new_cost = (
                    current_cost
                    +
                    step_cost
                )

                neighbour = (
                    nx,
                    ny
                )


                if (
                    new_cost
                    <
                    best_cost.get(
                        neighbour,
                        float("inf")
                    )
                ):

                    best_cost[
                        neighbour
                    ] = new_cost


                    priority = (
                        new_cost
                        +
                        self._path_heuristic(
                            neighbour,
                            goal
                        )
                    )


                    heapq.heappush(
                        open_heap,
                        (
                            priority,
                            new_cost,
                            neighbour
                        )
                    )


        return False


    def _layout_is_solvable(
        self,
        obstacle_layout
    ):
        """
        A layout is accepted only when BOTH required route
        segments are physically reachable.
        """

        occupancy = (
            self._build_path_occupancy_grid(
                obstacle_layout
            )
        )


        start_to_package = (
            self._astar_path_exists(
                occupancy,
                self.start_position,
                self.package_position
            )
        )


        if not start_to_package:

            return False


        package_to_goal = (
            self._astar_path_exists(
                occupancy,
                self.package_position,
                self.goal_position
            )
        )


        return bool(
            package_to_goal
        )


    # ============================================================
    # APPLY ACCEPTED LAYOUT TO GENESIS
    # ============================================================

    def _apply_obstacle_layout(
        self,
        obstacle_layout
    ):

        for i, obstacle_data in enumerate(
            obstacle_layout
        ):

            (
                x,
                y,
                sx,
                sy,
                sz
            ) = obstacle_data


            self.obstacles[i].set_pos(
                np.array(
                    [
                        x,
                        y,
                        0.0
                    ],
                    dtype=np.float32
                )
            )


        self.obstacle_data = [
            obstacle.copy()
            for obstacle
            in obstacle_layout
        ]


    # ============================================================
    # RANDOMIZE OBSTACLES WITH SOLVABILITY FILTER
    # ============================================================

    def _randomize_obstacles(self):
        """
        Generate a curriculum-appropriate random layout.

        In goal_only mode, obstacles are disabled entirely.

        A candidate is accepted ONLY when:
          1. local clearance rules pass;
          2. START -> PACKAGE is reachable;
          3. PACKAGE -> GOAL is reachable.

        The A* route is discarded immediately after validation.
        SAC receives no route information.
        """

        if self.task_mode in [
            "goal_only",
            "package_goal"
        ]:

            self._disable_obstacles_for_goal_only()

            return


        if self.task_mode == "one_fixed":

            self._set_stage2_one_fixed_obstacle()

            return


        if self.task_mode == "one_offset":

            self._set_stage2a_one_offset_obstacle()

            return


        if self.task_mode == "one_mid":

            self._set_stage2b_one_mid_obstacle()

            return


        if self.task_mode == "one_random_narrow":

            self._set_stage2c_one_random_narrow_obstacle()

            return


        if self.task_mode == "one_random_wide":

            self._set_stage2d_one_random_wide_obstacle()

            return


        if self.task_mode == "one_random_wider":

            self._set_stage2e_one_random_wider_obstacle()

            return


        if self.task_mode == "two_random":

            self._set_stage3_two_random_obstacles()

            return


        if self.task_mode == "two_random_easy":

            self._set_stage3a_two_random_easy_obstacles()

            return


        if self.task_mode == "one_random_routewide":

            self._set_one_random_routewide_obstacle()

            return


        if self.task_mode == "multi_random_routewide":

            self._set_multi_random_routewide_obstacles(
                force_count=None
            )

            return


        if self.task_mode == "three_random_routewide":

            self._set_multi_random_routewide_obstacles(
                force_count=3
            )

            return


        if self.task_mode == "four_random_routewide":

            self._set_multi_random_routewide_obstacles(
                force_count=4
            )

            return


        if self.task_mode == "showcase_six_random":

            self._set_showcase_six_random_obstacles()

            return


        if self.task_mode == "showcase_dual_delivery":

            self._set_dual_delivery_random_obstacles()

            return


        accepted_layout = None


        for layout_attempt in range(
            1,
            self.MAX_LAYOUT_GENERATION_ATTEMPTS
            +
            1
        ):

            candidate_layout = (
                self._sample_candidate_layout()
            )


            if candidate_layout is None:

                continue


            if not self._layout_is_solvable(
                candidate_layout
            ):

                continue


            accepted_layout = (
                candidate_layout
            )

            self.last_layout_generation_attempts = (
                layout_attempt
            )

            break


        # ========================================================
        # FALLBACK TO BASE LAYOUT
        # ========================================================
        #
        # This should almost never be reached. It prevents a reset
        # from hanging forever if random generation repeatedly
        # fails. The base layout is still checked for solvability.
        # ========================================================

        if accepted_layout is None:

            base_layout = []


            for i, (
                sx,
                sy,
                sz
            ) in enumerate(
                self.obstacle_sizes
            ):

                x, y = (
                    self.base_obstacle_positions[i]
                )

                base_layout.append(
                    [
                        float(x),
                        float(y),
                        float(sx),
                        float(sy),
                        float(sz)
                    ]
                )


            if not self._layout_is_solvable(
                base_layout
            ):

                raise RuntimeError(
                    "Could not generate a solvable obstacle "
                    "layout, and the fallback base layout is "
                    "also not solvable."
                )


            accepted_layout = (
                base_layout
            )

            self.last_layout_generation_attempts = (
                self.MAX_LAYOUT_GENERATION_ATTEMPTS
            )


        self._apply_obstacle_layout(
            accepted_layout
        )

    # ============================================================
    # HELPERS
    # ============================================================

    def _normalize_angle(
        self,
        angle
    ):

        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    def _get_robot_yaw(self):

        quat = (
            self.robot
            .get_quat()
            .cpu()
            .numpy()
        )

        w, x, y, z = quat

        return math.atan2(

            2.0
            *
            (
                w * z
                +
                x * y
            ),

            1.0
            -
            2.0
            *
            (
                y * y
                +
                z * z
            )
        )

    def _get_robot_position(self):

        pos = (
            self.robot
            .get_pos()
            .cpu()
            .numpy()
        )

        return np.array(
            [
                pos[0],
                pos[1]
            ],
            dtype=np.float32
        )

    def _get_active_target(self):

        if not self.package_picked:

            return self.package_position

        return self.goal_position

    def _get_target_information(self):

        robot_position = (
            self._get_robot_position()
        )

        robot_yaw = (
            self._get_robot_yaw()
        )

        target = (
            self._get_active_target()
        )

        dx = (
            target[0]
            -
            robot_position[0]
        )

        dy = (
            target[1]
            -
            robot_position[1]
        )

        distance = math.sqrt(
            dx * dx
            +
            dy * dy
        )

        target_angle = math.atan2(
            dy,
            dx
        )

        angle_error = (
            self._normalize_angle(
                target_angle
                -
                robot_yaw
            )
        )

        return (
            distance,
            angle_error
        )

    # ============================================================
    # LIDAR
    # ============================================================

    def _ray_rectangle_distance(
        self,
        origin,
        direction,
        cx,
        cy,
        sx,
        sy
    ):

        min_x = cx - sx / 2.0
        max_x = cx + sx / 2.0

        min_y = cy - sy / 2.0
        max_y = cy + sy / 2.0

        ox, oy = origin
        dx, dy = direction

        distances = []

        if abs(dx) > 1e-8:

            t = (
                min_x
                -
                ox
            ) / dx

            if t >= 0:

                y_hit = (
                    oy
                    +
                    t * dy
                )

                if (
                    min_y
                    <=
                    y_hit
                    <=
                    max_y
                ):

                    distances.append(
                        t
                    )

            t = (
                max_x
                -
                ox
            ) / dx

            if t >= 0:

                y_hit = (
                    oy
                    +
                    t * dy
                )

                if (
                    min_y
                    <=
                    y_hit
                    <=
                    max_y
                ):

                    distances.append(
                        t
                    )

        if abs(dy) > 1e-8:

            t = (
                min_y
                -
                oy
            ) / dy

            if t >= 0:

                x_hit = (
                    ox
                    +
                    t * dx
                )

                if (
                    min_x
                    <=
                    x_hit
                    <=
                    max_x
                ):

                    distances.append(
                        t
                    )

            t = (
                max_y
                -
                oy
            ) / dy

            if t >= 0:

                x_hit = (
                    ox
                    +
                    t * dx
                )

                if (
                    min_x
                    <=
                    x_hit
                    <=
                    max_x
                ):

                    distances.append(
                        t
                    )

        if len(
            distances
        ) == 0:

            return None

        return min(
            distances
        )

    def _get_lidar(self):

        robot_position = (
            self._get_robot_position()
        )

        robot_yaw = (
            self._get_robot_yaw()
        )

        lidar_values = []

        for i in range(
            self.NUM_LIDAR_RAYS
        ):

            relative_angle = (

                2.0
                *
                np.pi
                *
                i
                /
                self.NUM_LIDAR_RAYS
            )

            world_angle = (
                robot_yaw
                +
                relative_angle
            )

            direction = np.array(
                [
                    math.cos(
                        world_angle
                    ),

                    math.sin(
                        world_angle
                    )
                ]
            )

            closest_distance = (
                self.LIDAR_MAX_DISTANCE
            )

            for (
                x,
                y,
                sx,
                sy,
                sz
            ) in self.obstacle_data:

                distance = (
                    self._ray_rectangle_distance(

                        robot_position,
                        direction,

                        x,
                        y,

                        sx,
                        sy
                    )
                )

                if (
                    distance is not None
                    and
                    distance < closest_distance
                ):

                    closest_distance = (
                        distance
                    )

            closest_distance = np.clip(

                closest_distance,

                self.LIDAR_MIN_DISTANCE,

                self.LIDAR_MAX_DISTANCE
            )

            lidar_values.append(

                closest_distance

                /

                self.LIDAR_MAX_DISTANCE
            )

        return np.array(
            lidar_values,
            dtype=np.float32
        )

    # ============================================================
    # OBSERVATION
    # ============================================================

    def _get_observation(self):

        lidar = (
            self._get_lidar()
        )

        distance, angle_error = (
            self._get_target_information()
        )

        carrying = (
            1.0
            if self.package_picked
            else 0.0
        )

        extra = np.array(
            [
                distance,
                angle_error,
                self.linear_velocity,
                self.angular_velocity,
                carrying
            ],
            dtype=np.float32
        )

        return np.concatenate(
            [
                lidar,
                extra
            ]
        ).astype(
            np.float32
        )

    # ============================================================
    # CONTROL
    # ============================================================

    def _velocity_to_wheels(
        self,
        v,
        omega
    ):

        left = (

            v

            -
            omega
            *
            self.TRACK_WIDTH
            /
            2.0

        ) / self.WHEEL_RADIUS

        right = (

            v

            +
            omega
            *
            self.TRACK_WIDTH
            /
            2.0

        ) / self.WHEEL_RADIUS

        return (
            left,
            right
        )

    def _apply_action(
        self,
        action
    ):

        action = np.asarray(
            action,
            dtype=np.float32
        )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        self.linear_velocity = (

            float(
                action[0]
            )

            *
            self.MAX_LINEAR_SPEED
        )

        self.angular_velocity = (

            float(
                action[1]
            )

            *
            self.MAX_ANGULAR_SPEED
        )

        left, right = (
            self._velocity_to_wheels(
                self.linear_velocity,
                self.angular_velocity
            )
        )

        wheel_velocities = np.array(
            [
                left,
                left,
                right,
                right
            ],
            dtype=np.float32
        )

        self.robot.control_dofs_velocity(

            wheel_velocities,

            self.wheel_dofs
        )

    # ============================================================
    # COLLISION
    # ============================================================

    def _check_collision(self):
        """
        Collision using the robot's rotated rectangular footprint.

        This replaces the old 0.80 m circular approximation which
        could declare collision even while a visible gap remained.
        """

        robot_position = np.asarray(
            self._get_robot_position(),
            dtype=np.float32
        )

        robot_yaw = float(
            self._get_robot_yaw()
        )

        half_length = float(
            self.ROBOT_COLLISION_HALF_LENGTH
            +
            self.COLLISION_CONTACT_MARGIN
        )

        half_width = float(
            self.ROBOT_COLLISION_HALF_WIDTH
            +
            self.COLLISION_CONTACT_MARGIN
        )

        # Robot local axes in world coordinates.
        forward = np.array(
            [
                math.cos(robot_yaw),
                math.sin(robot_yaw)
            ],
            dtype=np.float32
        )

        left = np.array(
            [
                -math.sin(robot_yaw),
                math.cos(robot_yaw)
            ],
            dtype=np.float32
        )


        for (
            x,
            y,
            sx,
            sy,
            sz
        ) in self.obstacle_data:

            obstacle_center = np.array(
                [
                    float(x),
                    float(y)
                ],
                dtype=np.float32
            )

            obstacle_half_x = float(sx) / 2.0
            obstacle_half_y = float(sy) / 2.0

            center_delta = (
                obstacle_center
                -
                robot_position
            )


            # ----------------------------------------------------
            # SAT axis: world X
            # ----------------------------------------------------

            robot_projection_x = (
                half_length
                *
                abs(
                    float(
                        forward[0]
                    )
                )
                +
                half_width
                *
                abs(
                    float(
                        left[0]
                    )
                )
            )

            if (
                abs(
                    float(
                        center_delta[0]
                    )
                )
                >
                robot_projection_x
                +
                obstacle_half_x
            ):
                continue


            # ----------------------------------------------------
            # SAT axis: world Y
            # ----------------------------------------------------

            robot_projection_y = (
                half_length
                *
                abs(
                    float(
                        forward[1]
                    )
                )
                +
                half_width
                *
                abs(
                    float(
                        left[1]
                    )
                )
            )

            if (
                abs(
                    float(
                        center_delta[1]
                    )
                )
                >
                robot_projection_y
                +
                obstacle_half_y
            ):
                continue


            # ----------------------------------------------------
            # SAT axis: robot forward
            # ----------------------------------------------------

            centre_forward = abs(
                float(
                    np.dot(
                        center_delta,
                        forward
                    )
                )
            )

            obstacle_on_forward = (
                obstacle_half_x
                *
                abs(
                    float(
                        forward[0]
                    )
                )
                +
                obstacle_half_y
                *
                abs(
                    float(
                        forward[1]
                    )
                )
            )

            if (
                centre_forward
                >
                half_length
                +
                obstacle_on_forward
            ):
                continue


            # ----------------------------------------------------
            # SAT axis: robot left
            # ----------------------------------------------------

            centre_left = abs(
                float(
                    np.dot(
                        center_delta,
                        left
                    )
                )
            )

            obstacle_on_left = (
                obstacle_half_x
                *
                abs(
                    float(
                        left[0]
                    )
                )
                +
                obstacle_half_y
                *
                abs(
                    float(
                        left[1]
                    )
                )
            )

            if (
                centre_left
                >
                half_width
                +
                obstacle_on_left
            ):
                continue


            # No separating axis -> actual footprint overlap.
            return True


        return False


    def _check_out_of_bounds(self):

        position = (
            self._get_robot_position()
        )

        return bool(

            abs(
                float(
                    position[0]
                )
            )
            >
            self.WORLD_LIMIT

            or

            abs(
                float(
                    position[1]
                )
            )
            >
            self.WORLD_LIMIT
        )

    # ============================================================
    # REWARD
    # ============================================================

    def _calculate_reward(
        self,
        progress,
        angle_error
    ):

        # ========================================================
        # CLEAN REWARD DESIGN
        # ========================================================
        # Main principle:
        #   - move closer to the active target -> positive reward
        #   - move farther away -> negative reward
        #   - standing still -> small negative reward
        #   - pickup/delivery -> large bonuses
        #   - unsafe proximity/collision/out-of-bounds -> penalties
        #
        # angle_error is intentionally not rewarded directly.
        # The robot may need to temporarily turn away from the
        # target in order to go around an obstacle.
        # ========================================================

        reward = 0.0

        # --------------------------------------------------------
        # 1. PROGRESS TOWARD ACTIVE TARGET
        # --------------------------------------------------------
        # progress = previous_distance - current_distance
        # Positive progress means the robot moved closer.
        # Negative progress means it moved farther away.
        # --------------------------------------------------------

        PROGRESS_WEIGHT = 15.0

        reward += (
            PROGRESS_WEIGHT
            *
            float(progress)
        )

        # --------------------------------------------------------
        # 2. TIME PENALTY
        # --------------------------------------------------------
        # Makes standing still slightly costly.
        # --------------------------------------------------------

        reward -= 0.02

        # --------------------------------------------------------
        # 3. PACKAGE PICKUP BONUS
        # --------------------------------------------------------

        if self.just_picked_package:

            reward += 25.0

        # Intermediate Delivery A is a useful positive transition
        # for the experience buffer, but it does NOT end the run.
        if self.just_delivered_a:

            reward += 50.0

        # --------------------------------------------------------
        # 4. FINAL DELIVERY B BONUS
        # --------------------------------------------------------

        if self.package_delivered:

            reward += 150.0

        # --------------------------------------------------------
        # 5. COLLISION PENALTY
        # --------------------------------------------------------

        if self.collision:

            reward -= 100.0

        # --------------------------------------------------------
        # 6. OUT-OF-BOUNDS PENALTY
        # --------------------------------------------------------

        if self.out_of_bounds:

            reward -= 100.0

        # --------------------------------------------------------
        # 7. OBSTACLE PROXIMITY PENALTY
        # --------------------------------------------------------
        # _get_lidar() returns normalized values in [0, 1].
        # Convert back to metres so the threshold is physically
        # meaningful and easy to inspect.
        # --------------------------------------------------------

        lidar_normalized = (
            self._get_lidar()
        )

        lidar_metres = (
            lidar_normalized
            *
            self.LIDAR_MAX_DISTANCE
        )

        min_obstacle_distance = float(
            np.min(
                lidar_metres
            )
        )

        SAFE_DISTANCE = 1.20

        if (
            min_obstacle_distance
            <
            SAFE_DISTANCE
        ):

            danger_ratio = (
                SAFE_DISTANCE
                -
                min_obstacle_distance
            ) / SAFE_DISTANCE

            reward -= (
                0.5
                *
                danger_ratio
            )

        return float(
            reward
        )

    # ============================================================
    # RESET
    # ============================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        # Stage 0 begins already carrying the package.
        self.package_picked = (
            self.task_mode == "goal_only"
        )

        self.package_delivered = False
        self.just_picked_package = False

        self.delivery_a_complete = False
        self.delivery_b_complete = False
        self.just_delivered_a = False

        # Every fresh mission starts by targeting Delivery A.
        self.goal_position = (
            self.goal_a_position.copy()
        )

        self.collision = False
        self.out_of_bounds = False

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0

        self.current_step = 0

        self.robot.set_pos(

            np.array(
                [
                    self.start_position[0],
                    self.start_position[1],
                    0.0
                ],
                dtype=np.float32
            )
        )

        self.robot.set_quat(

            np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ],
                dtype=np.float32
            )
        )

        if self.task_mode == "goal_only":

            # Keep the physical package away from the robot during
            # Stage 0. The carrying state is logical only here.
            self.package.set_pos(
                np.array(
                    [
                        25.0,
                        25.0,
                        2.0
                    ],
                    dtype=np.float32
                )
            )

        else:

            # Two parcels are visible side-by-side inside the SAME
            # yellow pickup zone.
            self.package.set_pos(
                np.array(
                    [
                        self.package_position[0],
                        self.package_position[1] - 0.22,
                        0.18
                    ],
                    dtype=np.float32
                )
            )

            self.package_b.set_pos(
                np.array(
                    [
                        self.package_position[0],
                        self.package_position[1] + 0.22,
                        0.18
                    ],
                    dtype=np.float32
                )
            )

        # ========================================================
        # RANDOMIZE ACCORDING TO CURRENT CURRICULUM
        # ========================================================

        self._randomize_obstacles()

        self.robot.control_dofs_velocity(

            np.zeros(
                4,
                dtype=np.float32
            ),

            self.wheel_dofs
        )

        for _ in range(10):

            self.scene.step()

        observation = (
            self._get_observation()
        )

        info = {

            "package_picked":
                self.package_picked,

            "package_delivered":
                False,

            "delivery_a_complete":
                self.delivery_a_complete,

            "delivery_b_complete":
                self.delivery_b_complete,

            "success":
                False,

            "collision":
                False,

            "out_of_bounds":
                False,

            "curriculum_level":
                self.curriculum_level,

            "layout_generation_attempts":
                self.last_layout_generation_attempts,

            "obstacles":
                [
                    obstacle.copy()
                    for obstacle
                    in self.obstacle_data
                ]
        }

        return (
            observation,
            info
        )

    # ============================================================
    # STEP
    # ============================================================

    def step(
        self,
        action
    ):

        self.current_step += 1

        self.just_picked_package = False
        self.just_delivered_a = False

        target_for_progress = (

            self.package_position.copy()

            if not self.package_picked

            else

            self.goal_position.copy()
        )

        robot_before = (
            self._get_robot_position()
        )

        previous_distance = float(

            np.linalg.norm(

                robot_before

                -

                target_for_progress
            )
        )

        robot_yaw = (
            self._get_robot_yaw()
        )

        dx = (
            target_for_progress[0]
            -
            robot_before[0]
        )

        dy = (
            target_for_progress[1]
            -
            robot_before[1]
        )

        target_angle = math.atan2(
            dy,
            dx
        )

        angle_error = (
            self._normalize_angle(
                target_angle
                -
                robot_yaw
            )
        )

        # ========================================================
        # ACTION
        # ========================================================

        self._apply_action(
            action
        )

        for _ in range(
            self.ACTION_REPEAT
        ):

            self.scene.step()

        robot_position = (
            self._get_robot_position()
        )

        current_same_target_distance = float(

            np.linalg.norm(

                robot_position

                -

                target_for_progress
            )
        )

        progress = (

            previous_distance

            -

            current_same_target_distance
        )

        # ========================================================
        # PICKUP — BOTH PARCELS AT ONCE
        # ========================================================

        if not self.package_picked:

            package_distance = float(
                np.linalg.norm(
                    robot_position
                    -
                    self.package_position
                )
            )

            if (
                package_distance
                <
                self.pickup_distance
            ):

                self.package_picked = True
                self.just_picked_package = True
                self.pickup_count += 1


        # ========================================================
        # PACKAGE FOLLOW
        # ========================================================

        if (
            self.task_mode == "showcase_dual_delivery"
            and
            self.package_picked
            and
            not self.package_delivered
        ):

            robot_3d = (
                self.robot
                .get_pos()
                .cpu()
                .numpy()
            )

            # Parcel A follows only until Delivery A.
            if not self.delivery_a_complete:

                self.package.set_pos(
                    np.array(
                        [
                            robot_3d[0],
                            robot_3d[1] - 0.18,
                            robot_3d[2] + 0.64
                        ],
                        dtype=np.float32
                    )
                )

            # Parcel B remains on the robot all the way to B.
            self.package_b.set_pos(
                np.array(
                    [
                        robot_3d[0],
                        robot_3d[1] + 0.18,
                        robot_3d[2] + 0.74
                    ],
                    dtype=np.float32
                )
            )


        elif (
            self.task_mode != "goal_only"
            and
            self.package_picked
            and
            not self.package_delivered
        ):

            robot_3d = (
                self.robot
                .get_pos()
                .cpu()
                .numpy()
            )

            self.package.set_pos(
                np.array(
                    [
                        robot_3d[0],
                        robot_3d[1],
                        robot_3d[2] + 0.65
                    ],
                    dtype=np.float32
                )
            )


        # ========================================================
        # DELIVERY A -> CONTINUE DIRECTLY TO B
        # ========================================================

        if (
            self.task_mode == "showcase_dual_delivery"
            and
            self.package_picked
            and
            not self.delivery_a_complete
        ):

            distance_to_a = float(
                np.linalg.norm(
                    robot_position
                    -
                    self.goal_a_position
                )
            )

            if (
                distance_to_a
                <
                self.delivery_distance
            ):

                self.delivery_a_complete = True
                self.just_delivered_a = True
                self.delivery_count += 1

                # ------------------------------------------------
                # Parcel A is considered delivered, then removed
                # BELOW the scene.
                #
                # This is intentional: the package Box has physical
                # collision. Leaving it on the Delivery A pad could
                # trap/block the robot as it tries to continue to B.
                #
                # Hiding it below ground gives the requested
                # "invisible / pass-through after delivery" behavior.
                # ------------------------------------------------

                self.package.set_pos(
                    np.array(
                        [
                            0.0,
                            0.0,
                            -8.0
                        ],
                        dtype=np.float32
                    )
                )

                # IMPORTANT:
                # The episode does NOT terminate here.
                # Target changes immediately to Delivery B.
                self.goal_position = (
                    self.goal_b_position.copy()
                )


        # ========================================================
        # DELIVERY B -> FINAL SUCCESS
        # ========================================================

        if (
            self.task_mode == "showcase_dual_delivery"
            and
            self.package_picked
            and
            self.delivery_a_complete
            and
            not self.delivery_b_complete
        ):

            distance_to_b = float(
                np.linalg.norm(
                    robot_position
                    -
                    self.goal_b_position
                )
            )

            if (
                distance_to_b
                <
                self.delivery_distance
            ):

                self.delivery_b_complete = True
                self.package_delivered = True
                self.delivery_count += 1

                # Final parcel is also hidden below the scene after
                # successful delivery so the delivery pad remains
                # physically clear.
                self.package_b.set_pos(
                    np.array(
                        [
                            0.0,
                            0.0,
                            -8.0
                        ],
                        dtype=np.float32
                    )
                )


        # ========================================================
        # ORIGINAL SINGLE-DELIVERY MODES
        # ========================================================

        elif (
            self.task_mode != "showcase_dual_delivery"
            and
            self.package_picked
            and
            not self.package_delivered
        ):

            goal_distance = float(
                np.linalg.norm(
                    robot_position
                    -
                    self.goal_position
                )
            )

            if (
                goal_distance
                <
                self.delivery_distance
            ):

                self.package_delivered = True
                self.delivery_count += 1

                self.package.set_pos(
                    np.array(
                        [
                            self.goal_position[0],
                            self.goal_position[1],
                            0.25
                        ],
                        dtype=np.float32
                    )
                )

        # ========================================================
        # SAFETY
        # ========================================================

        self.collision = (
            self._check_collision()
        )

        self.out_of_bounds = (
            self._check_out_of_bounds()
        )

        # ========================================================
        # REWARD
        # ========================================================

        reward = (
            self._calculate_reward(
                progress,
                angle_error
            )
        )

        terminated = bool(

            self.package_delivered

            or

            self.collision

            or

            self.out_of_bounds
        )

        truncated = bool(

            self.current_step

            >=

            self.max_steps
        )

        observation = (
            self._get_observation()
        )

        current_distance, _ = (
            self._get_target_information()
        )

        info = {

            "package_picked":
                self.package_picked,

            "package_delivered":
                self.package_delivered,

            "delivery_a_complete":
                self.delivery_a_complete,

            "delivery_b_complete":
                self.delivery_b_complete,

            "just_delivered_a":
                self.just_delivered_a,

            "success":
                self.package_delivered,

            "collision":
                self.collision,

            "out_of_bounds":
                self.out_of_bounds,

            "distance_to_target":
                float(
                    current_distance
                ),

            "progress":
                float(
                    progress
                ),

            "pickup_count":
                self.pickup_count,

            "delivery_count":
                self.delivery_count,

            "curriculum_level":
                self.curriculum_level,

            "layout_generation_attempts":
                self.last_layout_generation_attempts,

            "obstacles":
                [
                    obstacle.copy()
                    for obstacle
                    in self.obstacle_data
                ],

            "step":
                self.current_step
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):

        pass