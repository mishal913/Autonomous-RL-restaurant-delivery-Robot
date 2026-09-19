import os
import math
import heapq
import random
import msvcrt
import csv
from datetime import datetime
from collections import deque

# Genesis viewer uses this folder for temporary video recording.
# Create it proactively so accidental recording hotkeys do not crash.
GENESIS_CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "genesis"
)
os.makedirs(
    GENESIS_CACHE_DIR,
    exist_ok=True
)


import numpy as np
import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer

from delivery_env_final_dual_delivery_v4_camera import DeliveryRobotEnv


# ============================================================
# FINAL PROJECT: HYBRID SAC + A* MULTI-RANDOM ROBOT
# ============================================================
#
# PURPOSE
# -------
# Finish the project without another RL training cycle.
#
# Architecture:
#
#   randomized obstacles
#          ↓
#   A* global path planner
#          ↓
#   safe local waypoint
#          ↓
#   SAC low-level action proposal
#          +
#   waypoint steering correction
#          ↓
#   robot
#
# SAC is still used every control step.
# A* does NOT directly teleport/move the robot; it only provides
# a collision-free waypoint direction.
#
# Six real URDF obstacles are widely randomized every episode.
# ============================================================

MODEL_PATH = (
    "models_general_fresh/"
    "best_general_sac.zip"
)

DEMO_EPISODES = 10

# Re-plan periodically because the robot may deviate from a path.
REPLAN_EVERY = 20

# Look several grid cells ahead so the robot does not chase every
# tiny A* cell.
WAYPOINT_LOOKAHEAD_CELLS = 5

# Blending:
# SAC proposes steering, planner makes the larger contribution.
SAC_TURN_WEIGHT = 0.25
PLANNER_TURN_WEIGHT = 0.75

# Safe path-following speed in action-space [-1, 1].
CRUISE_ACTION = 0.42
SLOW_ACTION = 0.20

# If heading error is very large, rotate before driving forward.
TURN_IN_PLACE_DEG = 55.0

# Consider a waypoint reached within this distance.
WAYPOINT_REACHED = 0.55


# ============================================================
# LOCAL DEADLOCK / TWO-OBSTACLE RECOVERY
# ============================================================
#
# Problem:
#   Between two nearby obstacles, SAC and waypoint correction can
#   alternate left/right. The robot then makes almost no forward
#   progress even though a route exists.
#
# Fix:
#   1. Detect sustained lack of progress.
#   2. Compare LiDAR clearance on LEFT vs RIGHT.
#   3. LOCK one escape direction temporarily.
#   4. If front is tightly blocked, reverse a little first.
#   5. Turn + commit forward.
#   6. Return control to normal SAC+A*.
#
# No training is performed.
# ============================================================

DEADLOCK_WINDOW = 45
DEADLOCK_MIN_PROGRESS = 0.10
DEADLOCK_MIN_TARGET_DISTANCE = 1.20

RECOVERY_REVERSE_STEPS = 9
RECOVERY_TURN_STEPS = 13
RECOVERY_FORWARD_STEPS = 24
RECOVERY_COOLDOWN_STEPS = 70

RECOVERY_REVERSE_ACTION = -0.16
RECOVERY_TURN_ACTION = 0.78
RECOVERY_FORWARD_ACTION = 0.34
RECOVERY_FORWARD_TURN = 0.42


# ============================================================
# PERSISTENT EXPERIENCE RECORDING
# ============================================================
#
# Demo remains inference-only: NO model weights are changed here.
#
# Every executed transition is saved:
#
#   observation_t
#   raw SAC action
#   actually executed hybrid action
#   reward
#   observation_t+1
#   terminated / truncated
#   robot x, y, yaw
#   task outcome information
#
# A persistent SB3 ReplayBuffer is also saved to disk so these
# transitions can later be loaded for SAC fine-tuning.
# ============================================================

EXPERIENCE_DIR = "experience_logs_final_dual_v5_deadlock_recovery"

REPLAY_BUFFER_PATH = os.path.join(
    EXPERIENCE_DIR,
    "delivery_replay_buffer.pkl"
)

EPISODE_INDEX_PATH = os.path.join(
    EXPERIENCE_DIR,
    "episode_index.csv"
)

REPLAY_BUFFER_SIZE = 100_000


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing trained SAC model: {MODEL_PATH}"
    )


# ============================================================
# GENESIS / ENVIRONMENT
# ============================================================

gs.init(
    backend=gs.cpu
)

env = DeliveryRobotEnv(
    render=True,
    task_mode="showcase_dual_delivery"
)

# Longer route + four obstacles. This changes only the maximum
# demo duration; it does not change the learned model or controller.
env.max_steps = 3800

model = SAC.load(
    MODEL_PATH,
    env=env,
    device="cpu"
)


# ============================================================
# LOAD / CREATE PERSISTENT REPLAY BUFFER
# ============================================================

os.makedirs(
    EXPERIENCE_DIR,
    exist_ok=True
)


if os.path.exists(
    REPLAY_BUFFER_PATH
):

    try:

        model.load_replay_buffer(
            REPLAY_BUFFER_PATH,
            truncate_last_traj=False
        )

        print()
        print(
            "[EXPERIENCE] Existing replay buffer loaded."
        )
        print(
            "[EXPERIENCE] Stored transitions:",
            model.replay_buffer.size()
        )

    except Exception as exc:

        print()
        print(
            "[EXPERIENCE] Existing replay buffer could not be loaded:"
        )
        print(
            exc
        )
        print(
            "[EXPERIENCE] Creating a fresh replay buffer."
        )

        model.replay_buffer = ReplayBuffer(
            buffer_size=REPLAY_BUFFER_SIZE,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=model.device,
            n_envs=1,
            optimize_memory_usage=False,
            handle_timeout_termination=True
        )

else:

    model.replay_buffer = ReplayBuffer(
        buffer_size=REPLAY_BUFFER_SIZE,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=model.device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True
    )

    print()
    print(
        "[EXPERIENCE] Fresh replay buffer created."
    )


# ============================================================
# EXPERIENCE SAVE HELPERS
# ============================================================

def _append_episode_index(
    filename,
    seed,
    steps,
    total_reward,
    outcome,
    package_picked,
    delivered,
    collision,
    timeout,
    out_of_bounds,
    manual_reset
):

    new_file = not os.path.exists(
        EPISODE_INDEX_PATH
    )

    with open(
        EPISODE_INDEX_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.writer(
            handle
        )

        if new_file:

            writer.writerow(
                [
                    "timestamp",
                    "filename",
                    "seed",
                    "steps",
                    "total_reward",
                    "outcome",
                    "package_picked",
                    "delivered",
                    "collision",
                    "timeout",
                    "out_of_bounds",
                    "manual_reset"
                ]
            )

        writer.writerow(
            [
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                filename,
                int(seed),
                int(steps),
                float(total_reward),
                outcome,
                int(bool(package_picked)),
                int(bool(delivered)),
                int(bool(collision)),
                int(bool(timeout)),
                int(bool(out_of_bounds)),
                int(bool(manual_reset))
            ]
        )


def save_episode_experience(
    episode_data,
    seed,
    attempt_number,
    outcome,
    package_picked=False,
    delivered=False,
    collision=False,
    timeout=False,
    out_of_bounds=False,
    manual_reset=False,
    obstacle_layout=None
):

    if len(
        episode_data["observations"]
    ) == 0:

        print(
            "[EXPERIENCE] No transitions to save for this attempt."
        )
        return


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"episode_{timestamp}_"
        f"attempt_{attempt_number:03d}.npz"
    )

    path = os.path.join(
        EXPERIENCE_DIR,
        filename
    )


    np.savez_compressed(
        path,

        observations=np.asarray(
            episode_data["observations"],
            dtype=np.float32
        ),

        sac_actions=np.asarray(
            episode_data["sac_actions"],
            dtype=np.float32
        ),

        executed_actions=np.asarray(
            episode_data["executed_actions"],
            dtype=np.float32
        ),

        rewards=np.asarray(
            episode_data["rewards"],
            dtype=np.float32
        ),

        next_observations=np.asarray(
            episode_data["next_observations"],
            dtype=np.float32
        ),

        terminated=np.asarray(
            episode_data["terminated"],
            dtype=np.bool_
        ),

        truncated=np.asarray(
            episode_data["truncated"],
            dtype=np.bool_
        ),

        dones=np.asarray(
            episode_data["dones"],
            dtype=np.bool_
        ),

        robot_pose=np.asarray(
            episode_data["robot_pose"],
            dtype=np.float32
        ),

        package_picked_step=np.asarray(
            episode_data["package_picked_step"],
            dtype=np.bool_
        ),

        delivery_a_step=np.asarray(
            episode_data["delivery_a_step"],
            dtype=np.bool_
        ),

        delivered_step=np.asarray(
            episode_data["delivered_step"],
            dtype=np.bool_
        ),

        collision_step=np.asarray(
            episode_data["collision_step"],
            dtype=np.bool_
        ),

        seed=np.asarray(
            [int(seed)],
            dtype=np.int64
        ),

        obstacle_layout=np.asarray(
            obstacle_layout
            if obstacle_layout is not None
            else [],
            dtype=np.float32
        )
    )


    total_reward = float(
        np.sum(
            np.asarray(
                episode_data["rewards"],
                dtype=np.float32
            )
        )
    )


    _append_episode_index(
        filename=filename,
        seed=seed,
        steps=len(
            episode_data["rewards"]
        ),
        total_reward=total_reward,
        outcome=outcome,
        package_picked=package_picked,
        delivered=delivered,
        collision=collision,
        timeout=timeout,
        out_of_bounds=out_of_bounds,
        manual_reset=manual_reset
    )


    # Save the persistent SB3 replay buffer separately.
    model.save_replay_buffer(
        REPLAY_BUFFER_PATH
    )


    print()
    print(
        "[EXPERIENCE] Episode saved:",
        path
    )
    print(
        "[EXPERIENCE] Steps saved:",
        len(
            episode_data["rewards"]
        )
    )
    print(
        "[EXPERIENCE] Total reward:",
        f"{total_reward:.2f}"
    )
    print(
        "[EXPERIENCE] Replay transitions:",
        model.replay_buffer.size()
    )


# ============================================================
# GRID HELPERS
# ============================================================

def grid_to_world(node):
    world_min = -float(
        env.WORLD_LIMIT
    )

    resolution = float(
        env.PATH_GRID_RESOLUTION
    )

    gx, gy = node

    return np.array(
        [
            world_min
            +
            gx * resolution,

            world_min
            +
            gy * resolution
        ],
        dtype=np.float32
    )


def heuristic(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def astar_path(
    occupancy,
    start_xy,
    goal_xy
):
    """
    Return a collision-free A* grid path.

    The environment already inflates obstacles by robot radius +
    safety margin when building occupancy, so this path includes
    clearance for the physical robot.
    """

    start = env._path_world_to_grid(
        float(start_xy[0]),
        float(start_xy[1])
    )

    goal = env._path_world_to_grid(
        float(goal_xy[0]),
        float(goal_xy[1])
    )

    nx_max = occupancy.shape[0]
    ny_max = occupancy.shape[1]


    def valid(node):
        gx, gy = node

        return (
            0 <= gx < nx_max
            and
            0 <= gy < ny_max
            and
            not occupancy[gx, gy]
        )


    if not valid(start):
        return None

    if not valid(goal):
        return None


    neighbours = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1,  1, math.sqrt(2.0)),
        ( 1, -1, math.sqrt(2.0)),
        ( 1,  1, math.sqrt(2.0)),
    ]


    open_heap = []

    heapq.heappush(
        open_heap,
        (
            heuristic(
                start,
                goal
            ),
            0.0,
            start
        )
    )

    cost_so_far = {
        start: 0.0
    }

    came_from = {
        start: None
    }


    while open_heap:

        _, current_cost, current = heapq.heappop(
            open_heap
        )

        if current == goal:
            break


        if (
            current_cost
            >
            cost_so_far.get(
                current,
                float("inf")
            )
        ):
            continue


        cx, cy = current


        for dx, dy, step_cost in neighbours:

            nxt = (
                cx + dx,
                cy + dy
            )


            if not valid(nxt):
                continue


            # Prevent diagonal corner cutting.
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


            if (
                new_cost
                <
                cost_so_far.get(
                    nxt,
                    float("inf")
                )
            ):

                cost_so_far[nxt] = (
                    new_cost
                )

                came_from[nxt] = (
                    current
                )

                priority = (
                    new_cost
                    +
                    heuristic(
                        nxt,
                        goal
                    )
                )

                heapq.heappush(
                    open_heap,
                    (
                        priority,
                        new_cost,
                        nxt
                    )
                )


    if goal not in came_from:
        return None


    path = []

    current = goal

    while current is not None:
        path.append(
            current
        )

        current = came_from[
            current
        ]


    path.reverse()

    return path


# ============================================================
# CONTROL HELPERS
# ============================================================

def normalize_angle(angle):

    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def current_active_target():

    if env.package_picked:
        return np.asarray(
            env.goal_position,
            dtype=np.float32
        )

    return np.asarray(
        env.package_position,
        dtype=np.float32
    )


def compute_path():

    robot_xy = np.asarray(
        env._get_robot_position(),
        dtype=np.float32
    )

    target_xy = (
        current_active_target()
    )

    occupancy = (
        env._build_path_occupancy_grid(
            env.obstacle_data
        )
    )

    return astar_path(
        occupancy,
        robot_xy,
        target_xy
    )


def choose_waypoint(
    path,
    robot_xy
):

    if not path:
        return current_active_target()


    world_points = [
        grid_to_world(node)
        for node in path
    ]


    # Find the nearest path point to the robot.
    distances = [
        float(
            np.linalg.norm(
                point
                -
                robot_xy
            )
        )
        for point
        in world_points
    ]

    nearest_index = int(
        np.argmin(
            distances
        )
    )

    target_index = min(
        nearest_index
        +
        WAYPOINT_LOOKAHEAD_CELLS,
        len(world_points) - 1
    )

    return world_points[
        target_index
    ]


def hybrid_action(
    obs,
    waypoint,
    return_sac_action=False
):
    """
    SAC provides its learned low-level proposal.
    A* waypoint provides the global steering direction.
    """

    sac_action, _ = model.predict(
        obs,
        deterministic=True
    )

    sac_action = np.asarray(
        sac_action,
        dtype=np.float32
    )


    robot_xy = np.asarray(
        env._get_robot_position(),
        dtype=np.float32
    )

    yaw = float(
        env._get_robot_yaw()
    )


    dx = float(
        waypoint[0]
        -
        robot_xy[0]
    )

    dy = float(
        waypoint[1]
        -
        robot_xy[1]
    )

    desired_heading = math.atan2(
        dy,
        dx
    )

    error = normalize_angle(
        desired_heading
        -
        yaw
    )

    error_deg = abs(
        math.degrees(
            error
        )
    )


    # Map heading error to angular action [-1, 1].
    planner_turn = float(
        np.clip(
            1.35
            *
            error,
            -1.0,
            1.0
        )
    )


    # Planner dominates direction, SAC still contributes.
    turn_action = float(
        np.clip(
            PLANNER_TURN_WEIGHT
            *
            planner_turn
            +
            SAC_TURN_WEIGHT
            *
            float(
                sac_action[1]
            ),
            -1.0,
            1.0
        )
    )


    # ========================================================
    # SPEED
    # ========================================================

    if (
        error_deg
        >
        TURN_IN_PLACE_DEG
    ):

        linear_action = 0.0

    elif (
        error_deg
        >
        28.0
    ):

        linear_action = (
            SLOW_ACTION
        )

    else:

        # Use SAC's speed if it is sensible; otherwise provide
        # moderate cruise motion so the old deadlock cannot hold
        # the hybrid robot stationary.
        sac_linear = float(
            sac_action[0]
        )

        linear_action = max(
            CRUISE_ACTION,
            sac_linear
        )

        linear_action = float(
            np.clip(
                linear_action,
                0.0,
                0.65
            )
        )


    # ========================================================
    # SMOOTHER LIDAR-BASED SPEED CONTROL
    # ========================================================
    #
    # Before:
    #   nearest < 1.15 m -> linear_action <= 0.10
    #
    # That was visually too slow.
    #
    # Now:
    #   very close obstacle   -> slow, but not crawling
    #   moderately close      -> medium-slow
    #   somewhat close        -> slightly reduced
    #
    # This keeps the robot safe while looking more natural in
    # the viewer.
    # ========================================================

    lidar_m = (
        np.asarray(
            env._get_lidar(),
            dtype=np.float32
        )
        *
        float(
            env.LIDAR_MAX_DISTANCE
        )
    )

    # Only obstacles in the FRONT cone should reduce forward
    # speed. A side/behind obstacle must not make the robot crawl.
    #
    # With 16 rays, front wraps around ray 0.
    front_indices = [
        14,
        15,
        0,
        1,
        2
    ]

    front_min = float(
        np.min(
            lidar_m[
                front_indices
            ]
        )
    )

    if front_min < 0.82:

        linear_action = min(
            linear_action,
            0.14
        )

    elif front_min < 1.05:

        linear_action = min(
            linear_action,
            0.22
        )

    elif front_min < 1.35:

        linear_action = min(
            linear_action,
            0.31
        )

    elif front_min < 1.65:

        linear_action = min(
            linear_action,
            0.39
        )

    # If front is actually clear, do not let a side obstacle
    # unnecessarily hold the robot at near-zero speed.
    if (
        front_min > 1.65
        and
        error_deg < 35.0
    ):

        linear_action = max(
            linear_action,
            0.34
        )


    executed_action = np.array(
        [
            linear_action,
            turn_action
        ],
        dtype=np.float32
    )


    if return_sac_action:

        return (
            executed_action,
            sac_action.copy()
        )


    return executed_action



# ============================================================
# LOCAL GAP / DEADLOCK RECOVERY HELPERS
# ============================================================

def lidar_clearance_summary():
    """
    Return robust front / left / right LiDAR clearance values.

    We use a percentile/median-like statistic for the side regions
    rather than one single ray, so one unusually long ray does not
    incorrectly look like a safe escape corridor.
    """

    lidar_m = (
        np.asarray(
            env._get_lidar(),
            dtype=np.float32
        )
        *
        float(
            env.LIDAR_MAX_DISTANCE
        )
    )

    front = lidar_m[
        [
            14,
            15,
            0,
            1,
            2
        ]
    ]

    # Approximate left/right hemispheres around the front.
    left = lidar_m[
        [
            2,
            3,
            4,
            5,
            6
        ]
    ]

    right = lidar_m[
        [
            10,
            11,
            12,
            13,
            14
        ]
    ]

    front_min = float(
        np.min(
            front
        )
    )

    left_clear = float(
        np.percentile(
            left,
            40
        )
    )

    right_clear = float(
        np.percentile(
            right,
            40
        )
    )

    return (
        front_min,
        left_clear,
        right_clear
    )


def choose_escape_direction():
    """
    +1 = turn left
    -1 = turn right

    Direction is chosen ONCE at recovery start and then locked.
    This prevents the left-right-left-right oscillation that happens
    between two obstacles.
    """

    (
        front_min,
        left_clear,
        right_clear
    ) = lidar_clearance_summary()

    if left_clear >= right_clear:

        direction = 1.0

    else:

        direction = -1.0

    return (
        direction,
        front_min,
        left_clear,
        right_clear
    )


def recovery_action(
    phase,
    direction
):
    """
    Temporary deterministic escape maneuver.

    Steering direction remains locked for the whole recovery.
    """

    direction = float(
        direction
    )

    if phase == "reverse":

        return np.array(
            [
                RECOVERY_REVERSE_ACTION,
                direction
                *
                0.46
            ],
            dtype=np.float32
        )

    if phase == "turn":

        return np.array(
            [
                0.05,
                direction
                *
                RECOVERY_TURN_ACTION
            ],
            dtype=np.float32
        )

    # forward commitment
    return np.array(
        [
            RECOVERY_FORWARD_ACTION,
            direction
            *
            RECOVERY_FORWARD_TURN
        ],
        dtype=np.float32
    )


# ============================================================
# DEMO
# ============================================================

print()
print("=" * 94)
print("FINAL TWO-STOP AUTONOMOUS DELIVERY ROBOT")
print("=" * 94)
print("Architecture : SAC low-level controller + A* waypoint planner")
print("Start        : visible START gate")
print("Pickup       : TWO parcels from one yellow pickup zone")
print("Delivery A   : first parcel drop")
print("Delivery B   : second parcel drop; same gate style")
print("Route spacing : START->A and A->B are equal-length sections")
print("Dropped boxes : hidden after delivery; cannot block robot")
print("Collision     : precise robot footprint, not 0.80m circle")
print("Deadlock      : locked left/right escape recovery enabled")
print("Speed         : front-cone slowdown only; side obstacles do not freeze robot")
print("Obstacles    : 4 open random before A + 4 open random before B")
print("LiDAR        : 16 rays")
print("Training     : NONE — experience recording only")
print("Viewer       : ON")
print("Speed mode   : smoother obstacle-near motion")
print("Episode limit: 3800 steps")
print("Experience   : persistent replay buffer + per-episode NPZ logs")
print()
print("MANUAL CONTROLS (CMD WINDOW):")
print("  N = NEW episode: reset current episode and generate a NEW random layout")
print("  Q = quit demo immediately")
print("=" * 94)


successes = 0
collisions = 0
timeouts = 0
oob_count = 0
manual_resets = 0

completed_episodes = 0
attempt_number = 0
quit_requested = False


while (
    completed_episodes < DEMO_EPISODES
    and
    not quit_requested
):

    attempt_number += 1

    seed = random.SystemRandom().randint(
        100000,
        999999
    )

    obs, info = env.reset(
        seed=seed
    )

    active_obstacles = [
        obstacle
        for obstacle
        in env.obstacle_data
        if abs(
            float(
                obstacle[0]
            )
        ) < 20.0
    ]

    print()
    print("-" * 94)
    print(
        f"EPISODE {completed_episodes + 1}/{DEMO_EPISODES}"
        f" | attempt={attempt_number}"
    )

    for i, obstacle in enumerate(
        active_obstacles,
        start=1
    ):

        print(
            f"Obstacle {i}: "
            f"({float(obstacle[0]):.2f}, "
            f"{float(obstacle[1]):.2f})"
        )

    print()
    print(
        "If robot gets stuck: click CMD window and press N."
    )
    print("-" * 94)


    terminated = False
    truncated = False
    steps = 0

    path = None
    previous_target_state = (
        bool(
            env.package_picked
        ),
        bool(
            env.delivery_a_complete
        )
    )

    announced_pickup = False
    announced_a = False

    reset_requested = False


    # ========================================================
    # DEADLOCK RECOVERY STATE
    # ========================================================

    progress_history = deque(
        maxlen=DEADLOCK_WINDOW
    )

    recovery_phase = None
    recovery_steps_left = 0
    recovery_direction = 1.0
    recovery_cooldown = 0


    # ========================================================
    # CURRENT EPISODE EXPERIENCE
    # ========================================================

    episode_data = {
        "observations": [],
        "sac_actions": [],
        "executed_actions": [],
        "rewards": [],
        "next_observations": [],
        "terminated": [],
        "truncated": [],
        "dones": [],
        "robot_pose": [],
        "package_picked_step": [],
        "delivery_a_step": [],
        "delivered_step": [],
        "collision_step": []
    }


    while not (
        terminated
        or
        truncated
    ):

        # --------------------------------------------------------
        # MANUAL DEMO CONTROL
        # --------------------------------------------------------
        #
        # msvcrt is built into Windows Python and does not require
        # any additional package.
        #
        # IMPORTANT:
        # We use N instead of R because Genesis viewer already
        # uses R for video recording.
        # --------------------------------------------------------

        if msvcrt.kbhit():

            key = (
                msvcrt.getwch()
                .lower()
            )

            if key == "n":

                reset_requested = True
                manual_resets += 1

                print()
                print(
                    "[MANUAL RESET] Current episode aborted."
                )
                print(
                    "Generating a new randomized obstacle layout..."
                )

                break

            elif key == "q":

                quit_requested = True

                print()
                print(
                    "[QUIT] Demo stopped by user."
                )

                break


        # Replan periodically and immediately when package becomes
        # picked and the active target changes to the goal.
        current_target_state = (
            bool(
                env.package_picked
            ),
            bool(
                env.delivery_a_complete
            )
        )

        target_switched = (
            current_target_state
            !=
            previous_target_state
        )


        # Friendly mission-status messages.
        if (
            env.package_picked
            and
            not announced_pickup
        ):

            announced_pickup = True

            print()
            print(
                "[PICKUP COMPLETE] Both parcels collected."
            )
            print(
                "Next target: DELIVERY A"
            )


        if (
            env.delivery_a_complete
            and
            not announced_a
        ):

            announced_a = True

            print()
            print(
                "[DELIVERY A COMPLETE] Parcel A delivered."
            )
            print(
                "Robot continues directly to DELIVERY B."
            )


        if (
            path is None
            or
            steps % REPLAN_EVERY == 0
            or
            target_switched
        ):

            path = compute_path()

            previous_target_state = (
                current_target_state
            )

            # A new pickup/delivery target invalidates the previous
            # progress history.
            if target_switched:

                progress_history.clear()


        # Preserve the CURRENT observation before the step.
        transition_obs = np.asarray(
            obs,
            dtype=np.float32
        ).copy()


        # --------------------------------------------------------
        # ALWAYS obtain the raw SAC proposal for logging.
        # --------------------------------------------------------

        sac_action, _ = model.predict(
            transition_obs,
            deterministic=True
        )

        sac_action = np.asarray(
            sac_action,
            dtype=np.float32
        )


        # --------------------------------------------------------
        # TRACK TARGET PROGRESS
        # --------------------------------------------------------

        target_distance, _ = (
            env._get_target_information()
        )

        progress_history.append(
            float(
                target_distance
            )
        )

        if recovery_cooldown > 0:

            recovery_cooldown -= 1


        # --------------------------------------------------------
        # START A LOCKED-DIRECTION RECOVERY IF STALLED
        # --------------------------------------------------------

        if (
            recovery_phase is None
            and
            recovery_cooldown <= 0
            and
            len(
                progress_history
            )
            ==
            DEADLOCK_WINDOW
            and
            float(
                target_distance
            )
            >
            DEADLOCK_MIN_TARGET_DISTANCE
        ):

            progress = (
                float(
                    progress_history[0]
                )
                -
                float(
                    progress_history[-1]
                )
            )

            if progress < DEADLOCK_MIN_PROGRESS:

                (
                    recovery_direction,
                    front_min,
                    left_clear,
                    right_clear
                ) = choose_escape_direction()


                # If the robot is tightly boxed in front, create
                # room first. Otherwise rotate directly.
                if front_min < 1.05:

                    recovery_phase = "reverse"
                    recovery_steps_left = (
                        RECOVERY_REVERSE_STEPS
                    )

                else:

                    recovery_phase = "turn"
                    recovery_steps_left = (
                        RECOVERY_TURN_STEPS
                    )


                recovery_cooldown = (
                    RECOVERY_COOLDOWN_STEPS
                )

                progress_history.clear()

                print()
                print(
                    "[DEADLOCK RECOVERY] "
                    f"progress={progress:.3f} m"
                )
                print(
                    "  escape side :",
                    "LEFT"
                    if recovery_direction > 0
                    else "RIGHT"
                )
                print(
                    "  front/left/right clearance:",
                    f"{front_min:.2f} / "
                    f"{left_clear:.2f} / "
                    f"{right_clear:.2f} m"
                )


        # --------------------------------------------------------
        # RECOVERY MODE
        # --------------------------------------------------------

        if recovery_phase is not None:

            action = recovery_action(
                recovery_phase,
                recovery_direction
            )

            recovery_steps_left -= 1


            if recovery_steps_left <= 0:

                if recovery_phase == "reverse":

                    recovery_phase = "turn"
                    recovery_steps_left = (
                        RECOVERY_TURN_STEPS
                    )

                elif recovery_phase == "turn":

                    recovery_phase = "forward"
                    recovery_steps_left = (
                        RECOVERY_FORWARD_STEPS
                    )

                else:

                    recovery_phase = None
                    recovery_steps_left = 0

                    # Force a fresh A* route after escaping.
                    path = compute_path()

                    progress_history.clear()

                    print(
                        "[DEADLOCK RECOVERY] complete -> "
                        "control returned to SAC + A*"
                    )


        # --------------------------------------------------------
        # NORMAL SAC + A* MODE
        # --------------------------------------------------------

        elif path is None:

            action = sac_action.copy()

        else:

            robot_xy = np.asarray(
                env._get_robot_position(),
                dtype=np.float32
            )

            waypoint = choose_waypoint(
                path,
                robot_xy
            )

            # hybrid_action calls model.predict internally, but we
            # already have the raw SAC action above. Keep existing
            # behavior unchanged for normal navigation.
            action = hybrid_action(
                transition_obs,
                waypoint
            )


        next_obs, reward, terminated, truncated, info = env.step(
            action
        )


        done = bool(
            terminated
            or
            truncated
        )


        # --------------------------------------------------------
        # ADD TRANSITION TO PERSISTENT SAC REPLAY BUFFER
        # --------------------------------------------------------
        #
        # SAC is off-policy, so later fine-tuning can learn from
        # the ACTUALLY EXECUTED hybrid action.
        #
        # Raw SAC action is also stored separately in the NPZ log.
        # --------------------------------------------------------

        replay_info = dict(
            info
        )

        replay_info[
            "TimeLimit.truncated"
        ] = bool(
            truncated
            and
            not terminated
        )


        model.replay_buffer.add(
            transition_obs[
                None,
                :
            ],

            np.asarray(
                next_obs,
                dtype=np.float32
            )[
                None,
                :
            ],

            np.asarray(
                action,
                dtype=np.float32
            )[
                None,
                :
            ],

            np.asarray(
                [reward],
                dtype=np.float32
            ),

            np.asarray(
                [done],
                dtype=np.float32
            ),

            [
                replay_info
            ]
        )


        # --------------------------------------------------------
        # HUMAN-READABLE / ANALYSIS-FRIENDLY EPISODE LOG
        # --------------------------------------------------------

        robot_position = np.asarray(
            env._get_robot_position(),
            dtype=np.float32
        )

        robot_yaw = float(
            env._get_robot_yaw()
        )


        episode_data[
            "observations"
        ].append(
            transition_obs
        )

        episode_data[
            "sac_actions"
        ].append(
            np.asarray(
                sac_action,
                dtype=np.float32
            ).copy()
        )

        episode_data[
            "executed_actions"
        ].append(
            np.asarray(
                action,
                dtype=np.float32
            ).copy()
        )

        episode_data[
            "rewards"
        ].append(
            float(
                reward
            )
        )

        episode_data[
            "next_observations"
        ].append(
            np.asarray(
                next_obs,
                dtype=np.float32
            ).copy()
        )

        episode_data[
            "terminated"
        ].append(
            bool(
                terminated
            )
        )

        episode_data[
            "truncated"
        ].append(
            bool(
                truncated
            )
        )

        episode_data[
            "dones"
        ].append(
            done
        )

        episode_data[
            "robot_pose"
        ].append(
            [
                float(
                    robot_position[0]
                ),
                float(
                    robot_position[1]
                ),
                robot_yaw
            ]
        )

        episode_data[
            "package_picked_step"
        ].append(
            bool(
                info.get(
                    "package_picked",
                    False
                )
            )
        )

        episode_data[
            "delivery_a_step"
        ].append(
            bool(
                info.get(
                    "delivery_a_complete",
                    False
                )
            )
        )

        episode_data[
            "delivered_step"
        ].append(
            bool(
                info.get(
                    "package_delivered",
                    False
                )
            )
        )

        episode_data[
            "collision_step"
        ].append(
            bool(
                info.get(
                    "collision",
                    False
                )
            )
        )


        obs = next_obs

        steps += 1


    if quit_requested:

        save_episode_experience(
            episode_data=episode_data,
            seed=seed,
            attempt_number=attempt_number,
            outcome="QUIT_PARTIAL",
            package_picked=bool(
                episode_data[
                    "package_picked_step"
                ][-1]
            )
            if episode_data[
                "package_picked_step"
            ]
            else False,
            manual_reset=False,
            obstacle_layout=active_obstacles
        )

        break


    # Manual reset does NOT count as a completed/failing episode,
    # but its transitions ARE still valuable experience.
    if reset_requested:

        save_episode_experience(
            episode_data=episode_data,
            seed=seed,
            attempt_number=attempt_number,
            outcome="MANUAL_RESET",
            package_picked=bool(
                episode_data[
                    "package_picked_step"
                ][-1]
            )
            if episode_data[
                "package_picked_step"
            ]
            else False,
            manual_reset=True,
            obstacle_layout=active_obstacles
        )

        continue


    completed_episodes += 1


    delivered = bool(
        info.get(
            "package_delivered",
            False
        )
    )

    collided = bool(
        info.get(
            "collision",
            False
        )
    )

    out_of_bounds = bool(
        info.get(
            "out_of_bounds",
            False
        )
    )

    timeout = bool(
        truncated
        and
        not delivered
        and
        not collided
        and
        not out_of_bounds
    )


    if delivered:

        outcome = "SUCCESS"
        successes += 1

    elif collided:

        outcome = "COLLISION"
        collisions += 1

    elif out_of_bounds:

        outcome = "OOB"
        oob_count += 1

    elif timeout:

        outcome = "TIMEOUT"
        timeouts += 1

    else:

        outcome = "OTHER"


    save_episode_experience(
        episode_data=episode_data,
        seed=seed,
        attempt_number=attempt_number,
        outcome=outcome,
        package_picked=bool(
            info.get(
                "package_picked",
                False
            )
        ),
        delivered=delivered,
        collision=collided,
        timeout=timeout,
        out_of_bounds=out_of_bounds,
        manual_reset=False,
        obstacle_layout=active_obstacles
    )


    print()
    print(
        "Outcome       :",
        outcome
    )

    print(
        "Both picked   :",
        bool(
            info.get(
                "package_picked",
                False
            )
        )
    )

    print(
        "Delivery A    :",
        bool(
            info.get(
                "delivery_a_complete",
                False
            )
        )
    )

    print(
        "Delivery B    :",
        bool(
            info.get(
                "delivery_b_complete",
                False
            )
        )
    )

    print(
        "Steps         :",
        steps
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 94)
print("FINAL SHOWCASE V4 SUMMARY")
print("=" * 94)

print(
    "Completed episodes :",
    completed_episodes
)

print(
    "Manual resets      :",
    manual_resets
)

if completed_episodes > 0:

    print(
        "Success            :",
        f"{successes}/{completed_episodes}",
        f"({100.0 * successes / completed_episodes:.1f}%)"
    )

    print(
        "Collision          :",
        f"{collisions}/{completed_episodes}",
        f"({100.0 * collisions / completed_episodes:.1f}%)"
    )

    print(
        "Timeout            :",
        f"{timeouts}/{completed_episodes}",
        f"({100.0 * timeouts / completed_episodes:.1f}%)"
    )

    print(
        "OOB                :",
        f"{oob_count}/{completed_episodes}",
        f"({100.0 * oob_count / completed_episodes:.1f}%)"
    )

print()
print("No training occurred.")
print("Experiences WERE recorded for later SAC fine-tuning.")
print(
    "Replay buffer       :",
    REPLAY_BUFFER_PATH
)
print(
    "Stored transitions  :",
    model.replay_buffer.size()
)
print(
    "Episode logs        :",
    EXPERIENCE_DIR
)
print("=" * 94)

env.close()
