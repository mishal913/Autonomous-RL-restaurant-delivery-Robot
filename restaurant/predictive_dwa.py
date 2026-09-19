from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class DWAPlannerDecision:
    action: np.ndarray
    intervened: bool
    reason: str
    predicted_dynamic_clearance: float
    predicted_static_clearance: float
    predicted_ttc: float


@dataclass
class _Rollout:
    action: np.ndarray
    safe: bool
    score: float
    dynamic_clearance: float
    static_clearance: float
    collision_time: float
    collision_kind: str | None
    final_distance: float


class PredictiveDWAPlanner:
    """Short-horizon local planner for moving people and restaurant furniture.

    The SAC+A* command is treated as the preferred velocity. On a moving-person
    collision course, the person slows and shifts left while the robot makes a
    short straight reverse followed by a bounded forward-left pass.
    """

    HORIZON_SECONDS = 1.60
    CONTROL_DT = 0.04
    FINE_HORIZON_SECONDS = 0.24
    FINE_ROLLOUT_DT = 0.01
    ROLLOUT_DT = 0.08
    EMERGENCY_DYNAMIC_CLEARANCE = 0.85
    DYNAMIC_CLEARANCE_MARGIN = 0.08
    STATIC_CLEARANCE_MARGIN = 0.03
    # Dynamic encounters get only a small straight nudge backwards.  The
    # old 0.18 m budget plus high-turn fallback made the robot appear to spin
    # in place instead of passing the person.
    REVERSE_LIMIT_METRES = 0.08
    CLEAR_STEPS_TO_REARM = 12
    COOPERATIVE_REVERSE_STEPS = 6
    COOPERATIVE_LEFT_STEPS = 12

    # Rear-approach protection. A person coming from behind must never cause
    # the robot to reverse into them. The robot instead prefers a collision-
    # checked forward/forward-arc escape while there is usable space ahead.
    REAR_TRIGGER_TTC = 2.20
    REAR_LATERAL_MARGIN = 0.24

    LINEAR_SAMPLES = (0.18, 0.38, 0.58, 0.70)
    # Keep all generic arcs bounded; a person encounter must never become a
    # near-360-degree recovery turn.
    TURN_SAMPLES = (-0.45, -0.28, -0.14, 0.0, 0.14, 0.28, 0.45)

    def __init__(self) -> None:
        print(
            "[Cooperative Left-Pass] active | person slow+left | "
            "robot short-reverse then forward-left"
        )
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(2, dtype=np.float32)
        self.local_phase = "nominal"
        self.intervention_events = 0
        self.reason_counts: dict[str, int] = defaultdict(int)
        self._intervening = False
        self._avoidance_active = False
        self._preferred_turn_sign = 0.0
        self._clear_steps = self.CLEAR_STEPS_TO_REARM
        self._reverse_used = 0.0
        self._cooperative_phase = "idle"
        self._cooperative_phase_steps = 0

    @staticmethod
    def _normalise_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _point_rectangle_clearance(point: np.ndarray, row) -> float:
        x, y, sx, sy, _ = row
        dx = max(abs(float(point[0]) - float(x)) - float(sx) / 2.0, 0.0)
        dy = max(abs(float(point[1]) - float(y)) - float(sy) / 2.0, 0.0)
        return math.hypot(dx, dy)

    @staticmethod
    def _pose_hits_box(
        xy: np.ndarray,
        yaw: float,
        row,
        half_length: float,
        half_width: float,
    ) -> bool:
        x, y, sx, sy, _ = row
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        delta = np.array([float(x), float(y)], dtype=np.float64) - xy
        obstacle_half_x = float(sx) / 2.0
        obstacle_half_y = float(sy) / 2.0

        robot_x = half_length * abs(forward[0]) + half_width * abs(left[0])
        robot_y = half_length * abs(forward[1]) + half_width * abs(left[1])
        if abs(delta[0]) > robot_x + obstacle_half_x:
            return False
        if abs(delta[1]) > robot_y + obstacle_half_y:
            return False

        obstacle_forward = (
            obstacle_half_x * abs(forward[0])
            + obstacle_half_y * abs(forward[1])
        )
        obstacle_left = (
            obstacle_half_x * abs(left[0])
            + obstacle_half_y * abs(left[1])
        )
        if abs(float(np.dot(delta, forward))) > half_length + obstacle_forward:
            return False
        if abs(float(np.dot(delta, left))) > half_width + obstacle_left:
            return False
        return True

    @staticmethod
    def _person_at(state: dict, future_time: float) -> np.ndarray:
        """Predict slowed route motion and the current cooperative side shift."""
        start = np.asarray(state["start"], dtype=np.float64)
        end = np.asarray(state["end"], dtype=np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 1e-8:
            return start.copy()
        direction = vector / length
        state_time = float(state.get("state_time", 0.0))
        elapsed = max(0.0, float(future_time) - state_time)
        travelled = (
            float(state.get("progress_at_pose", state["offset"]))
            + elapsed * float(state.get("current_speed", state["speed"]))
        ) % (2.0 * length)
        if travelled <= length:
            xy = start + direction * travelled
            heading = direction
        else:
            xy = end - direction * (travelled - length)
            heading = -direction
        person_left = np.array([-heading[1], heading[0]], dtype=np.float64)
        current_offset = float(state.get("avoidance_offset", 0.0))
        target_offset = float(
            state.get("target_avoidance_offset", current_offset)
        )
        predicted_offset = current_offset + float(
            state.get("lateral_speed_mps", 0.0)
        ) * elapsed
        if target_offset >= current_offset:
            predicted_offset = min(predicted_offset, target_offset)
        else:
            predicted_offset = max(predicted_offset, target_offset)
        return xy + person_left * predicted_offset

    @staticmethod
    def _reciprocal_passing_side(env) -> float:
        """Return the active person's local side; robot uses the same local side."""
        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        best_distance = float("inf")
        best_side = 0.0
        for state in env.person_states[: env.active_people_count]:
            if not state.get("reciprocal_active", False):
                continue
            person_xy = np.asarray(state.get("current_xy", [99.0, 99.0]), dtype=np.float64)
            distance = float(np.linalg.norm(person_xy - robot_xy))
            if distance < best_distance:
                best_distance = distance
                best_side = float(state.get("avoidance_side", 1.0)) or 1.0
        return best_side

    def _prepare_context(self, env, start_xy: np.ndarray):
        """Prepare nearby geometry once, instead of once per velocity sample."""
        reach = self.HORIZON_SECONDS * float(env.MAX_LINEAR_SPEED) + 2.2
        all_static = (
            env.static_obstacle_data
            + env.table_obstacle_data
            + getattr(env, "decor_obstacle_data", [])
        )
        static_layout = [
            row
            for row in all_static
            if self._point_rectangle_clearance(start_xy, row) <= reach
        ]
        people = []
        for state in env.person_states[: env.active_people_count]:
            current = self._person_at(state, float(env.sim_time))
            if float(np.linalg.norm(current - start_xy)) <= reach + 0.8:
                people.append(state)

        # The environment advances people once at the start of the next
        # 0.04-second control step, before applying the robot command.  Model
        # that discrete update explicitly, then use exact close-range samples
        # so a short overlap cannot hide between DWA instants.
        fine_count = int(round(self.FINE_HORIZON_SECONDS / self.FINE_ROLLOUT_DT))
        sample_times = [
            sample * self.FINE_ROLLOUT_DT for sample in range(1, fine_count + 1)
        ]
        elapsed = self.FINE_HORIZON_SECONDS + self.ROLLOUT_DT
        while elapsed < self.HORIZON_SECONDS - 1e-8:
            sample_times.append(elapsed)
            elapsed += self.ROLLOUT_DT
        if not sample_times or sample_times[-1] < self.HORIZON_SECONDS - 1e-8:
            sample_times.append(self.HORIZON_SECONDS)

        future_people = []
        for elapsed in sample_times:
            future_people.append(
                [
                    (
                        self._person_at(
                            state,
                            float(env.sim_time)
                            + max(self.CONTROL_DT, float(elapsed)),
                        ),
                        state,
                    )
                    for state in people
                ]
            )
        return static_layout, sample_times, future_people

    def _fast_dynamic_gate(self, env, proposed: np.ndarray) -> tuple[bool, float]:
        """Cheap velocity-obstacle gate; full DWA runs only near collision courses."""
        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        yaw = float(env._get_robot_yaw())
        robot_velocity = float(proposed[0]) * float(env.MAX_LINEAR_SPEED) * np.array(
            [math.cos(yaw), math.sin(yaw)], dtype=np.float64
        )
        robot_radius = math.hypot(
            float(env.ROBOT_COLLISION_HALF_LENGTH),
            float(env.ROBOT_COLLISION_HALF_WIDTH),
        )
        best_clearance = 99.0
        possible_risk = False
        for state in env.person_states[: env.active_people_count]:
            # env.step() moves the person to the next control instant before
            # the robot starts moving, so gate against that pose rather than a
            # stale pre-step position.
            now = self._person_at(
                state, float(env.sim_time) + self.CONTROL_DT
            )
            soon = self._person_at(
                state, float(env.sim_time) + self.CONTROL_DT + 0.10
            )
            person_velocity = (soon - now) / 0.10
            relative = now - robot_xy
            relative_velocity = person_velocity - robot_velocity
            velocity_sq = float(np.dot(relative_velocity, relative_velocity))
            if velocity_sq < 1e-8:
                closest_time = 0.0
            else:
                closest_time = float(
                    np.clip(
                        -float(np.dot(relative, relative_velocity)) / velocity_sq,
                        0.0,
                        self.HORIZON_SECONDS,
                    )
                )
            combined_radius = robot_radius + float(state["footprint"]) / 2.0
            clearance = float(
                np.linalg.norm(relative + relative_velocity * closest_time)
                - combined_radius
            )
            best_clearance = min(best_clearance, clearance)
            # Enter the local planner before contact. Waiting for the old
            # 0.58 m threshold made the first response a late reverse.
            if clearance < self.EMERGENCY_DYNAMIC_CLEARANCE:
                possible_risk = True
        return possible_risk, best_clearance

    def _rear_approach_risk(
        self, env, proposed: np.ndarray
    ) -> tuple[bool, float]:
        """Detect a person closing on the robot from behind.

        The generic velocity-obstacle gate is 360-degree, but it does not tell
        the action selector *where* the threat is. Rear threats need special
        treatment because stop/reverse actions can make the collision worse.
        """
        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        yaw = float(env._get_robot_yaw())
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        robot_speed = float(proposed[0]) * float(env.MAX_LINEAR_SPEED)
        best_ttc = 99.0
        risk = False

        for state in env.person_states[: env.active_people_count]:
            now = self._person_at(state, float(env.sim_time) + self.CONTROL_DT)
            soon = self._person_at(
                state, float(env.sim_time) + self.CONTROL_DT + 0.10
            )
            person_velocity = (soon - now) / 0.10
            relative = now - robot_xy
            front = float(np.dot(relative, forward))
            lateral = float(np.dot(relative, left))
            # Only people genuinely behind the robot belong to this case.
            if front >= -0.10:
                continue

            person_forward_speed = float(np.dot(person_velocity, forward))
            person_lateral_speed = float(np.dot(person_velocity, left))
            closing_speed = person_forward_speed - robot_speed
            if closing_speed <= 0.05:
                continue

            longitudinal_limit = (
                float(env.ROBOT_COLLISION_HALF_LENGTH)
                + float(env.COLLISION_CONTACT_MARGIN)
                + float(state["footprint"]) / 2.0
            )
            lateral_limit = (
                float(env.ROBOT_COLLISION_HALF_WIDTH)
                + float(env.COLLISION_CONTACT_MARGIN)
                + float(state["footprint"]) / 2.0
                + self.REAR_LATERAL_MARGIN
            )
            gap = max((-front) - longitudinal_limit, 0.0)
            ttc = gap / closing_speed
            future_lateral = lateral + person_lateral_speed * ttc
            if (
                0.0 <= ttc <= self.REAR_TRIGGER_TTC
                and abs(future_lateral) <= lateral_limit
            ):
                risk = True
                best_ttc = min(best_ttc, float(ttc))

        return risk, best_ttc

    def _rear_escape_actions(self):
        """Forward-biased candidates for a closing pedestrian behind."""
        # Straight acceleration first, then bounded arcs. The trajectory
        # simulator rejects any candidate that would hit furniture or a second
        # pedestrian, so these do not bypass static/dynamic safety.
        for linear, turn in (
            (0.82, 0.00),
            (0.78, 0.18),
            (0.78, -0.18),
            (0.70, 0.30),
            (0.70, -0.30),
            (0.60, 0.42),
            (0.60, -0.42),
        ):
            yield np.array([linear, turn], dtype=np.float32)

    def _simulate(
        self,
        env,
        action: np.ndarray,
        target: np.ndarray,
        context=None,
    ) -> _Rollout:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        start_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        xy = start_xy.copy()
        yaw = float(env._get_robot_yaw())
        start_distance = float(np.linalg.norm(np.asarray(target) - start_xy))
        linear = float(action[0]) * float(env.MAX_LINEAR_SPEED)
        angular = float(action[1]) * float(env.MAX_ANGULAR_SPEED)
        if context is None:
            context = self._prepare_context(env, start_xy)
        static_layout, sample_times, future_people = context
        static_half_length = (
            float(env.ROBOT_COLLISION_HALF_LENGTH)
            + float(env.COLLISION_CONTACT_MARGIN)
            + self.STATIC_CLEARANCE_MARGIN
        )
        static_half_width = (
            float(env.ROBOT_COLLISION_HALF_WIDTH)
            + float(env.COLLISION_CONTACT_MARGIN)
            + self.STATIC_CLEARANCE_MARGIN
        )
        dynamic_half_length = (
            float(env.ROBOT_COLLISION_HALF_LENGTH)
            + float(env.COLLISION_CONTACT_MARGIN)
        )
        dynamic_half_width = (
            float(env.ROBOT_COLLISION_HALF_WIDTH)
            + float(env.COLLISION_CONTACT_MARGIN)
        )
        robot_radius = math.hypot(dynamic_half_length, dynamic_half_width)
        min_static = 99.0
        min_dynamic = 99.0
        collision_time = 99.0
        collision_kind = None

        # Check the current pose before integrating. If the moving person has
        # already entered the footprint between control updates, moving away
        # for 0.01 s must not make the trajectory look falsely safe.
        for row in static_layout:
            if self._pose_hits_box(
                xy, yaw, row, static_half_length, static_half_width
            ):
                collision_time = 0.0
                collision_kind = "static"
                break
        if collision_kind is None:
            for state in env.person_states[: env.active_people_count]:
                person_xy = self._person_at(
                    state, float(env.sim_time) + self.CONTROL_DT
                )
                footprint = float(state["footprint"])
                row = [
                    float(person_xy[0]),
                    float(person_xy[1]),
                    footprint,
                    footprint,
                    1.75,
                ]
                min_dynamic = min(
                    min_dynamic,
                    float(np.linalg.norm(person_xy - xy))
                    - robot_radius
                    - footprint / 2.0,
                )
                if self._pose_hits_box(
                    xy, yaw, row, dynamic_half_length, dynamic_half_width
                ):
                    collision_time = 0.0
                    collision_kind = "dynamic"
                    break
        previous_elapsed = 0.0

        for sample_index, elapsed in enumerate(sample_times):
            if collision_kind is not None:
                break
            dt = float(elapsed) - previous_elapsed
            previous_elapsed = float(elapsed)
            mid_yaw = yaw + 0.5 * angular * dt
            xy += linear * dt * np.array(
                [math.cos(mid_yaw), math.sin(mid_yaw)], dtype=np.float64
            )
            yaw = self._normalise_angle(yaw + angular * dt)

            if (
                xy[0] < float(env.WORLD_X_MIN) + static_half_length
                or xy[0] > float(env.WORLD_X_MAX) - static_half_length
                or xy[1] < float(env.WORLD_Y_MIN) + static_half_length
                or xy[1] > float(env.WORLD_Y_MAX) - static_half_length
            ):
                collision_time = elapsed
                collision_kind = "boundary"
                break

            for row in static_layout:
                min_static = min(
                    min_static,
                    self._point_rectangle_clearance(xy, row) - robot_radius,
                )
                if self._pose_hits_box(
                    xy, yaw, row, static_half_length, static_half_width
                ):
                    collision_time = elapsed
                    collision_kind = "static"
                    break
            if collision_kind is not None:
                break

            for person_xy, state in future_people[sample_index]:
                footprint = float(state["footprint"])
                row = [
                    float(person_xy[0]),
                    float(person_xy[1]),
                    footprint,
                    footprint,
                    1.75,
                ]
                min_dynamic = min(
                    min_dynamic,
                    float(np.linalg.norm(person_xy - xy))
                    - robot_radius
                    - footprint / 2.0,
                )
                # Only a true footprint intersection rejects the trajectory.
                # The soft safety margin belongs in scoring; otherwise every
                # outward escape is incorrectly rejected when agents are close.
                if self._pose_hits_box(
                    xy, yaw, row, dynamic_half_length, dynamic_half_width
                ):
                    collision_time = elapsed
                    collision_kind = "dynamic"
                    break
            if collision_kind is not None:
                break

        final_delta = np.asarray(target, dtype=np.float64) - xy
        final_distance = float(np.linalg.norm(final_delta))
        desired_heading = math.atan2(float(final_delta[1]), float(final_delta[0]))
        heading_alignment = math.cos(self._normalise_angle(desired_heading - yaw))
        progress = start_distance - final_distance
        proposed_delta = float(np.linalg.norm(action - self.last_action))
        clearance_bonus = min(max(min_dynamic, 0.0), 1.2)
        soft_margin_penalty = max(
            self.DYNAMIC_CLEARANCE_MARGIN - min_dynamic, 0.0
        )
        score = (
            7.0 * progress
            + 1.25 * heading_alignment
            + 0.95 * max(float(action[0]), 0.0)
            + 0.45 * clearance_bonus
            - 0.22 * abs(float(action[1]))
            - 0.18 * proposed_delta
            - 1.8 * max(-float(action[0]), 0.0)
            - 2.5 * soft_margin_penalty
        )
        safe = collision_kind is None
        return _Rollout(
            action=action,
            safe=safe,
            score=float(score),
            dynamic_clearance=float(min_dynamic),
            static_clearance=float(min_static),
            collision_time=float(collision_time),
            collision_kind=collision_kind,
            final_distance=final_distance,
        )

    @staticmethod
    def _unique(values) -> list[float]:
        output = []
        for value in values:
            clipped = float(np.clip(value, -1.0, 1.0))
            if not any(abs(clipped - old) < 1e-4 for old in output):
                output.append(clipped)
        return output

    def _candidate_actions(self, proposed: np.ndarray, allow_reverse: bool):
        linear_values = self._unique((float(proposed[0]),) + self.LINEAR_SAMPLES + (0.0,))
        # A learned/proposed turn can be ±1.0.  In the dynamic-person path
        # clamp that suggestion to the same bounded arc set so it cannot
        # resurrect a spin from outside the candidate table.
        bounded_proposed_turn = float(np.clip(proposed[1], -0.45, 0.45))
        turn_values = self._unique((bounded_proposed_turn,) + self.TURN_SAMPLES)
        for linear in linear_values:
            for turn in turn_values:
                if linear == 0.0 and abs(turn) > 0.78:
                    continue
                yield np.array([linear, turn], dtype=np.float32)
        if allow_reverse:
            # Reverse is deliberately straight and short.  Turning while
            # reversing was the source of the visible 360-degree recovery.
            for linear in (-0.06,):
                yield np.array([linear, 0.0], dtype=np.float32)

    def _emergency_side_actions(self, env):
        """Short arcs that avoid both current and predicted pedestrian motion.

        Head-on encounters can use the pedestrian's current lateral side.  A
        crossing pedestrian is different: turning away from where the person is
        *now* can steer directly into where that person is moving next.  For a
        clear lateral crossing, choose the side opposite the pedestrian's
        lateral velocity instead.
        """
        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        yaw = float(env._get_robot_yaw())
        forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
        left = np.array([-forward[1], forward[0]], dtype=np.float64)
        closest = None
        closest_state = None
        closest_distance = float("inf")
        sample_time = float(env.sim_time) + self.CONTROL_DT
        for state in env.person_states[: env.active_people_count]:
            person_xy = self._person_at(state, sample_time)
            distance = float(np.linalg.norm(person_xy - robot_xy))
            if distance < closest_distance:
                closest_distance = distance
                closest = person_xy
                closest_state = state
        if closest is None or closest_state is None:
            return

        relative = closest - robot_xy
        side = float(np.dot(relative, left))

        # Estimate the person's velocity in the robot frame.  In the controlled
        # crossing test the pedestrian travels bottom-to-top: before reaching
        # the robot centreline the old rule saw the person on the right and
        # commanded a left arc -- exactly the direction the person was moving.
        # Detect predominantly lateral motion and choose the opposite direction.
        soon = self._person_at(closest_state, sample_time + 0.10)
        person_velocity = (soon - closest) / 0.10
        lateral_velocity = float(np.dot(person_velocity, left))
        forward_velocity = float(np.dot(person_velocity, forward))
        clear_crossing = (
            abs(lateral_velocity) >= 0.16
            and abs(lateral_velocity) >= 1.25 * abs(forward_velocity)
        )

        if clear_crossing:
            # Pedestrian moving left in the robot frame -> robot turns right;
            # pedestrian moving right -> robot turns left.  This yields behind
            # the moving pedestrian rather than chasing its future path.
            turn_sign = -math.copysign(1.0, lateral_velocity)
            # Lock this side for the encounter so the generic DWA scoring does
            # not flip back to the old current-position choice on the next step.
            self._preferred_turn_sign = turn_sign
        elif abs(side) > 0.20:
            # Head-on / non-crossing case: person on robot-left -> turn right;
            # person on robot-right -> turn left.
            turn_sign = -math.copysign(1.0, side)
        elif self._preferred_turn_sign != 0.0:
            turn_sign = self._preferred_turn_sign
        else:
            turn_sign = 1.0

        for linear, turn_strength in (
            (0.62, 0.30),
            (0.52, 0.36),
            (0.42, 0.42),
            (0.30, 0.45),
        ):
            yield np.array(
                [linear, turn_sign * turn_strength], dtype=np.float32
            )

    def _current_dynamic_clearance(self, env) -> float:
        robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
        robot_radius = math.hypot(
            float(env.ROBOT_COLLISION_HALF_LENGTH),
            float(env.ROBOT_COLLISION_HALF_WIDTH),
        )
        values = []
        for state in env.person_states[: env.active_people_count]:
            person_xy = self._person_at(state, float(env.sim_time))
            values.append(
                float(np.linalg.norm(person_xy - robot_xy))
                - robot_radius
                - float(state["footprint"]) / 2.0
            )
        return min(values) if values else 99.0

    def _selection_score(self, rollout: _Rollout, nominal_unsafe: bool) -> float:
        score = rollout.score
        turn = float(rollout.action[1])
        if self._preferred_turn_sign != 0.0 and abs(turn) > 0.10:
            if math.copysign(1.0, turn) == self._preferred_turn_sign:
                score += 1.20 if nominal_unsafe else 0.35
            else:
                score -= 4.00
        return score

    def _finish(
        self,
        rollout: _Rollout,
        proposed: np.ndarray,
        reason: str,
    ) -> DWAPlannerDecision:
        action = rollout.action.astype(np.float32)
        intervened = bool(np.linalg.norm(action - proposed) > 0.035)
        if intervened and not self._intervening:
            self.intervention_events += 1
        self._intervening = intervened
        self.local_phase = reason if intervened else "nominal"
        if intervened:
            self.reason_counts[reason] += 1
        control_dt = 0.01 * 4.0
        if float(action[0]) < 0.0:
            # Stored in metres, not normalized action units.
            self._reverse_used += abs(float(action[0])) * 1.5 * control_dt
        self.last_action = action.copy()
        return DWAPlannerDecision(
            action=action,
            intervened=intervened,
            reason=reason,
            predicted_dynamic_clearance=rollout.dynamic_clearance,
            predicted_static_clearance=rollout.static_clearance,
            predicted_ttc=rollout.collision_time,
        )

    def _cooperative_command(
        self,
        env,
        proposed: np.ndarray,
        target: np.ndarray,
    ) -> DWAPlannerDecision:
        """Execute short straight reverse, bounded left turn, then pass forward."""
        phase = self._cooperative_phase
        if phase == "short-reverse":
            candidates = (
                np.array([-0.06, 0.0], dtype=np.float32),
                np.array([-0.04, 0.0], dtype=np.float32),
                np.array([0.0, 0.0], dtype=np.float32),
            )
            reason = "cooperative-short-reverse"
        elif phase == "left-turn":
            candidates = (
                np.array([0.58, 0.42], dtype=np.float32),
                np.array([0.50, 0.34], dtype=np.float32),
                np.array([0.40, 0.50], dtype=np.float32),
            )
            reason = "cooperative-left-turn"
        else:
            candidates = (
                np.array([0.62, 0.10], dtype=np.float32),
                np.array([0.52, 0.16], dtype=np.float32),
                np.array([0.40, 0.24], dtype=np.float32),
            )
            reason = "cooperative-pass-forward"

        context = self._prepare_context(
            env, np.asarray(env._get_robot_position(), dtype=np.float64)
        )
        static_context = (
            context[0],
            context[1],
            [[] for _ in context[1]],
        )
        viable: list[_Rollout] = []
        for candidate in candidates:
            static_rollout = self._simulate(env, candidate, target, static_context)
            if static_rollout.safe:
                viable.append(self._simulate(env, candidate, target, context))

        # IMPORTANT: in the multi-person restaurant a scripted cooperative
        # command may be static-safe but dynamically unsafe because a second
        # waiter/customer enters the same space.  Never continue the scripted
        # pass merely because it is safe with respect to furniture.
        dynamic_safe = [rollout for rollout in viable if rollout.safe]
        advance_sequence = True

        if dynamic_safe:
            if phase == "short-reverse":
                # Preserve the original straight-reverse preference, but only
                # when that reverse is also safe with respect to every person.
                chosen = dynamic_safe[0]
            else:
                chosen = max(
                    dynamic_safe,
                    key=lambda item: (
                        item.dynamic_clearance,
                        item.collision_time,
                        item.score,
                    ),
                )
        else:
            # A second pedestrian (or a changed pedestrian trajectory) has
            # invalidated the cooperative maneuver.  Temporarily suspend the
            # scripted phase and use the same predictive emergency candidates
            # as the generic DWA.  The cooperative phase is not advanced until
            # a dynamically safe command exists again.
            advance_sequence = False
            emergency_rollouts: list[_Rollout] = []
            for candidate in self._emergency_side_actions(env):
                static_rollout = self._simulate(
                    env, candidate, target, static_context
                )
                if not static_rollout.safe:
                    continue
                emergency_rollouts.append(
                    self._simulate(env, candidate, target, context)
                )

            emergency_safe = [
                rollout for rollout in emergency_rollouts if rollout.safe
            ]
            if emergency_safe:
                chosen = max(
                    emergency_safe,
                    key=lambda item: (
                        item.dynamic_clearance,
                        item.collision_time,
                        item.score,
                    ),
                )
                reason = "cooperative-multiperson-escape"
            else:
                # Try stop / short straight reverse before accepting any
                # predicted contact.  These candidates are also checked against
                # all people and all nearby static obstacles.
                fallback_rollouts: list[_Rollout] = []
                for candidate in (
                    np.array([0.0, 0.0], dtype=np.float32),
                    np.array([-0.06, 0.0], dtype=np.float32),
                    np.array([-0.04, 0.0], dtype=np.float32),
                ):
                    static_rollout = self._simulate(
                        env, candidate, target, static_context
                    )
                    if not static_rollout.safe:
                        continue
                    fallback_rollouts.append(
                        self._simulate(env, candidate, target, context)
                    )

                fallback_safe = [
                    rollout for rollout in fallback_rollouts if rollout.safe
                ]
                if fallback_safe:
                    chosen = max(
                        fallback_safe,
                        key=lambda item: (
                            item.dynamic_clearance,
                            item.collision_time,
                            item.score,
                        ),
                    )
                    reason = "cooperative-multiperson-hold"
                else:
                    # If contact is already unavoidable at the next control
                    # instant, choose the non-static emergency trajectory that
                    # maximizes separation rather than blindly continuing the
                    # forward cooperative pass.
                    emergency_pool = emergency_rollouts + fallback_rollouts
                    if emergency_pool:
                        chosen = max(
                            emergency_pool,
                            key=lambda item: (
                                item.collision_kind not in ("static", "boundary"),
                                item.dynamic_clearance,
                                item.collision_time,
                                abs(float(item.action[1])),
                                item.score,
                            ),
                        )
                        reason = "cooperative-emergency-separate"
                    else:
                        chosen = self._simulate(
                            env, np.zeros(2, dtype=np.float32), target, context
                        )
                        reason = "cooperative-static-hold"

        if advance_sequence:
            self._cooperative_phase_steps += 1
            if (
                phase == "short-reverse"
                and self._cooperative_phase_steps >= self.COOPERATIVE_REVERSE_STEPS
            ):
                self._cooperative_phase = "left-turn"
                self._cooperative_phase_steps = 0
            elif (
                phase == "left-turn"
                and self._cooperative_phase_steps >= self.COOPERATIVE_LEFT_STEPS
            ):
                self._cooperative_phase = "pass-forward"
                self._cooperative_phase_steps = 0
        return self._finish(chosen, proposed, reason)

    def filter_action(self, env, proposed_action, target) -> DWAPlannerDecision:
        proposed = np.clip(np.asarray(proposed_action, dtype=np.float32), -1.0, 1.0)
        target = np.asarray(target, dtype=np.float32)
        reciprocal_side = self._reciprocal_passing_side(env)
        reciprocal_active = any(
            bool(state.get("reciprocal_active", False))
            for state in env.person_states[: env.active_people_count]
        )
        if reciprocal_active:
            self._avoidance_active = True
            self._clear_steps = 0
            self._preferred_turn_sign = 1.0
            if self._cooperative_phase == "idle":
                self._cooperative_phase = "short-reverse"
                self._cooperative_phase_steps = 0

        if self._cooperative_phase != "idle":
            minimum_sequence_complete = self._cooperative_phase == "pass-forward"
            if minimum_sequence_complete and not reciprocal_active:
                self._cooperative_phase = "idle"
                self._cooperative_phase_steps = 0
                self._avoidance_active = False
                self._preferred_turn_sign = 0.0
                self._reverse_used = 0.0
            else:
                return self._cooperative_command(env, proposed, target)
        possible_risk, gate_clearance = self._fast_dynamic_gate(env, proposed)
        rear_risk, rear_ttc = self._rear_approach_risk(env, proposed)
        possible_risk = bool(possible_risk or rear_risk)
        dynamic_emergency = bool(
            rear_risk
            or (possible_risk and gate_clearance < self.EMERGENCY_DYNAMIC_CLEARANCE)
        )

        # Most restaurant steps take this O(number-of-people) path. Static
        # navigation is already protected by A* plus furniture-only LiDAR.
        if not possible_risk and not self._avoidance_active:
            robot_xy = np.asarray(env._get_robot_position(), dtype=np.float64)
            nominal = _Rollout(
                action=proposed,
                safe=True,
                score=0.0,
                dynamic_clearance=float(gate_clearance),
                static_clearance=99.0,
                collision_time=99.0,
                collision_kind=None,
                final_distance=float(np.linalg.norm(target - robot_xy)),
            )
            return self._finish(nominal, proposed, "nominal-sac-a-star")

        context = self._prepare_context(
            env, np.asarray(env._get_robot_position(), dtype=np.float64)
        )
        if dynamic_emergency:
            self._avoidance_active = True
            self._clear_steps = 0
        nominal = self._simulate(env, proposed, target, context)

        # Rear approach is asymmetric: stop/reverse can let a pedestrian drive
        # directly into the robot. Prefer a forward or forward-arc escape, but
        # only after checking the complete 1.6 s rollout against ALL people and
        # static geometry.
        if rear_risk:
            rear_rollouts: list[_Rollout] = []
            rear_safe: list[_Rollout] = []
            for candidate in self._rear_escape_actions():
                rollout = self._simulate(env, candidate, target, context)
                rear_rollouts.append(rollout)
                if rollout.safe:
                    rear_safe.append(rollout)
            if rear_safe:
                best_rear = max(
                    rear_safe,
                    key=lambda item: (
                        item.dynamic_clearance,
                        item.score,
                        float(item.action[0]),
                    ),
                )
                return self._finish(
                    best_rear, proposed, "dwa-rear-forward-escape"
                )
            # If every fully safe rear escape is already unavailable, do not
            # reverse into the person. Pick the non-static candidate that buys
            # the most separation/time and reassess on the next 0.04 s step.
            non_static = [
                item
                for item in rear_rollouts
                if item.collision_kind not in ("static", "boundary")
            ]
            if non_static:
                best_rear = max(
                    non_static,
                    key=lambda item: (
                        item.dynamic_clearance,
                        item.collision_time,
                        item.score,
                    ),
                )
                return self._finish(
                    best_rear, proposed, "dwa-rear-emergency-separate"
                )

        # SAC+A* remains fully in control whenever its short future is safe.
        if nominal.safe and not self._avoidance_active:
            return self._finish(nominal, proposed, "nominal-sac-a-star")

        current_dynamic_clearance = self._current_dynamic_clearance(env)
        if not nominal.safe and nominal.collision_kind == "dynamic":
            self._avoidance_active = True
            self._clear_steps = 0
        elif self._avoidance_active and nominal.safe and not reciprocal_active:
            if current_dynamic_clearance > 0.75:
                self._clear_steps += 1
            else:
                self._clear_steps = 0
            if current_dynamic_clearance > 0.75 and self._clear_steps >= 4:
                self._avoidance_active = False
                self._preferred_turn_sign = 0.0
                self._reverse_used = 0.0
                return self._finish(nominal, proposed, "nominal-sac-a-star")

        allow_reverse = (
            not rear_risk
            and nominal.collision_kind == "dynamic"
            and nominal.collision_time <= 0.65
            and self._reverse_used < self.REVERSE_LIMIT_METRES
        )
        forward_safe: list[_Rollout] = []
        fallback_safe: list[_Rollout] = []
        for candidate in self._candidate_actions(proposed, False):
            rollout = self._simulate(env, candidate, target, context)
            if not rollout.safe:
                continue
            if float(candidate[0]) > 0.05:
                forward_safe.append(rollout)
            else:
                fallback_safe.append(rollout)

        side_escape_safe: list[_Rollout] = []
        if dynamic_emergency or reciprocal_active:
            for candidate in self._emergency_side_actions(env):
                rollout = self._simulate(env, candidate, target, context)
                if rollout.safe:
                    side_escape_safe.append(rollout)

        # A usable forward arc always wins over waiting or reversing.
        pool = forward_safe if forward_safe else fallback_safe
        if dynamic_emergency and side_escape_safe:
            # A lateral escape is preferred while a person is inside the early
            # dynamic envelope; this prevents a same-line bounded reverse.
            pool = side_escape_safe
        if reciprocal_active and forward_safe:
            same_side = [
                rollout
                for rollout in forward_safe
                if abs(float(rollout.action[1])) > 0.10
                and math.copysign(1.0, float(rollout.action[1]))
                == self._preferred_turn_sign
            ]
            if same_side:
                pool = same_side
        best = (
            max(pool, key=lambda item: self._selection_score(item, not nominal.safe))
            if pool
            else None
        )

        if best is not None and float(best.action[0]) > 0.05 and abs(float(best.action[1])) > 0.10:
            chosen_sign = math.copysign(1.0, float(best.action[1]))
            if self._preferred_turn_sign == 0.0:
                self._preferred_turn_sign = chosen_sign

        # Reverse samples are evaluated only if no non-reversing motion is safe.
        if best is None and allow_reverse:
            reverse_safe = []
            for candidate in self._candidate_actions(proposed, True):
                if float(candidate[0]) >= -0.01:
                    continue
                rollout = self._simulate(env, candidate, target, context)
                if rollout.safe:
                    reverse_safe.append(rollout)
            if reverse_safe:
                best = max(
                    reverse_safe,
                    key=lambda item: self._selection_score(item, not nominal.safe),
                )

        if best is None and dynamic_emergency:
            # If every trajectory still intersects at the first instant,
            # choose the action that separates fastest and delays contact.
            # Stopping/reversing here was the source of the repeated hits.
            side_rollouts = []
            side_candidates = list(self._emergency_side_actions(env))
            for candidate in side_candidates:
                side_rollouts.append(
                    self._simulate(env, candidate, target, context)
                )
            emergency_rollouts = list(side_rollouts)
            emergency_candidates = []
            # Do not re-introduce rotating/reversing candidates here.  If the
            # cooperative trigger was missed, the same bounded side escape is
            # still the correct dynamic-person fallback.
            emergency_candidates.extend(self._emergency_side_actions(env))
            emergency_candidates.extend(self._candidate_actions(proposed, False))
            for candidate in emergency_candidates:
                emergency_rollouts.append(
                    self._simulate(env, candidate, target, context)
                )
            if emergency_rollouts:
                # Even if the current pose is already inside the conservative
                # envelope, choose a turning command first. Returning the
                # straight SAC command here recreates the same-line collision;
                # a lateral component is the only action that can separate the
                # footprints on the next control interval.
                lateral_rollouts = [
                    item
                    for item in emergency_rollouts
                    if abs(float(item.action[1])) > 0.10
                ]
                # Side candidates encode the sign away from the closest
                # person. Keep that sign even when every rollout is already
                # unsafe at t=0; choosing an arbitrary opposite turn is what
                # caused the old planner to steer back into the person.
                emergency_pool = side_rollouts or lateral_rollouts or emergency_rollouts
                best = max(
                    emergency_pool,
                    key=lambda item: (
                        item.collision_kind not in ("static", "boundary"),
                        item.dynamic_clearance,
                        item.collision_time,
                        item.score,
                    ),
                )
                if abs(float(best.action[1])) > 0.10:
                    return self._finish(best, proposed, "dwa-emergency-lateral")

        if best is None:
            stop = self._simulate(
                env, np.zeros(2, dtype=np.float32), target, context
            )
            if stop.safe:
                return self._finish(stop, proposed, "dwa-emergency-stop")
            if allow_reverse:
                reverse = self._simulate(
                    env,
                    np.array([-0.06, 0.0], dtype=np.float32),
                    target,
                    context,
                )
                return self._finish(reverse, proposed, "dwa-short-reverse")
            return self._finish(stop, proposed, "dwa-no-safe-velocity")

        linear = float(best.action[0])
        turn = float(best.action[1])
        if linear < -0.01:
            reason = "dwa-short-reverse"
        elif linear < 0.03:
            reason = "dwa-stop-or-rotate"
        elif reciprocal_active and turn > 0.12:
            reason = "reciprocal-left-pass"
        elif reciprocal_active and turn < -0.12:
            reason = "reciprocal-right-pass"
        elif turn > 0.12:
            reason = "dwa-left-arc"
        elif turn < -0.12:
            reason = "dwa-right-arc"
        else:
            reason = "dwa-slow-forward"
        return self._finish(best, proposed, reason)
