"""Train the residual human-avoidance SAC policy under the DWA safety shield.

Usage (run on a machine with genesis-world[render] installed; a GPU is not
required for Genesis's CPU backend but training will be slow without one --
see the timestep guidance in README.md):

    python train_residual_sac.py --total-timesteps 300_000 \
        --out dynamic_sac_residual_v1.zip

IMPORTANT SAFETY NOTE: this policy only ever outputs a bounded *correction*
on top of a fixed baseline, and every action -- however undertrained -- is
still passed through the exact same PredictiveDWAPlanner used in production
before it reaches the robot (see residual_env.py). That means training for
fewer timesteps cannot make the system less safe than the current DWA-only
runtime; it can only mean the residual has learned less, so it contributes
less improvement on top of what the shield already guarantees. If you can
only afford a short training budget, that is a legitimate, safe choice --
you are trading "how much better than baseline" for "how long you wait",
not trading away safety itself.

The curriculum is expressed as FRACTIONS of --total-timesteps (see
CURRICULUM_FRACTIONS below), so it automatically rescales whether you train
for 300k steps or 2M steps -- you do not need to hand-edit stage boundaries
when you change --total-timesteps.

Training can also be resumed across multiple short sessions instead of one
long continuous run: pass --resume path/to/checkpoint.zip to continue from
a saved checkpoint (see --checkpoint-every / the checkpoints/ folder).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from residual_env import ResidualEnvConfig, ResidualHumanAvoidanceEnv
from restaurant_env_final import RestaurantDeliveryEnv

# (fraction_of_total_timesteps_at_which_stage_starts, people_count, cooperation_probability)
CURRICULUM_FRACTIONS = (
    (0.00, 1, 1.00),
    (0.15, 2, 0.75),
    (0.35, 3, 0.55),
    (0.60, 4, 0.35),  # production-matching final stage
)


class CurriculumCallback(BaseCallback):
    """Advances active_people_count / human_cooperation_probability on the
    underlying RestaurantDeliveryEnv as training progresses. Uses
    set_curriculum() for the people-count/speed pairing already defined on
    the env, then overrides cooperation probability directly since
    set_curriculum() does not expose that knob."""

    STAGE_TO_SET_CURRICULUM = {1: 1, 2: 2, 3: 2, 4: 3}

    def __init__(self, base_env: RestaurantDeliveryEnv, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.base_env = base_env
        self.total_timesteps = int(total_timesteps)
        self._stage_index = -1

    def _current_stage(self) -> int:
        stage = 0
        progress = self.num_timesteps / max(1, self.total_timesteps)
        for index, (start_frac, _, _) in enumerate(CURRICULUM_FRACTIONS):
            if progress >= start_frac:
                stage = index
        return stage

    def _on_step(self) -> bool:
        stage = self._current_stage()
        if stage != self._stage_index:
            self._stage_index = stage
            _, people, cooperation = CURRICULUM_FRACTIONS[stage]
            set_curriculum_stage = self.STAGE_TO_SET_CURRICULUM[people]
            self.base_env.set_curriculum(set_curriculum_stage)
            # set_curriculum() only offers people-count in {0,1,2,4}; override
            # with the exact curriculum-table count (adds an intermediate
            # 3-person stage) after it sets the paired speed scale.
            self.base_env.active_people_count = int(people)
            self.base_env.human_cooperation_probability = float(cooperation)
            if self.verbose:
                print(
                    f"[curriculum] t={self.num_timesteps} -> "
                    f"people={people} cooperation={cooperation:.2f}"
                )
        return True


class InterventionRateCallback(BaseCallback):
    """Logs rolling DWA-intervention rate and dynamic-collision count to
    stdout/tensorboard so you can watch the residual policy learn to need
    the shield less over time -- this is the metric that should trend down,
    not the raw episodic reward, since the shield already prevents most
    collisions and can mask a policy that has merely learned to lean on it."""

    def __init__(self, log_every: int = 5_000, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = log_every
        self._intervened = 0
        self._dynamic_collisions = 0
        self._steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self._steps += 1
            if info.get("dwa_intervened"):
                self._intervened += 1
            if info.get("collision_type") == "dynamic":
                self._dynamic_collisions += 1
        if self._steps >= self.log_every:
            rate = self._intervened / max(1, self._steps)
            if self.verbose:
                print(
                    f"[intervention] t={self.num_timesteps} "
                    f"shield_rate={rate:.3f} dynamic_collisions={self._dynamic_collisions}"
                )
            self.logger.record("custom/dwa_intervention_rate", rate)
            self.logger.record("custom/dynamic_collisions", self._dynamic_collisions)
            self._intervened = 0
            self._dynamic_collisions = 0
            self._steps = 0
        return True


def build_env(project: Path, seed: int) -> tuple[ResidualHumanAvoidanceEnv, RestaurantDeliveryEnv]:
    base_env = RestaurantDeliveryEnv(
        render=False,
        embedded_gui=False,
        robot_urdf=project / "restaurant_delivery_robot.urdf",
        table_urdf=project / "restaurant_table_set.urdf",
        waiter_urdf=project / "restaurant_waiter.urdf",
        customer_urdf=project / "restaurant_customer.urdf",
        active_people_count=1,
        human_cooperation_probability=1.0,
    )
    wrapped = ResidualHumanAvoidanceEnv(base_env, ResidualEnvConfig())
    wrapped.reset(seed=seed)
    return wrapped, base_env


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train residual human-avoidance SAC")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000,
        help="Lower is safe, just less-trained -- the DWA shield still guards "
        "every action regardless of how much training happened. 300k is a "
        "realistic budget for a single machine over ~1-2 days; 2M is the "
        "'ideally fully converged' number, not a requirement.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=project / "dynamic_sac_residual_v1.zip")
    parser.add_argument("--checkpoint-dir", type=Path, default=project / "checkpoints")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25_000,
        help="Smaller = more frequent safety-net saves, useful if you can "
        "only train in short sessions and need to resume often.",
    )
    parser.add_argument("--assets-dir", type=Path, default=project)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a previously saved .zip checkpoint to continue training "
        "from instead of starting fresh. --total-timesteps is the ADDITIONAL "
        "steps to run, not a new grand total.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gs.init(backend=gs.cpu)

    wrapped_env, base_env = build_env(args.assets_dir, args.seed)
    monitored = Monitor(wrapped_env)

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        model = SAC.load(str(args.resume), env=monitored)
    else:
        model = SAC(
            "MlpPolicy",
            monitored,
            learning_rate=3e-4,
            buffer_size=300_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            learning_starts=5_000,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(args.assets_dir / "tb_logs"),
            seed=args.seed,
            verbose=1,
        )

    callbacks = CallbackList(
        [
            CurriculumCallback(base_env, total_timesteps=args.total_timesteps),
            InterventionRateCallback(),
            CheckpointCallback(
                save_freq=args.checkpoint_every,
                save_path=str(args.checkpoint_dir),
                name_prefix="residual_sac",
            ),
        ]
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.resume is None),
    )
    model.save(str(args.out))
    print(f"Saved trained residual policy to {args.out}")


if __name__ == "__main__":
    main()
