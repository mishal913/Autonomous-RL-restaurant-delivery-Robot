"""Train Residual SAC V2 to perform moving-human avoidance itself.

Recommended fresh run:
    python train_residual_sac_v2.py --total-timesteps 500000

The final stage uses four NON-COOPERATIVE moving people. This deliberately
removes the old person-slowing/yield behavior so the learned policy, rather
than a scripted human or DWA routine, must solve the encounter.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from residual_env_v2 import ResidualHumanAvoidanceEnvV2, ResidualEnvV2Config
from restaurant_env_final import RestaurantDeliveryEnv

# fraction, people count, people-speed scale. Cooperation stays 0.0 throughout.
CURRICULUM = (
    (0.00, 1, 0.45),
    (0.18, 2, 0.65),
    (0.40, 3, 0.82),
    (0.65, 4, 1.00),
)


class HumanAvoidanceCurriculum(BaseCallback):
    def __init__(self, base_env: RestaurantDeliveryEnv, total_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.base_env = base_env
        self.total_timesteps = int(total_timesteps)
        self._stage = -1

    def _on_step(self) -> bool:
        progress = self.num_timesteps / max(1, self.total_timesteps)
        stage = 0
        for index, (start, _, _) in enumerate(CURRICULUM):
            if progress >= start:
                stage = index
        if stage != self._stage:
            self._stage = stage
            _, people, speed_scale = CURRICULUM[stage]
            self.base_env.active_people_count = int(people)
            self.base_env.dynamic_people_enabled = people > 0
            self.base_env.people_speed_scale = float(speed_scale)
            self.base_env.human_cooperation_probability = 0.0
            if self.verbose:
                print(
                    f"[curriculum-v2] t={self.num_timesteps} "
                    f"people={people} speed_scale={speed_scale:.2f} cooperation=0.00"
                )
        return True


class SafetyMetricsCallback(BaseCallback):
    def __init__(self, log_every: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.log_every = int(log_every)
        self.steps = 0
        self.shield = 0
        self.dynamic_collisions = 0
        self.successes = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            self.steps += 1
            self.shield += int(bool(info.get("emergency_shield_intervened", False)))
            self.dynamic_collisions += int(info.get("collision_type") == "dynamic")
            if "episode" in info or info.get("success") or info.get("collision") or info.get("out_of_bounds"):
                if info.get("success") or info.get("collision") or info.get("out_of_bounds"):
                    self.episodes += 1
                    self.successes += int(bool(info.get("success", False)))

        if self.steps >= self.log_every:
            shield_rate = self.shield / max(1, self.steps)
            if self.verbose:
                print(
                    f"[sac-v2] t={self.num_timesteps} shield_rate={shield_rate:.3f} "
                    f"dynamic_collisions={self.dynamic_collisions} "
                    f"episode_successes={self.successes}/{max(1, self.episodes)}"
                )
            self.logger.record("v2/emergency_shield_rate", shield_rate)
            self.logger.record("v2/dynamic_collisions", self.dynamic_collisions)
            self.steps = self.shield = self.dynamic_collisions = 0
            self.successes = self.episodes = 0
        return True


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train Residual SAC V2 human avoidance")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=project / "dynamic_sac_residual_v2.zip")
    parser.add_argument("--checkpoint-dir", type=Path, default=project / "checkpoints_v2")
    parser.add_argument("--checkpoint-every", type=int, default=25_000)
    parser.add_argument("--assets-dir", type=Path, default=project)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def build_env(project: Path, seed: int):
    base_env = RestaurantDeliveryEnv(
        render=False,
        embedded_gui=False,
        robot_urdf=project / "restaurant_delivery_robot.urdf",
        table_urdf=project / "restaurant_table_set.urdf",
        waiter_urdf=project / "restaurant_waiter.urdf",
        customer_urdf=project / "restaurant_customer.urdf",
        active_people_count=1,
        human_cooperation_probability=0.0,
    )
    base_env.people_speed_scale = 0.45
    wrapped = ResidualHumanAvoidanceEnvV2(base_env, ResidualEnvV2Config())
    wrapped.reset(seed=seed)
    return wrapped, base_env


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.cpu)

    wrapped, base_env = build_env(args.assets_dir, args.seed)
    monitored = Monitor(wrapped)

    if args.resume is not None:
        print(f"Resuming V2 checkpoint: {args.resume}")
        model = SAC.load(str(args.resume), env=monitored, device="cpu")
    else:
        model = SAC(
            "MlpPolicy",
            monitored,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            learning_starts=10_000,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(args.assets_dir / "tb_logs_v2"),
            seed=args.seed,
            verbose=1,
        )

    callbacks = CallbackList([
        HumanAvoidanceCurriculum(base_env, args.total_timesteps),
        SafetyMetricsCallback(),
        CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(args.checkpoint_dir),
            name_prefix="residual_sac_v2",
        ),
    ])

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.resume is None),
    )
    model.save(str(args.out))
    print(f"Saved Residual SAC V2: {args.out}")


if __name__ == "__main__":
    main()
