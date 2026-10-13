"""Configuration dataclasses and CLI for FluxSwarm."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Optional

import yaml

SIM_LENGTH_X = 100.0
SIM_LENGTH_Y = 2.0
MEMBER_RADIUS = 0.25


@dataclass
class SimConfig:
    length_x: float = SIM_LENGTH_X
    length_y: float = SIM_LENGTH_Y
    resolution_scale: float = 10.0  # cells per mm
    dt: float = 0.005
    total_time: float = 100.0
    dt_substeps: int = 40
    viscosity: float = 3.0
    inflow_velocity: float = 400.0
    inflow_period: float = 1.0
    fluid_density: float = 1.06

    @property
    def resolution(self) -> tuple[int, int]:
        return (
            max(2, int(self.length_x * self.resolution_scale)),
            max(2, int(self.length_y * self.resolution_scale)),
        )

    @property
    def dx(self) -> float:
        return self.length_x / self.resolution[0]

    @property
    def dy(self) -> float:
        return self.length_y / self.resolution[1]

    @property
    def time_steps(self) -> int:
        return int(self.total_time / self.dt)

    @property
    def substeps(self) -> int:
        return max(1, self.dt_substeps)


@dataclass
class SwarmConfig:
    num_x: int = 8
    num_y: int = 2
    member_radius: float = MEMBER_RADIUS
    member_density: float = 15.12
    member_max_force: float = 1491.0 / 2.0
    use_centered_layout: bool = True
    left_location: Optional[float] = None
    bottom_location: Optional[float] = None
    member_interval_x: Optional[float] = None
    member_interval_y: Optional[float] = None

    @property
    def num_members(self) -> int:
        return self.num_x * self.num_y


@dataclass
class TaskConfig:
    episode_duration: float = 10.0
    success_x: float = 20.0
    failure_x: float = 80.0
    progress_reward: Literal["potential", "fluid-relative", "legacy"] = "potential"
    w_progress: float = 1.0
    w_energy: float = 1.0
    w_smooth: float = 1.0


@dataclass
class ObsConfig:
    preset: Literal["rich", "legacy"] = "rich"
    localization: Literal["none", "imu", "displacement", "absolute-y", "full"] = "none"
    ring_points: int = 4
    ring_radius_factor: float = 3.0
    history: int = 4
    neighbor_k: int = 5
    neighbor_radius: float = 10.0
    velocity_frame: Literal["fluid", "lab"] = "fluid"
    imu_bias: float = 1.0
    imu_noise: float = 0.1


@dataclass
class TrainConfig:
    batch_envs: int = 64
    n_steps: int = 200
    batch_size: int = 4
    update_epochs: int = 4
    ent_coef: float = 0.01
    clip_coef: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    lr: float = 1e-3
    adam_eps: float = 1e-8
    total_timesteps_per_env: int = 25_000
    use_pcgrad: bool = True
    pcgrad_protect_progress: bool = False
    adv_norm: Literal["independent", "weighted", "off"] = "independent"
    orthogonal_init: bool = False
    use_action_x_prior: bool = True
    action_x_prior_warmup_fraction: float = 0.2
    actor: Literal["mlp", "recurrent"] = "mlp"
    hidden_sizes: tuple[int, ...] = (256, 256)
    log_std_init: float = -1.0
    max_grad_norm: float = 0.5
    device: str = "cuda"
    device_split: Literal["gpu", "cpu-shard"] = "gpu"
    cpu_shard_fraction: float = 0.25
    seed: int = 0
    save_fields: bool = False
    live_swarm_tb_every: int = 100  # 0 disables; env-0 position image to TensorBoard
    live_swarm_tb_dpi: int = 150
    live_swarm_tb_scale: float = 1.5
    live_swarm_tb_aspect: float = 4.0  # figure width:height (x:y display)
    tensorboard_port: int = 6006
    resume: bool = True
    resume_from: Optional[str] = None


@dataclass
class RunConfig:
    output_root: str = "runs_new"
    tag: str = ""
    wall_policy_mode: Literal["static", "dynamic"] = "static"


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        def _section(section_cls, key: str):
            raw = d.get(key, {}) or {}
            valid = {f.name for f in fields(section_cls)}
            return section_cls(**{k: v for k, v in raw.items() if k in valid})

        return cls(
            sim=_section(SimConfig, "sim"),
            swarm=_section(SwarmConfig, "swarm"),
            task=_section(TaskConfig, "task"),
            obs=_section(ObsConfig, "obs"),
            train=_section(TrainConfig, "train"),
            run=_section(RunConfig, "run"),
        )

    @classmethod
    def load(cls, path: Path | str) -> "Config":
        with Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f) or {})


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    mapping = {
        "sim_length_x": ("sim", "length_x"),
        "sim_length_y": ("sim", "length_y"),
        "resolution_scale": ("sim", "resolution_scale"),
        "dt": ("sim", "dt"),
        "total_time": ("sim", "total_time"),
        "dt_substeps": ("sim", "dt_substeps"),
        "inflow_velocity": ("sim", "inflow_velocity"),
        "swarm_num_x": ("swarm", "num_x"),
        "swarm_num_y": ("swarm", "num_y"),
        "member_radius": ("swarm", "member_radius"),
        "swarm_max_force": ("swarm", "member_max_force"),
        "episode_duration": ("task", "episode_duration"),
        "success_x": ("task", "success_x"),
        "failure_x": ("task", "failure_x"),
        "progress_reward": ("task", "progress_reward"),
        "w_progress": ("task", "w_progress"),
        "w_energy": ("task", "w_energy"),
        "w_smooth": ("task", "w_smooth"),
        "obs_preset": ("obs", "preset"),
        "obs_localization": ("obs", "localization"),
        "obs_ring_points": ("obs", "ring_points"),
        "obs_history": ("obs", "history"),
        "neighbor_k": ("obs", "neighbor_k"),
        "neighbor_radius": ("obs", "neighbor_radius"),
        "velocity_frame": ("obs", "velocity_frame"),
        "imu_bias": ("obs", "imu_bias"),
        "imu_noise": ("obs", "imu_noise"),
        "batch_envs": ("train", "batch_envs"),
        "n_steps": ("train", "n_steps"),
        "batch_size": ("train", "batch_size"),
        "update_epochs": ("train", "update_epochs"),
        "ent_coef": ("train", "ent_coef"),
        "clip_coef": ("train", "clip_coef"),
        "gamma": ("train", "gamma"),
        "lr": ("train", "lr"),
        "adam_eps": ("train", "adam_eps"),
        "total_timesteps_per_env": ("train", "total_timesteps_per_env"),
        "adv_norm": ("train", "adv_norm"),
        "device_split": ("train", "device_split"),
        "seed": ("train", "seed"),
        "save_fields": ("train", "save_fields"),
        "live_swarm_tb_every": ("train", "live_swarm_tb_every"),
        "live_swarm_tb_dpi": ("train", "live_swarm_tb_dpi"),
        "live_swarm_tb_scale": ("train", "live_swarm_tb_scale"),
        "live_swarm_tb_aspect": ("train", "live_swarm_tb_aspect"),
        "tensorboard_port": ("train", "tensorboard_port"),
        "resume_from": ("train", "resume_from"),
        "actor": ("train", "actor"),
        "output_root": ("run", "output_root"),
        "tag": ("run", "tag"),
        "wall_policy_mode": ("run", "wall_policy_mode"),
    }

    for flag, (section, attr) in mapping.items():
        if not hasattr(args, flag):
            continue
        val = getattr(args, flag)
        if val is None:
            continue
        setattr(getattr(cfg, section), attr, val)

    if getattr(args, "no_pcgrad", False):
        cfg.train.use_pcgrad = False
    if getattr(args, "pcgrad_protect_progress", False):
        cfg.train.pcgrad_protect_progress = True
    if getattr(args, "orthogonal_init", None) is not None:
        cfg.train.orthogonal_init = bool(args.orthogonal_init)
    if getattr(args, "use_action_x_prior", None) is not None:
        cfg.train.use_action_x_prior = args.use_action_x_prior
    if getattr(args, "action_x_prior_warmup_fraction", None) is not None:
        cfg.train.action_x_prior_warmup_fraction = args.action_x_prior_warmup_fraction
    if getattr(args, "no_save_fields", False):
        cfg.train.save_fields = False

    return cfg


def build_argparser(description: str | None = None) -> argparse.ArgumentParser:
    """Build CLI parser. Arg defaults are None (unset); help shows Config defaults."""
    d = Config()  # source of truth for documented defaults
    parser = argparse.ArgumentParser(
        description=description
        or "FluxSwarm: multi-objective multi-agent PPO for swarms in fluid channels.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file; CLI flags override it (default: none)",
    )

    g = parser.add_argument_group("simulation")
    g.add_argument("--sim-length-x", type=float, default=None, help=f"domain length x, mm ({d.sim.length_x})")
    g.add_argument("--sim-length-y", type=float, default=None, help=f"domain length y, mm ({d.sim.length_y})")
    g.add_argument("--resolution-scale", type=float, default=None, help=f"grid cells per mm ({d.sim.resolution_scale})")
    g.add_argument("--dt", type=float, default=None, help=f"RL timestep, s ({d.sim.dt})")
    g.add_argument("--total-time", type=float, default=None, help=f"sim horizon used for time_steps, s ({d.sim.total_time})")
    g.add_argument("--dt-substeps", type=int, default=None, help=f"fluid substeps per RL step ({d.sim.dt_substeps})")
    g.add_argument("--inflow-velocity", type=float, default=None, help=f"peak inflow centerline velocity, mm/s ({d.sim.inflow_velocity})")

    g = parser.add_argument_group("swarm")
    g.add_argument("--swarm-num-x", type=int, default=None, help=f"members along x ({d.swarm.num_x})")
    g.add_argument("--swarm-num-y", type=int, default=None, help=f"members along y ({d.swarm.num_y})")
    g.add_argument("--member-radius", type=float, default=None, help=f"member radius, mm ({d.swarm.member_radius})")
    g.add_argument("--swarm-max-force", type=float, default=None, help=f"max thrust per member, mg·mm/s² ({d.swarm.member_max_force})")

    g = parser.add_argument_group("task")
    g.add_argument("--episode-duration", type=float, default=None, help=f"episode truncation time, s ({d.task.episode_duration})")
    g.add_argument("--success-x", type=float, default=None, help=f"success if mean member x ≤ this, mm ({d.task.success_x})")
    g.add_argument("--failure-x", type=float, default=None, help=f"failure if any member x > this, mm ({d.task.failure_x})")
    g.add_argument(
        "--progress-reward",
        choices=["potential", "fluid-relative", "legacy"],
        default=None,
        help=(
            f"progress objective formulation ({d.task.progress_reward}); "
            f"prefer 'potential' for dense upstream gradients"
        ),
    )
    g.add_argument("--w-progress", type=float, default=None, help=f"progress reward weight ({d.task.w_progress})")
    g.add_argument("--w-energy", type=float, default=None, help=f"energy reward weight ({d.task.w_energy})")
    g.add_argument("--w-smooth", type=float, default=None, help=f"smoothness reward weight ({d.task.w_smooth})")

    g = parser.add_argument_group("observations")
    g.add_argument(
        "--obs-preset",
        choices=["rich", "legacy"],
        default=None,
        help=f"observation preset ({d.obs.preset})",
    )
    g.add_argument(
        "--obs-localization",
        choices=["none", "imu", "displacement", "absolute-y", "full"],
        default=None,
        help=(
            f"localization rung ({d.obs.localization}): none=no self pose/vel; "
            f"displacement=unbiased IMU deltas; imu=biased IMU; "
            f"absolute-y=lab y; full=lab pos+vel"
        ),
    )
    g.add_argument("--obs-ring-points", type=int, default=None, help=f"pressure/velocity ring samples ({d.obs.ring_points})")
    g.add_argument("--obs-history", type=int, default=None, help=f"stacked observation frames ({d.obs.history})")
    g.add_argument(
        "--neighbor-k",
        type=int,
        default=None,
        help=f"k-nearest neighbor slots ({d.obs.neighbor_k})",
    )
    g.add_argument(
        "--neighbor-radius",
        type=float,
        default=None,
        help=(
            f"neighbor sensing radius, mm; zeros any k-NN beyond this "
            f"({d.obs.neighbor_radius})"
        ),
    )
    g.add_argument(
        "--velocity-frame",
        choices=["fluid", "lab"],
        default=None,
        help=(
            f"unused for rich obs; self-velocity comes from --obs-localization "
            f"({d.obs.velocity_frame})"
        ),
    )
    g.add_argument("--imu-bias", type=float, default=None, help=f"IMU bias std, mm/s² ({d.obs.imu_bias})")
    g.add_argument("--imu-noise", type=float, default=None, help=f"IMU white-noise std per step, mm/s² ({d.obs.imu_noise})")

    g = parser.add_argument_group("training")
    g.add_argument(
        "--batch-envs",
        "--num-envs",
        dest="batch_envs",
        type=int,
        default=None,
        help=f"parallel environments in one batched step ({d.train.batch_envs})",
    )
    g.add_argument("--n-steps", type=int, default=None, help=f"rollout steps per update ({d.train.n_steps})")
    g.add_argument("--batch-size", type=int, default=None, help=f"PPO minibatch size ({d.train.batch_size})")
    g.add_argument("--update-epochs", type=int, default=None, help=f"PPO epochs per update ({d.train.update_epochs})")
    g.add_argument("--ent-coef", type=float, default=None, help=f"entropy coefficient ({d.train.ent_coef})")
    g.add_argument("--clip-coef", type=float, default=None, help=f"PPO clip coefficient ({d.train.clip_coef})")
    g.add_argument("--gamma", type=float, default=None, help=f"discount factor ({d.train.gamma})")
    g.add_argument("--lr", type=float, default=None, help=f"Adam learning rate ({d.train.lr})")
    g.add_argument("--adam-eps", type=float, default=None, help=f"Adam epsilon ({d.train.adam_eps})")
    g.add_argument(
        "--adv-norm",
        choices=["independent", "weighted", "off"],
        default=None,
        help=(
            f"advantage normalization ({d.train.adv_norm}); "
            f"'weighted' re-applies task weights after whitening so they are not cancelled"
        ),
    )
    g.add_argument(
        "--total-timesteps-per-env",
        type=int,
        default=None,
        help=(
            f"environment steps per parallel env; multiplied by batch-envs for the run "
            f"({d.train.total_timesteps_per_env})"
        ),
    )
    g.add_argument(
        "--no-pcgrad",
        action="store_true",
        default=False,
        help=f"disable PCGrad on actor gradients (default use_pcgrad={d.train.use_pcgrad})",
    )
    g.add_argument(
        "--pcgrad-protect-progress",
        action="store_true",
        default=False,
        help="never project the progress gradient away; only project energy/smooth off it",
    )
    g.add_argument(
        "--orthogonal-init",
        dest="orthogonal_init",
        action="store_true",
        default=None,
        help=f"orthogonal actor/critic init (default={d.train.orthogonal_init})",
    )
    g.add_argument(
        "--no-orthogonal-init",
        dest="orthogonal_init",
        action="store_false",
        default=None,
        help="disable orthogonal init",
    )
    g.add_argument(
        "--no-action-x-prior",
        dest="use_action_x_prior",
        action="store_false",
        default=None,
        help=f"disable −x action prior (default use_action_x_prior={d.train.use_action_x_prior})",
    )
    g.add_argument(
        "--action-x-prior-warmup-fraction",
        type=float,
        default=None,
        help=f"fraction of training to relax −x prior ({d.train.action_x_prior_warmup_fraction})",
    )
    g.add_argument(
        "--actor",
        choices=["mlp", "recurrent"],
        default=None,
        help=f"actor architecture ({d.train.actor})",
    )
    g.add_argument(
        "--device-split",
        choices=["gpu", "cpu-shard"],
        default=None,
        help=f"env device placement ({d.train.device_split})",
    )
    g.add_argument("--seed", type=int, default=None, help=f"RNG seed ({d.train.seed})")
    g.add_argument(
        "--save-fields",
        dest="save_fields",
        action="store_true",
        default=None,
        help=f"save velocity/pressure npz snapshots (default={d.train.save_fields})",
    )
    g.add_argument(
        "--no-save-fields",
        action="store_true",
        default=False,
        help="disable field saving",
    )
    g.add_argument(
        "--live-swarm-tb-every",
        type=int,
        default=None,
        help=(
            f"log env-0 swarm position image to TensorBoard every N global steps "
            f"(0=off; default {d.train.live_swarm_tb_every})"
        ),
    )
    g.add_argument(
        "--live-swarm-tb-dpi",
        type=int,
        default=None,
        help=f"TensorBoard swarm image DPI (default {d.train.live_swarm_tb_dpi})",
    )
    g.add_argument(
        "--live-swarm-tb-scale",
        type=float,
        default=None,
        help=f"TensorBoard swarm figure scale (default {d.train.live_swarm_tb_scale})",
    )
    g.add_argument(
        "--live-swarm-tb-aspect",
        type=float,
        default=None,
        help=f"TensorBoard swarm figure width:height ratio (default {d.train.live_swarm_tb_aspect})",
    )
    g.add_argument(
        "--tensorboard-port",
        type=int,
        default=None,
        help=f"TensorBoard port ({d.train.tensorboard_port})",
    )
    g.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "former run folder name or path; start a new run from that checkpoint "
            "(weights, optimizer, obs-norm, global steps; skip action-x-prior warmup)"
        ),
    )

    g = parser.add_argument_group("run")
    g.add_argument("--output-root", type=str, default=None, help=f"run output directory ({d.run.output_root})")
    g.add_argument("--tag", type=str, default=None, help=f"suffix for run folder name ({d.run.tag!r})")
    g.add_argument(
        "--wall-policy-mode",
        choices=["static", "dynamic"],
        default=None,
        help=f"brute-wall baseline row assignment ({d.run.wall_policy_mode})",
    )

    return parser


def parse_config(argv: list[str] | None = None) -> Config:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = Config.load(args.config) if args.config else Config()
    return _apply_cli_overrides(cfg, args)
