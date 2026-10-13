"""Unit tests that do not require a full fluid solve."""
from __future__ import annotations

import math

import numpy as np
import torch

from fluxswarm.agents.networks import Actor, tanh_log_prob, tanh_sample
from fluxswarm.agents.pcgrad import pcgrad_merge_actor
from fluxswarm.config import Config, parse_config
from fluxswarm.envs.observations import ObservationBuilder, obs_dim, obs_dim_stacked
from fluxswarm.envs.rewards import _energy, _smoothness
from fluxswarm.physics.inflow import beat_waveform, poiseuille_profile
from fluxswarm.physics.swarm import member_mass_2d, project_actions_to_unit_disk, resolve_collisions, stokes_drag_2d
from fluxswarm.physics.domain import Domain


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.obs.localization = "imu"
    path = tmp_path / "cfg.yaml"
    cfg.save(path)
    loaded = Config.load(path)
    assert loaded.sim.dt_substeps == 40
    assert loaded.obs.localization == "imu"


def test_cli_resume_from():
    cfg = parse_config(["--resume-from", "my_old_run", "--batch-envs", "2"])
    assert cfg.train.resume_from == "my_old_run"
    assert cfg.train.batch_envs == 2


def test_resolve_resume_dir(tmp_path):
    from fluxswarm.agents.momappo import find_resume_checkpoint, resolve_resume_dir

    root = tmp_path / "runs_new"
    run = root / "former_run"
    (run / "models").mkdir(parents=True)
    (run / "config.yaml").write_text("sim: {}\n")
    ckpt = run / "models" / "model_latest.pt"
    ckpt.write_bytes(b"x")
    assert resolve_resume_dir("former_run", root) == run.resolve()
    assert find_resume_checkpoint(run) == ckpt


def test_obs_norm_roundtrip():
    from fluxswarm.agents.momappo import RunningNorm

    device = torch.device("cpu")
    n = RunningNorm(3, device)
    n.update(torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]))
    n2 = RunningNorm(3, device)
    n2.load_state_dict(n.state_dict(), device)
    assert torch.allclose(n.mean, n2.mean)
    assert abs(n.count - n2.count) < 1e-9


def test_stokes_drag_opposes_relative_velocity():
    vel = torch.tensor([[[10.0, 0.0]]])
    fu = torch.tensor([[0.0]])
    fv = torch.tensor([[0.0]])
    f = stokes_drag_2d(vel, fu, fv, mu=3.0)
    # Agent faster than fluid in +x → drag in -x
    assert float(f[0, 0, 0]) < 0.0
    assert abs(float(f[0, 0, 1])) < 1e-9


def test_mass_is_2d_disc():
    m = member_mass_2d(15.12, 0.25)
    assert abs(m - 15.12 * math.pi * 0.25 ** 2) < 1e-9


def test_beat_waveform_period():
    a0 = beat_waveform(0.0, 400.0, 1.0)
    a1 = beat_waveform(1.0, 400.0, 1.0)
    assert abs(a0 - a1) < 1e-6


def test_poiseuille_peaks_center():
    y = torch.linspace(0, 2, 21)
    p = poiseuille_profile(y, 2.0)
    assert float(p[10]) > float(p[0])
    assert float(p[0]) < 1e-6


def test_unit_disk_projection():
    a = torch.tensor([[[3.0, 4.0]]])
    p = project_actions_to_unit_disk(a)
    assert abs(float(torch.linalg.norm(p)) - 1.0) < 1e-6


def test_obs_dim_no_position_in_rich_none():
    cfg = Config().obs
    cfg.preset = "rich"
    cfg.localization = "none"
    d = obs_dim(cfg, 16)
    # rings + neighbor_k slots + prev action; no self velocity unless localization adds it
    assert d == cfg.ring_points + 2 * cfg.ring_points + cfg.neighbor_k * 4 + 2
    assert obs_dim_stacked(cfg, 16) == d * cfg.history
    cfg.localization = "displacement"
    assert obs_dim(cfg, 16) == d + 4
    cfg.localization = "full"
    assert obs_dim(cfg, 16) == d + 4


def test_tanh_logprob_consistency():
    actor = Actor(8, 2, (32, 32))
    obs = torch.randn(4, 3, 8)
    dist, _ = actor.dist(obs)
    action, logp = tanh_sample(dist)
    logp2 = tanh_log_prob(dist, action)
    assert torch.allclose(logp, logp2, atol=1e-4)


def test_pcgrad_actor_only():
    actor = Actor(4, 2, (16, 16))
    obs = torch.randn(8, 4)
    dist, _ = actor.dist(obs)
    action, logp = tanh_sample(dist)
    # Fake advantages
    loss1 = -(logp.sum() * 1.0)
    loss2 = -(logp.sum() * -0.5)
    pcgrad_merge_actor(actor, [loss1, loss2])
    grads = [p.grad for p in actor.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_energy_and_smooth_bounds():
    class _Swarm:
        action = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        prev_action = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    class _S:
        swarm = _Swarm()

    e = _energy(_S())
    sm = _smoothness(_S())
    assert float(e[0, 0]) == 0.0
    assert float(e[0, 1]) == 1.0
    assert abs(float(sm[0, 0]) - 1.0) < 1e-6
    assert abs(float(sm[0, 1]) - 1.0) < 1e-6


def test_domain_clamp():
    d = Domain(100, 2, 1000, 20, 0.1, 0.1)
    pos = torch.tensor([[[ -1.0, 0.5], [50.0, 5.0]]])
    radii = torch.tensor([[0.25, 0.25]])
    clamped, hit = d.clamp_positions(pos, radii)
    assert hit.any()
    assert float(clamped[0, 0, 0]) >= 0.25
    assert float(clamped[0, 1, 1]) <= 2.0 - 0.25


def test_advantage_whitening_cancels_weights():
    from fluxswarm.agents.momappo import normalize_objective_advantages

    torch.manual_seed(0)
    base = torch.randn(20, 4, 3)
    a9 = normalize_objective_advantages(base * 9.0, mode="independent", weight=9.0)
    a1 = normalize_objective_advantages(base * 1.0, mode="independent", weight=1.0)
    assert torch.allclose(a9, a1, atol=1e-5)

    z9 = normalize_objective_advantages(base * 9.0, mode="weighted", weight=9.0)
    z1 = normalize_objective_advantages(base * 1.0, mode="weighted", weight=1.0)
    assert float(z9.abs().mean()) > 5.0 * float(z1.abs().mean())


def test_advantage_degenerate_std_is_zero():
    from fluxswarm.agents.momappo import normalize_objective_advantages

    const = torch.full((8, 2, 3), -0.1)
    out = normalize_objective_advantages(const, mode="independent", weight=1.0)
    assert torch.equal(out, torch.zeros_like(out))


def test_gae_masks_done():
    from fluxswarm.agents.momappo import compute_gae

    T, B, N = 3, 1, 1
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]]])
    values = torch.zeros(T, B, N)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last_v = torch.tensor([[100.0]])
    last_done = torch.zeros(B)
    adv, ret = compute_gae(rewards, values, dones, last_v, last_done, gamma=0.99, lam=1.0)
    # t=1 is done: no bootstrap, return == reward
    assert abs(float(ret[1, 0, 0]) - 1.0) < 1e-5
    # t=2 is a new episode; last_v bootstraps
    assert float(ret[2, 0, 0]) > 50.0


def test_dummy_autoreset_returns_start_obs():
    from fluxswarm.config import Config
    from fluxswarm.envs.dummy import DummyUpstreamEnv

    cfg = Config()
    env = DummyUpstreamEnv(cfg, batch=2, n_agents=3, max_steps=2, drift=5.0, speed=0.0)
    obs, _ = env.reset(seed=0)
    start_x = float(obs[0, 0, 0])
    actions = np.zeros((2, 3, 2))
    saw_reset = False
    for _ in range(5):
        next_obs, _, terms, truncs, info = env.step(actions)
        if bool(terms[0] or truncs[0]):
            assert abs(float(next_obs[0, 0, 0]) - start_x) < 1e-4
            saw_reset = True
    assert saw_reset


def test_obs_history_reset_envs():
    cfg = Config().obs
    cfg.preset = "rich"
    cfg.history = 3
    cfg.localization = "none"
    builder = ObservationBuilder(cfg, num_members=2, batch=2, device=torch.device("cpu"))
    frame = torch.ones(2, 2, builder.frame_dim)
    builder._history.append(frame.clone())
    builder._history.append(frame.clone())
    builder.reset_envs(torch.tensor([1]))
    assert float(builder._history[0][0].sum()) != 0
    assert float(builder._history[0][1].sum()) == 0


def test_cli_learning_flags():
    cfg = parse_config(
        [
            "--progress-reward",
            "legacy",
            "--obs-localization",
            "full",
            "--no-action-x-prior",
            "--batch-envs",
            "16",
            "--total-timesteps-per-env",
            "100000",
            "--adv-norm",
            "weighted",
            "--pcgrad-protect-progress",
            "--orthogonal-init",
            "--batch-size",
            "256",
            "--lr",
            "0.0003",
        ]
    )
    assert cfg.task.progress_reward == "legacy"
    assert cfg.obs.localization == "full"
    assert cfg.train.use_action_x_prior is False
    assert cfg.train.batch_envs == 16
    assert cfg.train.total_timesteps_per_env == 100000
    assert cfg.train.adv_norm == "weighted"
    assert cfg.train.pcgrad_protect_progress is True
    assert cfg.train.orthogonal_init is True
    assert cfg.train.batch_size == 256
    assert abs(cfg.train.lr - 3e-4) < 1e-12
