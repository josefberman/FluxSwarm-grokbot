"""Unit tests for success-rate metrics, return normalization, LR schedule."""
import numpy as np
import torch
import torch.nn as nn

from fluxswarm.agents.stability import ReturnNormalizer, SuccessTracker, lr_at


def test_window_rate_matches_legacy_list_mean():
    rng = np.random.default_rng(0)
    outs = rng.random(500) < 0.4
    tr = SuccessTracker(window=128)
    legacy = []
    for o in outs:
        tr.add(bool(o))
        legacy.append(1.0 if o else 0.0)
        assert abs(tr.window_rate() - np.mean(legacy[-128:])) < 1e-12
    assert tr.episodes_in_window == 128


def test_ema_ignores_empty_updates_and_weights_by_count():
    tr = SuccessTracker(window=32, ema_beta=0.5)
    assert tr.ema() is None
    for _ in range(4):
        tr.add(True)
    tr.end_update()
    assert tr.ema() == 1.0 and tr.last_update_rate == 1.0
    tr.end_update()  # no episodes: rate unchanged
    assert tr.ema() == 1.0 and tr.last_update_rate is None
    for _ in range(4):
        tr.add(False)
    tr.end_update()
    # num = 0.25*4 + 0 = 1 ; den = 0.25*4 + 4 = 5
    assert abs(tr.ema() - 0.2) < 1e-12


def test_mirror_excluded_from_train_and_update_rate():
    tr = SuccessTracker(window=32)
    tr.add(True, source="mirror")
    tr.add(False, source="train")
    tr.end_update()
    assert tr.window_rate() == 0.5
    assert tr.train_window_rate() == 0.0
    assert tr.mirror_window_rate() == 1.0
    assert tr.last_update_rate == 0.0


def test_popart_preserves_denormalized_outputs():
    torch.manual_seed(0)
    heads = nn.ModuleList([nn.Linear(5, 4) for _ in range(3)])
    rn = ReturnNormalizer(3, torch.device("cpu"))
    feat = torch.randn(7, 5)

    def out():
        v = torch.stack([h(feat) for h in heads], dim=1).permute(0, 2, 1)  # (7,4,3)
        return rn.denormalize(v)

    before = out().detach().clone()
    rn.update(torch.randn(100, 4, 3) * torch.tensor([50.0, 2.0, 0.5]) + torch.tensor([10.0, -1.0, 0.0]),
              heads=heads)
    assert torch.allclose(out(), before, atol=1e-4)
    assert rn.std[0] > 10 * rn.std[2]


def test_running_norm_matches_numpy():
    rn = ReturnNormalizer(2, torch.device("cpu"))
    xs = [torch.randn(50, 2) * 3 + 1 for _ in range(4)]
    for x in xs:
        rn.update(x)
    allx = torch.cat(xs).numpy()
    assert np.allclose(rn.mean.numpy(), allx.mean(0), atol=1e-3)
    assert np.allclose(rn.std.numpy(), allx.std(0), atol=1e-2)


def test_lr_schedules():
    assert lr_at(5, 10, 1e-3, "constant", 0.1) == 1e-3
    assert lr_at(0, 10, 1e-3, "linear", 0.1) == 1e-3
    assert abs(lr_at(9, 10, 1e-3, "linear", 0.1) - 1e-4) < 1e-12
    assert abs(lr_at(9, 10, 1e-3, "cosine", 0.1) - 1e-4) < 1e-12


def test_dummy_terminal_bonus_applied_on_success():
    from fluxswarm.config import Config
    from fluxswarm.envs.dummy import DummyUpstreamEnv

    cfg = Config()
    env = DummyUpstreamEnv(cfg, batch=2, n_agents=2, reward_mode="legacy", terminal_bonus=50.0)
    env.reset(seed=0)
    env.solver.swarm.pos[..., 0] = cfg.task.success_x + 0.5
    act = np.zeros((2, 2, 2), dtype=np.float32)
    act[..., 0] = -1.0
    _, _, term, _, info = env.step(act)
    assert term.all()
    assert (info["reward_matrix"][..., 0] > 50.0).all()


def test_legacy_bonus_is_opt_in_and_cli_parses():
    from fluxswarm.config import Config, parse_config

    assert Config().task.terminal_bonus == 10.0
    assert Config().task.terminal_bonus_legacy is False
    cfg = parse_config(["--terminal-bonus", "50", "--terminal-bonus-legacy", "--value-norm", "popart",
                        "--lr-schedule", "linear", "--clip-vloss"])
    assert cfg.task.terminal_bonus == 50.0 and cfg.task.terminal_bonus_legacy
    assert cfg.train.value_norm == "popart" and cfg.train.lr_schedule == "linear" and cfg.train.clip_vloss


def test_trainer_smoke_with_value_norm(tmp_path):
    from fluxswarm.agents.momappo import train_momappo
    from fluxswarm.config import Config
    from fluxswarm.envs.dummy import DummyUpstreamEnv
    import json

    cfg = Config()
    cfg.train.device = "cpu"
    cfg.train.batch_envs = 4
    cfg.train.n_steps = 16
    cfg.train.total_timesteps_per_env = 48
    cfg.train.hidden_sizes = (16, 16)
    cfg.train.value_norm = "popart"
    cfg.train.lr_schedule = "linear"
    cfg.train.live_swarm_tb_every = 0
    cfg.train.resume = False
    cfg.run.output_root = str(tmp_path)
    env = DummyUpstreamEnv(cfg, batch=4, n_agents=2, max_steps=10)
    rd = train_momappo(cfg, env=env)
    rows = [json.loads(l) for l in (rd / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert "success_rate_ema" in rows[-1] and "epochs_run" in rows[-1]
    assert (rd / "episodes.jsonl").read_text().strip()
    ck = torch.load(rd / "models" / "model_latest.pt", weights_only=False)
    assert "ret_norm" in ck


def test_per_env_inflow_clock_matches_scalar_when_synchronized():
    from fluxswarm.config import SimConfig
    from fluxswarm.physics.torch_fluid import TorchFluidSolver

    sim = SimConfig(resolution_scale=1.0)
    a = TorchFluidSolver(sim, 3, torch.device("cpu"))
    b = TorchFluidSolver(sim, 3, torch.device("cpu"))
    a.apply_inflow(0.3, 0.01)
    b.apply_inflow(torch.full((3,), 0.3, dtype=torch.float64), 0.01)
    assert torch.allclose(a.u, b.u)


def test_per_env_inflow_clock_uses_each_env_time():
    from fluxswarm.config import SimConfig
    from fluxswarm.physics.inflow import beat_waveform
    from fluxswarm.physics.torch_fluid import TorchFluidSolver

    sim = SimConfig(resolution_scale=1.0)
    f = TorchFluidSolver(sim, 2, torch.device("cpu"))
    u0 = f.u.clone()
    t = torch.tensor([0.0, 0.4], dtype=torch.float64)
    f.apply_inflow(t, 0.01)
    d = (f.u - u0).amax(dim=(1, 2))
    amp = sim.inflow_velocity
    exp = [float(beat_waveform(x + 0.01, amp) - beat_waveform(x, amp)) for x in (0.0, 0.4)]
    pmax = float(f.parabolic.max())
    for k in range(2):
        if exp[k] > 0:
            assert abs(float(d[k]) - exp[k] * pmax) < 1e-6 * max(1.0, abs(exp[k]))


def test_per_env_clock_blanks_only_fresh_envs():
    from fluxswarm.config import Config
    from fluxswarm.envs.swarm_env import BatchedSwarmEnv

    cfg = Config()
    cfg.sim.resolution_scale = 2.0
    cfg.sim.dt_substeps = 2
    cfg.sim.per_env_clock = True
    cfg.train.use_action_x_prior = False
    env = BatchedSwarmEnv(cfg, batch=2, device="cpu")
    env.reset(seed=0)
    a = np.zeros((2, env.n, 2), dtype=np.float32)
    for _ in range(3):
        env.step(a)
    env.solver.episode_time[1] = 0.0  # pretend env 1 just reset
    env.solver.step(torch.zeros(2, env.n, 2, dtype=torch.float64))
    # the pre-step ring sample of env 0 is still a real fluid sample (non-zero drag input)
    assert float(env.solver.last_drag[0].abs().sum()) > 0
