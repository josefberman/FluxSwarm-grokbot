"""GPU Navier–Stokes with Brinkman obstacles and a DCT Poisson solve.

Two-way Brinkman/DCT solver. The Poisson matrix stays obstacle-free, so each
pressure projection is a pair of batched DCTs instead of SciPy CG+ILU.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from fluxswarm.config import SimConfig
from fluxswarm.physics.inflow import beat_waveform, poiseuille_on_grid


def _dct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unnormalized DCT-II along ``dim`` (Makhoul FFT trick)."""
    x = x.transpose(dim, -1)
    n = x.shape[-1]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    vc = torch.fft.fft(v, dim=-1)
    k = -torch.arange(n, device=x.device, dtype=x.dtype) * math.pi / (2 * n)
    out = 2.0 * (vc.real * torch.cos(k) - vc.imag * torch.sin(k))
    return out.transpose(dim, -1)


def _idct(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`_dct` (DCT-III with matching scale)."""
    x = x.transpose(dim, -1)
    n = x.shape[-1]
    xv = x / 2.0
    k = torch.arange(n, device=x.device, dtype=x.dtype) * math.pi / (2 * n)
    w_r, w_i = torch.cos(k), torch.sin(k)
    vt_i = torch.cat([torch.zeros_like(xv[..., :1]), -xv.flip(-1)[..., :-1]], dim=-1)
    time = torch.fft.ifft(torch.complex(xv * w_r - vt_i * w_i, xv * w_i + vt_i * w_r), dim=-1).real
    out = torch.empty_like(time)
    out[..., ::2] = time[..., : n - (n // 2)]
    out[..., 1::2] = time.flip(-1)[..., : n // 2]
    return out.transpose(dim, -1)


def _dct2(x: torch.Tensor) -> torch.Tensor:
    return _dct(_dct(x, dim=-1), dim=-2)


def _idct2(x: torch.Tensor) -> torch.Tensor:
    return _idct(_idct(x, dim=-2), dim=-1)


def _sample_bilinear(
    field: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    """Bilinear sample of ``field`` (B, ni, nj) at physical ``x,y``."""
    _, ni, nj = field.shape
    gx = (x - x0) / dx
    gy = (y - y0) / dy
    i0 = gx.floor().long().clamp(0, max(ni - 2, 0))
    j0 = gy.floor().long().clamp(0, max(nj - 2, 0))
    i1 = (i0 + 1).clamp(max=ni - 1)
    j1 = (j0 + 1).clamp(max=nj - 1)
    wx = (gx - i0.to(gx.dtype)).clamp(0.0, 1.0)
    wy = (gy - j0.to(gy.dtype)).clamp(0.0, 1.0)
    b = torch.arange(field.shape[0], device=field.device).view(-1, *([1] * (gx.ndim - 1)))
    b = b.expand_as(i0)
    f00 = field[b, i0, j0]
    f01 = field[b, i0, j1]
    f10 = field[b, i1, j0]
    f11 = field[b, i1, j1]
    return (1.0 - wx) * (1.0 - wy) * f00 + (1.0 - wx) * wy * f01 + wx * (1.0 - wy) * f10 + wx * wy * f11


def _nn_sample(
    field: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
) -> torch.Tensor:
    _, ni, nj = field.shape
    ix = ((x - x0) / dx).round().long().clamp(0, ni - 1)
    iy = ((y - y0) / dy).round().long().clamp(0, nj - 1)
    b = torch.arange(field.shape[0], device=field.device).view(-1, *([1] * (x.ndim - 1)))
    return field[b.expand_as(ix), ix, iy]


class TorchFluidSolver:
    """Batched MAC-grid fluid with Brinkman discs and DCT pressure projection."""

    def __init__(self, sim: SimConfig, batch: int, device: torch.device):
        self.sim = sim
        self.batch = batch
        self.device = device
        self.dtype = torch.float64
        self.nx, self.ny = sim.resolution
        self.dx, self.dy = sim.dx, sim.dy
        self.lx, self.ly = sim.length_x, sim.length_y

        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        k = torch.arange(nx, device=device, dtype=self.dtype)
        l = torch.arange(ny, device=device, dtype=self.dtype)
        lam = (2.0 * torch.cos(math.pi * k / nx) - 2.0) / dx**2
        lam = lam[:, None] + (2.0 * torch.cos(math.pi * l / ny) - 2.0) / dy**2
        lam[0, 0] = 1.0
        self.lam = lam

        self.parabolic = poiseuille_on_grid(ny, sim.length_y, device=device)
        self.delta_chi = 0.5 * min(dx, dy)
        self.last_chi_u: torch.Tensor | None = None
        self.last_chi_v: torch.Tensor | None = None

        self.u = torch.zeros(batch, nx + 1, ny, device=device, dtype=self.dtype)
        self.v = torch.zeros(batch, nx, ny + 1, device=device, dtype=self.dtype)
        self.p = torch.zeros(batch, nx, ny, device=device, dtype=self.dtype)
        self.apply_inflow(0.0, 0.0, absolute=True)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None or env_ids.numel() == self.batch:
            self.u.zero_()
            self.v.zero_()
            self.p.zero_()
            self.apply_inflow(0.0, 0.0, absolute=True)
            return
        ids = env_ids.long()
        self.u[ids] = 0
        self.v[ids] = 0
        self.p[ids] = 0
        self.apply_inflow(0.0, 0.0, absolute=True, env_ids=ids)

    def apply_inflow(
        self,
        t: float,
        dt_sub: float,
        absolute: bool = False,
        chi_u: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        amp, period = self.sim.inflow_velocity, self.sim.inflow_period
        if absolute:
            scale = beat_waveform(t, amp, period)
            profile = scale * self.parabolic[None, None, :]
            if env_ids is None:
                self.u[:] = profile
                self.v.zero_()
            else:
                ids = env_ids.long()
                self.u[ids] = profile
                self.v[ids] = 0
            return
        scale = beat_waveform(t + dt_sub, amp, period) - beat_waveform(t, amp, period)
        if isinstance(scale, torch.Tensor) and scale.dim() > 0:
            # per-env clock: (B,) -> (B, 1, 1)
            scale = scale.to(self.u.dtype).view(-1, 1, 1)
        delta = scale * self.parabolic[None, None, :]
        if chi_u is not None:
            delta = delta * (1.0 - chi_u)
        self.u.add_(delta)

    def _lap_u(self) -> torch.Tensor:
        u = self.u
        ux = F.pad(u, (0, 0, 1, 1), mode="replicate")
        d2x = (ux[:, 2:, :] - 2.0 * u + ux[:, :-2, :]) / self.dx**2
        uy = torch.cat([-u[:, :, :1], u, -u[:, :, -1:]], dim=-1)
        d2y = (uy[:, :, 2:] - 2.0 * u + uy[:, :, :-2]) / self.dy**2
        return d2x + d2y

    def _lap_v(self) -> torch.Tensor:
        v = self.v
        vx = F.pad(v, (0, 0, 1, 1), mode="replicate")
        d2x = (vx[:, 2:, :] - 2.0 * v + vx[:, :-2, :]) / self.dx**2
        vy = F.pad(v, (1, 1, 0, 0), mode="constant", value=0.0)
        d2y = (vy[:, :, 2:] - 2.0 * v + vy[:, :, :-2]) / self.dy**2
        return d2x + d2y

    def diffuse(
        self,
        nu: float,
        dt: float,
        chi_u: torch.Tensor | None = None,
        chi_v: torch.Tensor | None = None,
    ) -> None:
        du = dt * nu * self._lap_u()
        dv = dt * nu * self._lap_v()
        if chi_u is not None:
            du = du * (1.0 - chi_u)
        if chi_v is not None:
            dv = dv * (1.0 - chi_v)
        self.u = self.u + du
        self.v = self.v + dv
        self.enforce_bc()

    def _v_on_u(self) -> torch.Tensor:
        v_at_u_y = 0.5 * (self.v[:, :, :-1] + self.v[:, :, 1:])
        out = torch.empty_like(self.u)
        out[:, 1:-1, :] = 0.5 * (v_at_u_y[:, :-1, :] + v_at_u_y[:, 1:, :])
        out[:, 0, :] = v_at_u_y[:, 0, :]
        out[:, -1, :] = v_at_u_y[:, -1, :]
        return out

    def _u_on_v(self) -> torch.Tensor:
        u_at_v_x = 0.5 * (self.u[:, :-1, :] + self.u[:, 1:, :])
        out = torch.empty_like(self.v)
        out[:, :, 1:-1] = 0.5 * (u_at_v_x[:, :, :-1] + u_at_v_x[:, :, 1:])
        out[:, :, 0] = u_at_v_x[:, :, 0]
        out[:, :, -1] = u_at_v_x[:, :, -1]
        return out

    def advect(self, dt: float) -> None:
        dx, dy, nx, ny = self.dx, self.dy, self.nx, self.ny
        iu = torch.arange(nx + 1, device=self.device, dtype=self.dtype)[:, None] * dx
        ju = (torch.arange(ny, device=self.device, dtype=self.dtype) + 0.5) * dy
        xu = iu - self.u * dt
        yu = ju - self._v_on_u() * dt
        u_new = _sample_bilinear(self.u, xu, yu, 0.0, 0.5 * dy, dx, dy)

        iv = (torch.arange(nx, device=self.device, dtype=self.dtype) + 0.5) * dx
        jv = torch.arange(ny + 1, device=self.device, dtype=self.dtype)[None, :] * dy
        xv = iv[:, None] - self._u_on_v() * dt
        yv = jv - self.v * dt
        v_new = _sample_bilinear(self.v, xv, yv, 0.5 * dx, 0.0, dx, dy)

        self.u, self.v = u_new, v_new
        self.enforce_bc()

    def rasterize(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        radii: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft obstacle masks and body velocity on the u/v faces."""
        chi_u, u_obs = self._mask_on_faces(pos, vel, radii, x_off=0.0, y_off=0.5, ni=self.nx + 1, nj=self.ny)
        chi_v, v_obs = self._mask_on_faces(pos, vel, radii, x_off=0.5, y_off=0.0, ni=self.nx, nj=self.ny + 1)
        return chi_u, chi_v, u_obs, v_obs

    def _mask_on_faces(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        radii: torch.Tensor,
        x_off: float,
        y_off: float,
        ni: int,
        nj: int,
        component: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs = (torch.arange(ni, device=self.device, dtype=self.dtype) + x_off) * self.dx
        ys = (torch.arange(nj, device=self.device, dtype=self.dtype) + y_off) * self.dy
        dx = xs[None, None, :, None] - pos[:, :, 0, None, None]
        dy = ys[None, None, None, :] - pos[:, :, 1, None, None]
        dist = torch.sqrt(dx * dx + dy * dy)
        sdf_disc = dist - radii[:, :, None, None]
        sdf_min, idx = sdf_disc.min(dim=1)
        # Disc ∪ channel walls (half-planes y≤0 and y≥L) so a touching body
        # does not poke through the no-slip boundary. obs still comes from the
        # nearest disc; wall faces override below.
        sdf_wall = torch.minimum(ys[None, None, :], self.ly - ys[None, None, :])
        sdf_min = torch.minimum(sdf_min, sdf_wall)
        chi = 0.5 * (1.0 - torch.tanh(sdf_min / self.delta_chi))
        b = torch.arange(self.batch, device=self.device)[:, None, None].expand_as(idx)
        if component is None:
            component = 0 if x_off == 0.0 else 1
        obs = vel[b, idx, component]
        # Wall faces: no-slip, never write agent velocity onto the wall layer.
        if y_off == 0.0:
            chi[:, :, 0] = 1.0
            chi[:, :, -1] = 1.0
            obs[:, :, 0] = 0.0
            obs[:, :, -1] = 0.0
        else:
            obs[:, :, 0] = 0.0
            obs[:, :, -1] = 0.0
        return chi, obs

    def brinkman(
        self,
        chi_u: torch.Tensor,
        chi_v: torch.Tensor,
        u_obs: torch.Tensor,
        v_obs: torch.Tensor,
        dt: float,
        eta: float | None = None,
    ) -> None:
        """Implicit finite-η volume penalization (still two-way)."""
        if eta is None:
            eta = max(dt, 1e-12)
        k = (dt / eta) * chi_u
        self.u = (self.u + k * u_obs) / (1.0 + k)
        k = (dt / eta) * chi_v
        self.v = (self.v + k * v_obs) / (1.0 + k)

    def project(self) -> None:
        div = (self.u[:, 1:, :] - self.u[:, :-1, :]) / self.dx + (self.v[:, :, 1:] - self.v[:, :, :-1]) / self.dy
        div = div - div.mean(dim=(-2, -1), keepdim=True)
        self.p = _idct2(_dct2(div) / self.lam)
        self.u[:, 1:-1, :] = self.u[:, 1:-1, :] - (self.p[:, 1:, :] - self.p[:, :-1, :]) / self.dx
        self.v[:, :, 1:-1] = self.v[:, :, 1:-1] - (self.p[:, :, 1:] - self.p[:, :, :-1]) / self.dy
        self.enforce_bc()

    def enforce_bc(self) -> None:
        self.v[:, :, 0] = 0.0
        self.v[:, :, -1] = 0.0
        # Wall-adjacent staggered-u rows (cell centers nearest y=0 and y=L).
        self.u[:, :, 0] = 0.0
        self.u[:, :, -1] = 0.0

    def substep_loop(
        self,
        t0: "float | torch.Tensor",
        dt: float,
        n_sub: int,
        pos: torch.Tensor,
        vel: torch.Tensor,
        radii: torch.Tensor,
        nu: float,
    ) -> None:
        dt_sub = dt / n_sub
        chi_u, chi_v, u_obs, v_obs = self.rasterize(pos, vel, radii)
        self.last_chi_u, self.last_chi_v = chi_u, chi_v
        eta = dt_sub
        for s in range(n_sub):
            t_sub = t0 + s * dt_sub
            self.apply_inflow(t_sub, dt_sub, absolute=False, chi_u=chi_u)
            self.diffuse(nu, dt_sub, chi_u=chi_u, chi_v=chi_v)
            self.advect(dt_sub)
            self.brinkman(chi_u, chi_v, u_obs, v_obs, dt_sub, eta=eta)
            self.project()
            self.brinkman(chi_u, chi_v, u_obs, v_obs, dt_sub, eta=eta)
            self.enforce_bc()

    def max_speed(self, env_index: int = 0) -> float:
        i = env_index
        return float(torch.maximum(self.u[i].abs().amax(), self.v[i].abs().amax()).item())

    def max_pressure(self, env_index: int = 0) -> float:
        return float(self.p[env_index].abs().amax().item())

    def chi_wall(self, env_index: int = 0) -> float:
        if self.last_chi_v is None:
            return 0.0
        chi = self.last_chi_v[env_index]
        return float(0.5 * (chi[:, 0].mean() + chi[:, -1].mean()).item())

    def sample_at(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = _nn_sample(self.p, x, y, 0.5 * self.dx, 0.5 * self.dy, self.dx, self.dy)
        u = _nn_sample(self.u, x, y, 0.0, 0.5 * self.dy, self.dx, self.dy)
        v = _nn_sample(self.v, x, y, 0.5 * self.dx, 0.0, self.dx, self.dy)
        return p, u, v

    def export(self, env_index: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = env_index
        return (
            self.u[i].detach().cpu().numpy(),
            self.v[i].detach().cpu().numpy(),
            self.p[i].detach().cpu().numpy(),
        )
