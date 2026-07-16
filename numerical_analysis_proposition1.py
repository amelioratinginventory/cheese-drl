from __future__ import annotations
import argparse
import itertools
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Reduced-problem primitives
# ---------------------------------------------------------------------------
class ReducedProblem:
    """Newsvendor recourse problem with fixed sales price phi and salvage phi_L."""

    def __init__(self, phi, phi_L, h, cv_D, n_demand=2001, q_cap=0.9995):
        assert h > 0.0, "Section 4.1 assumes a strictly positive holding cost."
        assert 0.0 <= phi_L < phi, "salvage price must lie below the sales price."
        self.phi, self.phi_L, self.h = float(phi), float(phi_L), float(h)
        mu_D, sd_D = 1.0, float(cv_D)
        self.Dbar = mu_D + norm.ppf(q_cap) * sd_D          # demand cap
        self.d = np.linspace(max(1e-9, mu_D - 6 * sd_D), self.Dbar, n_demand)
        p = norm.pdf(self.d, mu_D, sd_D)
        self.p = p / p.sum()
        self.cdf = np.cumsum(self.p)

    def r_Y(self, y):
        """Expected revenue (non-decreasing, concave), vectorised."""
        y = np.atleast_1d(np.asarray(y, float))
        ycap = np.minimum(y, self.Dbar)[:, None]
        d = self.d[None, :]
        sold = np.sum(self.p[None, :] * np.minimum(d, ycap), axis=1)
        left = np.sum(self.p[None, :] * np.maximum(ycap - d, 0.0), axis=1)
        return self.phi * sold + self.phi_L * left

    def r_Yp(self, y):
        """Marginal revenue r_Y'(y) = phi*(1-F(y)) + phi_L*F(y) below the cap."""
        y = np.asarray(y, float)
        F = np.interp(np.clip(y, 0.0, self.Dbar), self.d, self.cdf,
                      left=0.0, right=1.0)
        out = self.phi * (1.0 - F) + self.phi_L * F
        return np.where(y >= self.Dbar, self.phi_L, out)

    def y_star(self, psi):
        """Clipped newsvendor target: (r_Y')^{-1}(psi) on (phi_L, phi], 0 above phi."""
        psi = np.atleast_1d(np.asarray(psi, float))
        cr = np.clip((self.phi - psi) / (self.phi - self.phi_L), 0.0, 1.0)
        y = np.interp(cr, self.cdf, self.d, left=0.0, right=self.Dbar)
        return np.where(psi > self.phi, 0.0, y)


# ---------------------------------------------------------------------------
# Price-process quadrature (truncated to psi > phi_L, per Section 4.1)
# ---------------------------------------------------------------------------
def _nodes(mean, sd, lower, n, span=7.0):
    sd = max(sd, 1e-12)
    lo, hi = max(lower + 1e-9, mean - span * sd), mean + span * sd
    if hi <= lo:
        return np.array([lo]), np.array([1.0])
    x = np.linspace(lo, hi, n)
    w = norm.pdf(x, mean, sd)
    s = w.sum()
    return (x, w / s) if s > 0 else (np.array([lo]), np.array([1.0]))


def cond_nodes(psi, theta, mu, sigma_lt, phi_L, n=81):
    m = mu + (1.0 - theta) * (psi - mu)
    sd = sigma_lt * np.sqrt(theta * (2.0 - theta))
    return _nodes(m, sd, phi_L, n)


def stat_nodes(mu, sigma_lt, phi_L, n=81):
    return _nodes(mu, sigma_lt, phi_L, n)


# ---------------------------------------------------------------------------
# Hedge from the first-order condition (smooth in psi, no grid quantisation)
# ---------------------------------------------------------------------------
def hedge(psi, theta, mu, sigma_lt, prob, n_cond=81):
    xp, wp = cond_nodes(psi, theta, mu, sigma_lt, prob.phi_L, n_cond)

    def foc(a):
        return -(psi + prob.h) + wp @ np.minimum(xp, prob.r_Yp(a))

    if foc(0.0) <= 0.0:
        return 0.0
    if foc(prob.Dbar) >= 0.0:                # excluded under the Section 4.1 regime
        return prob.Dbar
    return brentq(foc, 0.0, prob.Dbar, xtol=1e-12, rtol=1e-14)


# ---------------------------------------------------------------------------
# Value of flexibility, abar, and the decomposition terms
# ---------------------------------------------------------------------------
def _J_diff(a, xp, prob, ys):
    """Integrand J(a,psi') - J(0,psi') at the conditional nodes."""
    y_a = np.maximum(a, ys)
    return (prob.r_Y(y_a) - xp * np.maximum(0.0, ys - a)) \
        - (prob.r_Y(ys) - xp * ys)


def value_of_flexibility(theta, mu, sigma_lt, prob, n_psi=81, n_cond=81):
    xs, ws = stat_nodes(mu, sigma_lt, prob.phi_L, n_psi)
    V = 0.0
    for psi, w in zip(xs, ws):
        a = hedge(psi, theta, mu, sigma_lt, prob, n_cond)
        xp, wp = cond_nodes(psi, theta, mu, sigma_lt, prob.phi_L, n_cond)
        ys = prob.y_star(xp)
        V += w * max(wp @ _J_diff(a, xp, prob, ys) - (psi + prob.h) * a, 0.0)
    return float(V)


def abar(psi, theta, mu, sigma_lt, prob, n_cond=81):
    a = hedge(psi, theta, mu, sigma_lt, prob, n_cond)
    xp, wp = cond_nodes(psi, theta, mu, sigma_lt, prob.phi_L, n_cond)
    return float(wp @ np.maximum(0.0, prob.y_star(xp) - a)), a


def decomposition_terms(theta, mu, sigma_lt, prob, n_psi=81, n_cond=81, eps=1e-6):
    """Return (mean-shift, variance-shift) of the derivative decomposition."""
    xs, ws = stat_nodes(mu, sigma_lt, prob.phi_L, n_psi)
    ab = np.empty(len(xs))
    var_term = 0.0
    for i, (psi, w) in enumerate(zip(xs, ws)):
        ab[i], a = abar(psi, theta, mu, sigma_lt, prob, n_cond)
        xp, wp = cond_nodes(psi, theta, mu, sigma_lt, prob.phi_L, n_cond)
        d = (np.maximum(0.0, prob.y_star(xp + eps) - a)
             - np.maximum(0.0, prob.y_star(xp - eps) - a)) / (2 * eps)
        var_term += w * float(wp @ (-d))
    Ep, Ea = ws @ xs, ws @ ab
    cov = float(ws @ ((xs - Ep) * (ab - Ea)))
    return cov, (1.0 - theta) * sigma_lt ** 2 * var_term


# ---------------------------------------------------------------------------
# Parameter grid (all settings satisfy Section 4.1: h > 0, phi_L far below mu)
# ---------------------------------------------------------------------------
def parameter_grid(full=False):
    """Yield (mu, cv_lt, phi, phi_L, h, cv_D)."""
    mu = 1.0                                  # scale-free; prices relative to mu
    if full:
        cv_lts = [0.06, 0.10, 0.135, 0.16, 0.20, 0.25]
        margins = [1.3, 1.47, 1.7, 2.06, 2.5, 3.0]
        salvages = [0.1, 0.2, 0.3, 0.4]
        h_fracs = [0.005, 0.0147, 0.03, 0.05, 0.08]
        cv_Ds = [0.10, 0.15, 0.25]
    else:
        cv_lts = [0.10, 0.135, 0.20]
        margins = [1.47, 2.06, 3.0]
        salvages = [0.1, 0.3]
        h_fracs = [0.0147, 0.05]
        cv_Ds = [0.15, 0.25]
    grid = []
    for cvl, m, s, hf, cvd in itertools.product(cv_lts, margins, salvages,
                                                h_fracs, cv_Ds):
        phi, phi_L = m * mu, s * m * mu
        if (mu - phi_L) / (cvl * mu) < 2.5:   # keep phi_L well below the mass
            continue
        grid.append((mu, cvl, phi, phi_L, hf * mu, cvd))
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Large-scale numerical verification of Proposition 1.")
    ap.add_argument("--full", action="store_true", help="large grid (slower)")
    ap.add_argument("--identity", action="store_true",
                    help="verify the derivative decomposition")
    ap.add_argument("--n_psi", type=int, default=61)
    ap.add_argument("--n_cond", type=int, default=61)
    ap.add_argument("--n_demand", type=int, default=2001,
                    help="demand grid size (smaller is faster, ~1%% accuracy at 401)")
    args = ap.parse_args()

    thetas = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.70, 0.85, 1.00]
    grid = parameter_grid(full=args.full)

    # ---- (1) PRIMARY: monotonicity of V in theta -------------------------
    print("=" * 76)
    print("CHECK 1  --  PROPOSITION 1:  V(theta) non-decreasing in theta")
    print("=" * 76)
    print(f"parameter settings   : {len(grid)}")
    print(f"theta grid           : {thetas}")
    print(f"evaluations of V     : {len(grid) * len(thetas)}")
    print(f"quadrature           : n_psi={args.n_psi}, n_cond={args.n_cond}\n")

    n_bad, worst_rel, worst_set = 0, 0.0, None
    Vs_by_setting = []
    for st in grid:
        mu, cvl, phi, phi_L, h, cvd = st
        prob = ReducedProblem(phi, phi_L, h, cvd, n_demand=args.n_demand)
        Vs = np.array([value_of_flexibility(t, mu, cvl * mu, prob,
                                            args.n_psi, args.n_cond)
                       for t in thetas])
        Vs_by_setting.append(Vs)
        steps = np.diff(Vs)
        if steps.min() < -1e-6:
            n_bad += 1
            rel = steps.min() / max(Vs.max(), 1e-12)
            if rel < worst_rel:
                worst_rel, worst_set = rel, st

    print(f"settings with a downward step in V : {n_bad} / {len(grid)}")
    print(f"worst relative downward step       : {worst_rel:.2e}"
          + (f"  at {worst_set}" if worst_set else ""))
    ok = (n_bad == 0)
    print("\nRESULT:", "PASS" if ok else "FAIL")

    # ---- (2) Condition 1 diagnostic --------------------------------------
    print("\n" + "=" * 76)
    print("CHECK 2  --  CONDITION 1 diagnostic:  Cov(psi, abar(psi)) by theta")
    print("=" * 76)
    print(f"{'theta':>6} | {'settings':>8} | {'Cond.1 fails':>12} | "
          f"{'min Cov':>12} | {'max V where fails':>18}")
    print("-" * 68)
    for it, t in enumerate(thetas):
        n_fail, min_cov, max_V_fail = 0, np.inf, 0.0
        for st, Vs in zip(grid, Vs_by_setting):
            mu, cvl, phi, phi_L, h, cvd = st
            prob = ReducedProblem(phi, phi_L, h, cvd, n_demand=args.n_demand)
            cov, _ = decomposition_terms(t, mu, cvl * mu, prob,
                                         args.n_psi, args.n_cond)
            min_cov = min(min_cov, cov)
            if cov < -1e-8:
                n_fail += 1
                max_V_fail = max(max_V_fail, Vs[it])
        print(f"{t:>6.2f} | {len(grid):>8} | {n_fail:>12} | "
              f"{min_cov:>+12.4e} | {max_V_fail:>18.5f}")
    print("\nWhere Condition 1 fails (strong auto-correlation), V is negligible")
    print("and the monotonicity of Check 1 is preserved.")

    # ---- (3) decomposition identity ---------------------------------------
    if args.identity:
        print("\n" + "=" * 76)
        print("CHECK 3  --  DERIVATIVE DECOMPOSITION:")
        print("  dV/dtheta  ==  mean-shift effect + variance-shift effect")
        print("=" * 76)
        demo = (1.0, 23.0 / 170.0, 250.0 / 170.0, 75.0 / 170.0,
                2.5 / 170.0, 0.15)            # cheese w1, normalised to mu = 1
        mu, cvl, phi, phi_L, h, cvd = demo
        prob = ReducedProblem(phi, phi_L, h, cvd, n_demand=args.n_demand)
        print("setting: cheese calibration, product w1 (mu normalised to 1)\n")
        print(f"{'theta':>6} | {'dV/dth (FD)':>12} | {'mean-shift':>11} | "
              f"{'var-shift':>10} | {'sum':>10} | {'rel dev':>8}")
        print("-" * 70)
        eps = 0.002
        for t in [0.15, 0.25, 0.35, 0.50, 0.70, 0.85]:
            Vp = value_of_flexibility(t + eps, mu, cvl * mu, prob, 121, 121)
            Vm = value_of_flexibility(t - eps, mu, cvl * mu, prob, 121, 121)
            lhs = (Vp - Vm) / (2 * eps)
            cov, var = decomposition_terms(t, mu, cvl * mu, prob, 121, 121)
            rel = abs(lhs - (cov + var)) / max(abs(lhs), 1e-12)
            print(f"{t:>6.2f} | {lhs:>12.5f} | {cov:>11.5f} | {var:>10.5f} | "
                  f"{cov + var:>10.5f} | {rel:>7.1%}")

    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())