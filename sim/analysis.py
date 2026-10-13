"""Observables for the bilayer GL runs: layer-resolved S(q), peak lists, mode amplitudes, seeds.

Conventions (REQUEST sec. 4): S_i(q)=sum_alpha |m^_ia(q)|^2 ; for a Neel cycloid of amplitude a the z-Fourier
coefficient at +-k is a N/2, so a_ij = 2|m^_iz(k_j)|/N, a_j = sqrt(sum_i a_ij^2); Landau amplitude = a_j/sqrt(|r|/u).
Energy scale: (energy per site)/(|r|^2/u) = Landau F.
"""
import numpy as np

# ---------------------------------------------------------------- seeds
def _round_k(k, L):
    """nearest box wavevector (integer pair) to a continuous k (rad/site)"""
    n = np.rint(np.asarray(k) * L / (2 * np.pi)).astype(int)
    return n


def cycloid_seed(modes, L, p, s_layer):
    """Field from modes=[(k(2) continuous, amp_landau)], amplitude in units sqrt(|r|/u).
    Layer weights s_layer(scale)->(w_layer1, w_layer2); `modes` entries are (k, amp, scale). Phases all zero
    (all amplitudes negative => triad phase sum pi, REQUEST sec. 4)."""
    x = np.arange(L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    m = np.zeros((6, L, L))
    base = np.sqrt(p['r_abs'] / p['u'])
    used = {}
    for k, a, sc in modes:
        n = _round_k(k, L)
        kk = 2 * np.pi * n / L
        kn = np.hypot(*kk)
        if kn == 0:
            continue
        kh = kk / kn
        ph = kk[0] * X + kk[1] * Y
        w = s_layer(sc)
        for i in range(2):
            amp = base * a * np.sqrt(w[i])
            m[3 * i + 2] += amp * np.cos(ph)
            m[3 * i + 0] += amp * kh[0] * np.sin(ph)
            m[3 * i + 1] += amp * kh[1] * np.sin(ph)
        used[tuple(n)] = used.get(tuple(n), 0) + 1
    return m


def layer_weights(c_rel):
    s = np.sqrt(1 - c_rel ** 2)
    w1 = ((1 + s) / 2, (1 - s) / 2)
    return {1: w1, 2: (w1[1], w1[0])}  # scale-1 (q_-) modes live mostly in layer 1; scale 2 mirrored


def _dirs(Q, angles):
    return [Q * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) for a in angles]


def qc_seed(p, L, amps=(0.607, 0.148, 0.619, 0.132), amp_sign=-1.0, big_first=True):
    """6-fold quasicrystal seed: rings at q_- (Q1) and R12*q_- (Q2), R12=2cos(pi/12), two C6 orbits each.
    amps = Landau orbit amplitudes (Q1 big, Q1 small, Q2 big, Q2 small)."""
    c = p['landau']['c_rel']
    R12 = 2 * np.cos(np.pi / 12)
    q1 = p['q_minus']
    W = layer_weights(c)
    A1, B1, A2, B2 = [amp_sign * a for a in amps]
    modes = ([(k, A1, 1) for k in _dirs(q1, [0, 60, 120])] + [(k, B1, 1) for k in _dirs(q1, [30, 90, 150])]
             + [(k, A2, 2) for k in _dirs(R12 * q1, [15, 75, 135])] + [(k, B2, 2) for k in _dirs(R12 * q1, [45, 105, 165])])
    return cycloid_seed(modes, L, p, lambda sc: W[sc])


def hex_seed(p, L, a1=0.65, a_sh=0.25, amp_sign=-1.0):
    """hexagonal multi-shell seed: Q1 triplet at q_-, sqrt3 shell (30deg rotated) and 2Q1 shell, layer weights of scale 2
    for the outer shells (REQUEST P3/P4)."""
    c = p['landau']['c_rel']
    W = layer_weights(c)
    q1 = p['q_minus']
    s = amp_sign
    modes = ([(k, s * a1, 1) for k in _dirs(q1, [0, 120, 240])]
             + [(k, s * a_sh, 2) for k in _dirs(np.sqrt(3) * q1, [30, 150, 270])]
             + [(k, s * a_sh, 2) for k in _dirs(2 * q1, [0, 120, 240])])
    return cycloid_seed(modes, L, p, lambda sc: W[sc])


# ---------------------------------------------------------------- observables
def structure_factors(m, mean_subtract=True):
    """S_i(q) for both layers, (2,L,L) float64 on the fftfreq grid."""
    L = m.shape[-1]
    S = np.zeros((2, L, L))
    for i in range(2):
        for a in range(3):
            f = m[3 * i + a]
            S[i] += np.abs(np.fft.fft2(f - (f.mean() if mean_subtract else 0))) ** 2
    return S


def radial_profile(S, nb=None):
    """radial average of S(q) in bins of the integer shell index |n|=|q|L/2pi (width 1)."""
    L = S.shape[-1]
    n = np.fft.fftfreq(L, 1.0 / L)
    NX, NY = np.meshgrid(n, n, indexing='ij')
    R = np.hypot(NX, NY)
    nb = nb or int(L // 2)
    idx = np.rint(R).astype(int)
    out = np.zeros((S.shape[0], nb))
    cnt = np.bincount(idx.ravel(), minlength=nb)[:nb]
    for i in range(S.shape[0]):
        out[i] = np.bincount(idx.ravel(), weights=S[i].ravel(), minlength=nb)[:nb] / np.maximum(cnt, 1)
    return out


def find_peaks(m, p, nmax=80, rel=2e-4):
    """peaks of total S(q) (half plane: one of each +-k). Returns list of dicts sorted by amplitude:
    n=(nx,ny) integer box coordinates, k, angle (deg, in [0,180)), a_i (layer amplitudes), a (mode amplitude),
    a_landau = a/sqrt(|r|/u)."""
    L = m.shape[-1]
    N = L * L
    mh = np.fft.fft2(m, axes=(1, 2))
    S = (np.abs(mh) ** 2)
    S = S[0:3].sum(0) + S[3:6].sum(0)
    S[0, 0] = 0
    # local maxima on the periodic grid
    loc = np.ones_like(S, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                loc &= S >= np.roll(np.roll(S, dx, 0), dy, 1)
    thr = rel * S.max()
    idx = np.argwhere(loc & (S > thr))
    nvec = np.fft.fftfreq(L, 1.0 / L).astype(int)
    peaks = []
    seen = set()
    base = np.sqrt(p['r_abs'] / p['u'])
    for ix, iy in idx:
        n = (int(nvec[ix]), int(nvec[iy]))
        key = n if (n[0] > 0 or (n[0] == 0 and n[1] > 0)) else (-n[0], -n[1])
        if key in seen:
            continue
        seen.add(key)
        ai = [2 * abs(mh[3 * i + 2, ix, iy]) / N for i in range(2)]
        # in-plane (cycloid) components consistent check: total amplitude from all 3 comps
        a = float(np.hypot(*ai))
        k = 2 * np.pi * np.array(key) / L
        peaks.append(dict(n=key, k=float(np.hypot(*k)), kn=float(np.hypot(*key)),
                          angle=float(np.degrees(np.arctan2(key[1], key[0])) % 180),
                          a1=float(ai[0]), a2=float(ai[1]), a=a, a_landau=a / base, S=float(S[ix, iy])))
    peaks.sort(key=lambda d: -d['a'])
    return peaks[:nmax]


def orbit_summary(peaks, p, tol=0.12):
    """group peaks into rings around q_-, q_+, R q_- ... report per ring: number of peaks, big/small orbit mean a_landau."""
    q1 = p['q_minus']
    R12 = 2 * np.cos(np.pi / 12)
    rings = {'Q1(q-)': q1, 'Q2(R12 q-)': R12 * q1, 'sqrt3 Q1': np.sqrt(3) * q1, '2 Q1': 2 * q1, 'q+': p['q_plus']}
    out = {}
    for name, kr in rings.items():
        sel = [d for d in peaks if abs(d['k'] - kr) < tol * kr]
        out[name] = dict(k_ring=float(kr), n_peaks=len(sel),
                         a_landau=[round(d['a_landau'], 4) for d in sorted(sel, key=lambda d: d['angle'])],
                         angles=[round(d['angle'], 1) for d in sorted(sel, key=lambda d: d['angle'])])
    return out


def phase_sum_triads(m, p, peaks, tol=1e-9, nmax=6):
    """Phi = phi1+phi2+phi3 of the m_z (layer-summed) Fourier phases for resonant triads k1+k2+k3=0 among the peaks
    (using +-k representatives). Returns list of (n1,n2,n3,Phi mod 2pi, product amplitude)."""
    L = m.shape[-1]
    mz = m[2] + m[5]
    h = np.fft.fft2(mz)
    keys = [d['n'] for d in peaks[:30]]
    out = []
    import itertools
    for a, b in itertools.combinations(keys, 2):
        for sa in (1, -1):
            for sb in (1, -1):
                n3 = (-(sa * a[0] + sb * b[0]), -(sa * a[1] + sb * b[1]))
                key3 = n3 if (n3[0] > 0 or (n3[0] == 0 and n3[1] > 0)) else (-n3[0], -n3[1])
                if key3 in keys and key3 != a and key3 != b:
                    c3 = n3
                    ph = np.angle(h[sa * a[0] % L, sa * a[1] % L]) + np.angle(h[sb * b[0] % L, sb * b[1] % L]) + np.angle(h[c3[0] % L, c3[1] % L])
                    out.append(((sa * a[0], sa * a[1]), (sb * b[0], sb * b[1]), c3, float(np.mod(ph, 2 * np.pi))))
    return out[:nmax * 6]


# ---------------------------------------------------------------- integer (box-commensurate) approximant seeds
def _band_cost_fn(p, L):
    """mismatch cost (lambda_min(k)+|r|)/|r| of a mode with integer box vector of squared length n2, from the real-space kernel.
    Cached on n2 (integer) so approximant searches are cheap."""
    from params import kernel
    from functools import lru_cache
    ks = np.linspace(0.005, 1.4, 700)
    lam = np.array([np.linalg.eigvalsh(kernel(p, k, 0))[0] for k in ks])
    sc = 2 * np.pi / L

    @lru_cache(maxsize=None)
    def f(n2):
        return float((np.interp(np.sqrt(n2) * sc, ks, lam) + p['r_abs']) / p['r_abs'])
    return f


def _int_vecs(n0, w):
    r = int(np.ceil(n0 + w))
    return [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if n0 - w <= np.hypot(a, b) <= n0 + w]


def _rot(v, deg):
    t = np.radians(deg); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return R @ np.asarray(v, float)


def _n2(v):
    return int(v[0] * v[0] + v[1] * v[1])


def _equilateral_pairs(V):
    out = []
    for v1 in V:
        for v2 in V:
            d = v1[0] * v2[0] + v1[1] * v2[1]
            if -0.6 < d / np.sqrt(_n2(v1) * _n2(v2)) < -0.4 and v1 < v2:
                out.append((v1, v2))
    return out


def hex_int_modes(p, L, amps=(0.65, 0.25, 0.25), width=3.0):
    """Best box-commensurate approximant of the multi-shell hexagonal state: integer triple v1+v2+v3=0 (Q1 shell),
    sqrt3 shell v_i - v_j, 2Q1 shell 2 v_i. Cost = sum amp^2 * (band mismatch) using the actual kernel."""
    f = _band_cost_fn(p, L)
    n0 = p['q_minus'] * L / (2 * np.pi)
    V = _int_vecs(n0, width)
    best = None
    for v1, v2 in _equilateral_pairs(V):
        v3 = (-(v1[0] + v2[0]), -(v1[1] + v2[1]))
        vs = [v1, v2, v3]
        sh3 = [(vs[i][0] - vs[(i + 1) % 3][0], vs[i][1] - vs[(i + 1) % 3][1]) for i in range(3)]
        cost = (amps[0] ** 2 * sum(f(_n2(v)) for v in vs) + amps[1] ** 2 * sum(f(_n2(v)) for v in sh3)
                + amps[2] ** 2 * sum(f(4 * _n2(v)) for v in vs))
        if best is None or cost < best[0]:
            best = (cost, vs, sh3)
    cost, vs, sh3 = best
    sh2 = [(2 * v[0], 2 * v[1]) for v in vs]
    modes = [(v, amps[0], 1) for v in vs] + [(v, amps[1], 2) for v in sh3] + [(v, amps[2], 2) for v in sh2]
    return modes, dict(cost=float(cost), Q1=vs, sqrt3=sh3, Q2x=sh2, n_minus=float(n0))


def qc_int_modes(p, L, amps=(0.607, 0.148, 0.619, 0.132), width=3.0):
    """Box-commensurate 6-fold quasicrystal approximant: Q1-big triple v_i (sum 0), Q1-small w_i near v_i rotated by +30deg,
    Q2 = v_i + w_i (15deg-ish, big orbit) and w_i + v_{i+1} (45deg-ish, small orbit); all vectors integer, triads close exactly."""
    from functools import lru_cache
    f = _band_cost_fn(p, L)
    n0 = p['q_minus'] * L / (2 * np.pi)
    V = _int_vecs(n0, width)

    @lru_cache(maxsize=None)
    def bestw(vi, vn):
        tgt = np.rint(_rot(vi, 30)).astype(int)
        best = None
        for a in range(-2, 3):
            for b in range(-2, 3):
                w = (int(tgt[0]) + a, int(tgt[1]) + b)
                c = (amps[1] ** 2 * f(_n2(w)) + amps[2] ** 2 * f(_n2((vi[0] + w[0], vi[1] + w[1])))
                     + amps[3] ** 2 * f(_n2((w[0] + vn[0], w[1] + vn[1]))))
                if best is None or c < best[0]:
                    best = (c, w)
        return best

    best = None
    for v1, v2 in _equilateral_pairs(V):
        v3 = (-(v1[0] + v2[0]), -(v1[1] + v2[1]))
        vs = [v1, v2, v3]
        cost = amps[0] ** 2 * sum(f(_n2(v)) for v in vs)
        ws = []
        for i in range(3):
            c, w = bestw(vs[i], vs[(i + 1) % 3]); cost += c; ws.append(w)
        if best is None or cost < best[0]:
            best = (cost, vs, ws)
    cost, vs, ws = best
    modes = ([(v, amps[0], 1) for v in vs] + [(w, amps[1], 1) for w in ws]
             + [((vs[i][0] + ws[i][0], vs[i][1] + ws[i][1]), amps[2], 2) for i in range(3)]
             + [((ws[i][0] + vs[(i + 1) % 3][0], ws[i][1] + vs[(i + 1) % 3][1]), amps[3], 2) for i in range(3)])
    return modes, dict(cost=float(cost), v=vs, w=ws, n_minus=float(n0))


def int_seed(modes, L, p, c_rel, amp_sign=-1.0):
    """field from integer-vector modes [(n, amp_landau, scale)] (phases 0, amplitudes negative)."""
    W = layer_weights(c_rel)
    kmodes = [(2 * np.pi * np.asarray(n, float) / L, amp_sign * a, sc) for n, a, sc in modes]
    return cycloid_seed(kmodes, L, p, lambda sc: W[sc])


def eig_spiral_seed(p, L, specs, band=0):
    """Seeds built from the exact lowest eigenvector of the 6x6 kernel at each box wavevector.
    specs = [((nx, ny), amp_landau)]; m = base*amp*sqrt(2)*Re[v exp(i k r)] (|v|=1)."""
    from params import kernel
    x = np.arange(L)
    X, Y = np.meshgrid(x, x, indexing='ij')
    m = np.zeros((6, L, L))
    base = np.sqrt(p['r_abs'] / p['u'])
    for n, a in specs:
        k = 2 * np.pi * np.asarray(n, float) / L
        w, V = np.linalg.eigh(kernel(p, k[0], k[1]))
        v = V[:, band]
        ph = np.exp(1j * (k[0] * X + k[1] * Y))
        for j in range(6):
            m[j] += base * a * np.sqrt(2) * np.real(v[j] * ph)
    return m
