"""
Coupling-matrix extraction for an N-pole inline (canonical) bandpass filter.
Runs in Pyto on iPad (needs numpy, scipy, matplotlib - all bundled with Pyto).

Design constants (N, f0, BW, ripple, Qu) are asked for at start-up; pressing Enter
accepts the defaults: N=5, f0=500 MHz, BW=50 MHz, 0.02 dB ripple, Qu=500 (50 ohm).

Flow:
  1. Read a saved .s2p (strict Touchstone 1.x, any unit, RI/MA/DB, R kept)
  2. Fit an N-pole inline coupling matrix (+ port loads + loss): global magnitude
     search, then complex S11/S21/S22 refinement when phase is available, with
     identifiability diagnostics (uncertainties, correlations, ambiguity check)
  3. Save the extracted response as a new .s2p (+ matrix as .txt)
  4. Plot original vs extracted S11 and S21
  5. Live editable-matrix editor (the page is embedded in this script): edit M / R1 / RN / Qu and watch S11, S21
     update in real time against the measured data (+ target comparison).
     "Apply next move" / "Auto-tune" apply the Next best move one parameter
     at a time until every "x tol" value reads 0.000.
  5b. Optional cross-couplings (transmission zeros): enter e.g. 2-4 at the design
     prompt; the model then uses a dense solve and the fit/advisor/editor include them
  5c. 'Synthesise target': designs a matched target (return loss + transmission-zero
     value or frequency) and writes it as a target file; choose it at the file prompt [s]
     or type 'synth' at the target prompt
  6. (optional) Level 1 Tuning Advisor: compare against a target design and
     print prioritised, model-based electrical tuning advice (+ .txt/.csv)

Model (lowpass-normalised, Cameron form):
  A = M + Omega*I - j(R + delta*I),  Omega = (f/f0 - f0/f)/FBW
  S11 = 1 + 2j*R1*inv(A)[0,0]
  S21 = -2j*sqrt(R1*RN)*inv(A)[N-1,0]
  delta = 1/(Qu*FBW)

Note: out-of-tolerance parameters never produce a HOLD or a warning while the
responses are being adjusted; they are simply ranked by how far out they are.

For no Transmission Zero type normal or tap enter blank.

For Transmission Zeros run through program once using s ( sythezize ).

Order and pair	 Cross value	Zero at
N = 3, pair 1-3	+0.447	580 MHz
N = 4, pair 2-4	+0.250	580 MHz
N = 5, pair 2-4	+0.149	580 MHz
N = 6, pair 3-5	+0.128	580 MHz
N = 7, pair 3-5	+0.112	580 MHz
N = 9, pair 4-6	+0.100	580 MHz

"""
import csv
import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize
import matplotlib.pyplot as plt

# ---------------- design specification ----------------
# Defaults, used whenever a prompt is answered with a blank (just Enter).
DEFAULT_DESIGN = dict(n=5, f0=500e6, bw=50e6, ripple_db=0.02, qu=500.0)
CROSS_PAIRS = []             # cross-couplings = transmission zeros; 0-based pairs, [(1, 3)] = M24
PROMPT_FOR_DESIGN = True     # False = skip the questions and use the defaults

# Live design constants (changed by set_design / choose_design at start-up).
N = DEFAULT_DESIGN["n"]
F0 = DEFAULT_DESIGN["f0"]
BW = DEFAULT_DESIGN["bw"]
RIPPLE_DB = DEFAULT_DESIGN["ripple_db"]
QU = DEFAULT_DESIGN["qu"]
Z0 = 50.0                    # default reference impedance (a file's own R is kept)
FBW = BW / F0

# ANSI escape codes (Pyto's console renders these)
BOLD = "\033[1m"
RESET = "\033[0m"


def set_design(n=None, f0=None, bw=None, ripple_db=None, qu=None, cross=None):
    """Set the design constants for the whole program (None = default value).

    Everything derived from them (FBW, parameter-vector layout) is recomputed.
    Returns a list of warnings; raises ValueError if a value is invalid, in
    which case the previous design is left untouched.
    """
    d = DEFAULT_DESIGN
    n = d["n"] if n is None else n
    f0 = d["f0"] if f0 is None else f0
    bw = d["bw"] if bw is None else bw
    ripple_db = d["ripple_db"] if ripple_db is None else ripple_db
    qu = d["qu"] if qu is None else qu
    cross = [] if cross is None else cross
    warns: List[str] = []
    if int(n) != n or not 2 <= n <= 15:
        raise ValueError("order N must be a whole number from 2 to 15")
    for name, v in (("f0", f0), ("BW", bw), ("ripple", ripple_db), ("Qu", qu)):
        if not np.isfinite(v) or v <= 0:
            raise ValueError("%s must be a positive number" % name)
    if bw >= f0:
        raise ValueError("BW must be smaller than f0 (BW=%g Hz, f0=%g Hz)" % (bw, f0))
    if ripple_db > 3.0:
        raise ValueError("ripple must be 3 dB or less")
    pairs = []
    for a, b in cross:
        a, b = int(min(a, b)), int(max(a, b))
        if a < 0 or b >= n or b - a < 2:
            raise ValueError("cross-coupling %d-%d is invalid (resonators 1..N, not adjacent)"
                             % (a + 1, b + 1))
        if (a, b) in pairs:
            raise ValueError("cross-coupling %d-%d is listed twice" % (a + 1, b + 1))
        pairs.append((a, b))
    fbw = bw / f0
    if fbw > 0.3:
        warns.append("Fractional bandwidth %.0f%% is wide: the narrowband "
                     "lowpass mapping used here becomes less accurate." % (100 * fbw))
    if qu * fbw < 3:
        warns.append("Qu*FBW = %.1f is very low (heavy loss); the fit may be poorly determined." % (qu * fbw))

    global N, F0, BW, RIPPLE_DB, QU, FBW, PARAM_LEN
    global IDX_DIAG, IDX_COUP, IDX_R1, IDX_RN, IDX_DELTA, CROSS_PAIRS, IDX_CROSS
    N, F0, BW, RIPPLE_DB, QU = int(n), float(f0), float(bw), float(ripple_db), float(qu)
    FBW = BW / F0
    CROSS_PAIRS = pairs
    PARAM_LEN = 2 * N + 2 + len(pairs)
    IDX_DIAG = slice(0, N)
    IDX_COUP = slice(N, 2 * N - 1)
    IDX_R1, IDX_RN, IDX_DELTA = 2 * N - 1, 2 * N, 2 * N + 1
    IDX_CROSS = slice(2 * N + 2, PARAM_LEN)
    return warns


_FREQ_UNITS = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}


def parse_freq(text: str) -> float:
    """'500e6', '500 MHz', '0.5ghz' -> Hz. A plain number below 10000 is MHz."""
    t = text.strip().lower().replace(" ", "").replace(",", "")
    m = re.fullmatch(r"([-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?)(hz|khz|mhz|ghz)?", t)
    if not m:
        raise ValueError("cannot read '%s' as a frequency" % text)
    v = float(m.group(1))
    if m.group(2):
        return v * _FREQ_UNITS[m.group(2)]
    return v * 1e6 if abs(v) < 1e4 else v


def parse_number(text: str) -> float:
    try:
        return float(text.strip().replace(",", ""))
    except ValueError:
        raise ValueError("cannot read '%s' as a number" % text)


def parse_order(text: str) -> int:
    v = parse_number(text)
    if v != int(v):
        raise ValueError("N must be a whole number")
    return int(v)


def parse_cross(text: str):
    """'2-4' or '1-3, 3-5' -> [(1, 3), (0, 2), ...] (0-based); 'none' -> []."""
    t = text.strip().lower()
    if t in ("none", "no", "n", "0", "-"):
        return []
    out = []
    for tok in re.split(r"[,;\s]+", t):
        if not tok:
            continue
        m = re.fullmatch(r"(\d+)[-:](\d+)", tok)
        if not m:
            raise ValueError("cannot read '%s' as a resonator pair (use e.g. 2-4)" % tok)
        out.append((int(m.group(1)) - 1, int(m.group(2)) - 1))
    return out


def cross_text() -> str:
    return ",".join("%d-%d" % (a + 1, b + 1) for a, b in CROSS_PAIRS) or "none"


def _ask(label, default_text, parser):
    """Prompt until valid; a blank answer returns None (= use the default)."""
    while True:
        raw = input("  %s [%s]: " % (label, default_text)).strip()
        if not raw:
            return None
        try:
            return parser(raw)
        except ValueError as exc:
            print("    ! %s - try again, or press Enter for the default" % exc)


def fmt_freq(hz: float) -> str:
    for unit, div in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if hz >= div:
            return "%.6g %s" % (hz / div, unit)
    return "%.6g Hz" % hz


def design_summary() -> str:
    return ("N=%d  f0=%s  BW=%s (FBW=%.3f)  ripple=%.4g dB  Qu=%.6g  cross=%s"
            % (N, fmt_freq(F0), fmt_freq(BW), FBW, RIPPLE_DB, QU, cross_text()))


def choose_design():
    """Ask for the design constants; a blank answer keeps that default."""
    d = DEFAULT_DESIGN
    while True:
        print("\nFilter design constants - press Enter to accept each default.")
        print("(frequencies: e.g. 500e6, 500 MHz or 0.5 GHz; a plain number "
              "below 10000 is taken as MHz)")
        print(BOLD + "WARNING: These values are for the Golden Filter "
              "Coupling Matrix for tuning reference" + RESET)
        vals = dict(
            n=_ask("Order N", "%d" % d["n"], parse_order),
            f0=_ask("Center frequency F0", fmt_freq(d["f0"]), parse_freq),
            bw=_ask("Bandwidth BW", fmt_freq(d["bw"]), parse_freq),
            ripple_db=_ask("Passband ripple (dB)", "%g" % d["ripple_db"], parse_number),
            qu=_ask("Unloaded Q (Qu)", "%g" % d["qu"], parse_number),
            cross=_ask("Cross-couplings = transmission zeros (e.g. 2-4 or 1-3,3-5)",
                       "none", parse_cross),
        )
        try:
            warns = set_design(**vals)
        except ValueError as exc:
            print("  ! Invalid design: %s. Please re-enter." % exc)
            continue
        print("Design: " + design_summary())
        for w in warns:
            print("  WARNING: " + w)
        return


# ---------------- prototype values ----------------
def cheb_g(n, ripple_db):
    beta = np.log(1.0 / np.tanh(ripple_db / 17.37))
    gamma = np.sinh(beta / (2 * n))
    a = [np.sin((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)]
    b = [gamma ** 2 + np.sin(k * np.pi / n) ** 2 for k in range(1, n + 1)]
    g = [2 * a[0] / gamma]
    for k in range(2, n + 1):
        g.append(4 * a[k - 2] * a[k - 1] / (b[k - 2] * g[k - 2]))
    load = 1.0 if n % 2 else 1.0 / np.tanh(beta / 4) ** 2
    return g, load


def nominal_params():
    g, load = cheb_g(N, RIPPLE_DB)
    diag = np.zeros(N)
    coup = np.array([1 / np.sqrt(g[i] * g[i + 1]) for i in range(N - 1)])
    r1 = 1.0 / g[0]
    rn = 1.0 / (g[-1] * load)
    delta = 1.0 / (QU * FBW)
    return np.concatenate([diag, coup, [r1, rn, delta], np.zeros(len(CROSS_PAIRS))])


# parameter vector: [M11..M55 (5), M12 M23 M34 M45 (4), R1, RN, delta]
def unpack(p):
    d = p[:N]
    c = p[N:2 * N - 1]
    r1, rn, delta = p[2 * N - 1:2 * N + 2]
    M = np.diag(d)
    for i in range(N - 1):
        M[i, i + 1] = M[i + 1, i] = c[i]
    for m, (a, b) in enumerate(CROSS_PAIRS):
        M[a, b] = M[b, a] = p[2 * N + 2 + m]
    return M, r1, rn, delta


def _chain_solve(om, d, c, r_first, r_last, delta):
    """Solve A x = e_first for the tridiagonal chain matrix, all frequencies at once.
    A = diag(om + d - j(delta + R)) + off-diagonal c. Returns (x_first, x_last)."""
    n = len(d)
    b = [om + d[i] - 1j * (delta + (r_first if i == 0 else 0.0) + (r_last if i == n - 1 else 0.0))
         for i in range(n)]
    bp, rp = [b[0]], [np.ones_like(om, dtype=complex)]
    for i in range(1, n):                               # Thomas algorithm
        w = c[i - 1] / bp[i - 1]
        bp.append(b[i] - w * c[i - 1])
        rp.append(-w * rp[i - 1])
    x = [None] * n
    x[n - 1] = rp[n - 1] / bp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (rp[i] - c[i] * x[i + 1]) / bp[i]
    return x[0], x[n - 1]


def model(p, f):
    """Returns S as array (nf, 2, 2).  Fast tridiagonal solver (dense if cross-couplings)."""
    f = np.asarray(f, float)
    if CROSS_PAIRS:
        return _model_dense(p, f)
    d, c = p[:N], p[N:2 * N - 1]
    r1, rn, delta = p[2 * N - 1], p[2 * N], p[2 * N + 1]
    om = (f / F0 - F0 / f) / FBW
    x_first, x_last = _chain_solve(om, d, c, r1, rn, delta)
    y_first, _ = _chain_solve(om, d[::-1], c[::-1], rn, r1, delta)   # = inv(A)[N-1,N-1]
    S = np.zeros((len(f), 2, 2), dtype=complex)
    S[:, 0, 0] = 1 + 2j * r1 * x_first
    S[:, 1, 1] = 1 + 2j * rn * y_first
    S[:, 1, 0] = S[:, 0, 1] = -2j * np.sqrt(r1 * rn) * x_last
    return S


def _model_dense(p, f):
    """Dense-inverse model: needed when cross-couplings make the matrix non-tridiagonal."""
    f = np.asarray(f, float)
    M, r1, rn, delta = unpack(p)
    om = (f / F0 - F0 / f) / FBW
    R = np.zeros((N, N))
    R[0, 0], R[-1, -1] = r1, rn
    A = M[None] + om[:, None, None] * np.eye(N) - 1j * (R + delta * np.eye(N))[None]
    Ai = np.linalg.inv(A)
    S = np.zeros((len(f), 2, 2), dtype=complex)
    S[:, 0, 0] = 1 + 2j * r1 * Ai[:, 0, 0]
    S[:, 1, 1] = 1 + 2j * rn * Ai[:, -1, -1]
    S[:, 1, 0] = S[:, 0, 1] = -2j * np.sqrt(r1 * rn) * Ai[:, -1, 0]
    return S


def _selftest():
    """Optional check (call by hand): the fast tridiagonal model must equal the dense one."""
    global CROSS_PAIRS
    saved, CROSS_PAIRS = CROSS_PAIRS, []
    try:
        p = nominal_params()
        f = np.linspace(F0 - 2 * BW, F0 + 2 * BW, 50)
        assert np.allclose(model(p, f), _model_dense(p, f))
    finally:
        CROSS_PAIRS = saved
    print("self-test OK")


# ---------------- Touchstone I/O ----------------
class S2PError(ValueError):
    """Raised when a Touchstone file is unsupported or malformed."""


_S2P_UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def read_s2p(path):
    """Strict Touchstone 1.x 2-port reader. Returns (f_Hz, S, z0).

    Accepts: one option line (# [unit] S [RI|MA|DB] [R z0]) and 9-number data
    lines (freq, S11, S21, S12, S22). Rejects, with a clear message: Touchstone
    2.0 keywords, non-S parameters, a second option line, noise-parameter
    blocks, short/long records, non-numeric or non-finite values, and
    frequencies that are not positive and strictly increasing.
    Defaults (Touchstone 1.x): GHz, MA, R 50.
    """
    unit, fmt, z0 = 1e9, "MA", 50.0
    have_opt = False
    rows = []
    try:
        fh = open(path, "r")
    except OSError as exc:
        raise S2PError("Cannot open file: %s" % exc)
    with fh:
        for ln, raw in enumerate(fh, 1):
            line = raw.split("!")[0].strip()
            if not line:
                continue
            if line.startswith("["):
                raise S2PError("Line %d: Touchstone 2.0 keyword '%s' - only "
                               "Touchstone 1.x files are supported."
                               % (ln, line.split("]")[0] + "]"))
            if line.startswith("#"):
                if have_opt:
                    raise S2PError("Line %d: more than one option line ('#')." % ln)
                if rows:
                    raise S2PError("Line %d: option line after data." % ln)
                have_opt = True
                tok = line[1:].upper().split()
                i = 0
                while i < len(tok):
                    t = tok[i]
                    if t in _S2P_UNITS:
                        unit = _S2P_UNITS[t]
                    elif t == "S":
                        pass
                    elif t in ("Y", "Z", "G", "H"):
                        raise S2PError("Line %d: %s-parameters are not supported "
                                       "(S-parameters only)." % (ln, t))
                    elif t in ("RI", "MA", "DB"):
                        fmt = t
                    elif t == "R":
                        try:
                            z0 = float(tok[i + 1])
                        except (IndexError, ValueError):
                            raise S2PError("Line %d: 'R' must be followed by a "
                                           "number." % ln)
                        if not np.isfinite(z0) or z0 <= 0:
                            raise S2PError("Line %d: reference impedance must be "
                                           "positive." % ln)
                        i += 1
                    else:
                        raise S2PError("Line %d: unknown option '%s'." % (ln, t))
                    i += 1
                continue
            try:
                vals = [float(x) for x in line.split()]
            except ValueError:
                raise S2PError("Line %d: non-numeric data." % ln)
            if len(vals) != 9:
                hint = (" (5 numbers: looks like a noise-parameter block, which "
                        "is not supported)" if len(vals) == 5 else "")
                raise S2PError("Line %d: expected 9 numbers (freq + S11 S21 S12 "
                               "S22), found %d%s." % (ln, len(vals), hint))
            if not np.all(np.isfinite(vals)):
                raise S2PError("Line %d: NaN/inf value." % ln)
            rows.append(vals)
    if not rows:
        raise S2PError("No data found.")
    d = np.array(rows)
    f = d[:, 0] * unit
    if f[0] <= 0:
        raise S2PError("Frequency must be positive (first point is %g Hz)." % f[0])
    if np.any(np.diff(f) <= 0):
        k = int(np.where(np.diff(f) <= 0)[0][0]) + 1
        raise S2PError("Frequencies must be strictly increasing (data point %d)." % (k + 1))

    def cx(a, b):
        if fmt == "RI":
            return a + 1j * b
        mag = 10 ** (a / 20) if fmt == "DB" else a
        return mag * np.exp(1j * np.deg2rad(b))

    S = np.zeros((len(f), 2, 2), dtype=complex)
    S[:, 0, 0] = cx(d[:, 1], d[:, 2])
    S[:, 1, 0] = cx(d[:, 3], d[:, 4])
    S[:, 0, 1] = cx(d[:, 5], d[:, 6])
    S[:, 1, 1] = cx(d[:, 7], d[:, 8])
    return f, S, z0


def write_s2p(path, f, S, comment="", z0=Z0):
    with open(path, "w") as fh:
        if comment:
            fh.write("! " + comment + "\n")
        fh.write("# MHZ S RI R %g\n" % z0)
        for i in range(len(f)):
            row = [f[i] / 1e6]
            for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
                row += [S[i, a, b].real, S[i, a, b].imag]
            fh.write(" ".join("%.9g" % v for v in row) + "\n")


# ---------------- extraction ----------------
# Fit strategy (extract_robust):
#   stage 1  magnitude multi-start (global search). Couplings are constrained
#            positive: for an inline chain the sign of every coupling is a
#            "gauge" choice (a similarity transform D*M*D with D = diag(+-1)
#            leaves |S11|, |S22| unchanged and at most flips the sign of S21),
#            so allowing signed couplings only adds equivalent duplicates.
#            |S22| is used too when available (it removes the port-reversal
#            ambiguity of the S11/S21-only fit).
#   stage 2  complex S11/S21/S22 refinement of every distinct stage-1 solution
#            (real + imaginary residuals) with nuisance terms for the
#            reference-plane delays and a constant S21 phase. The candidate
#            with the best complex cost wins; runners-up are reported.
#   stage 3  Jacobian identifiability analysis at the solution: 1-sigma
#            uncertainties, correlations, conditioning, weakest combination.
FIT_MODE = "auto"           # "auto" | "complex" | "magnitude"
NOISE_FLOOR = 2e-4          # minimum trace noise (linear |S|) assumed for uncertainties
TAU_MAX_FACTOR = 0.25       # reference-plane delay search range = +-0.25/BW
N_STARTS_DEFAULT = 24
CHI2_OK = 3.0               # residual/noise chi2 considered 'at the noise level'
MAX_ESCALATIONS = 2         # extra search rounds when the fit stays above the noise
AMBIG_DCHI2 = 30.0          # runner-up closer than this (delta chi^2) cannot be rejected


def db(x, floor=1e-3):
    return 20 * np.log10(np.maximum(np.abs(x), floor))   # default -60 dB floor


def estimate_noise(S) -> float:
    """Trace-noise sigma (per real/imag component) from the data's own roughness:
    robust (MAD) estimate on the second difference across frequency."""
    vals = []
    for a, b in ((0, 0), (1, 0), (1, 1)):
        x = S[:, a, b]
        if len(x) < 5 or np.max(np.abs(x)) < 1e-6:
            continue
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
        for comp in (d2.real, d2.imag):
            if np.max(np.abs(comp)) > 0:                 # skip all-zero parts (magnitude-only files)
                vals.append(np.abs(comp))
    if not vals:
        return NOISE_FLOOR
    mad = np.median(np.concatenate(vals))
    return float(max(1.4826 * mad / np.sqrt(6.0), NOISE_FLOOR))


def has_phase(f, S) -> bool:
    """True if the data carries usable phase (not a magnitude-only file)."""
    f = np.asarray(f, float)
    band = np.abs(f - F0) <= BW
    if band.sum() < 5:
        band = np.ones(len(f), bool)
    s = S[band, 1, 0]
    if np.max(np.abs(s)) <= 1e-9:
        s = S[band, 0, 0]
    if np.max(np.abs(s.imag)) <= 1e-9 * max(np.max(np.abs(s)), 1e-12):
        return False                                   # all angles ~0
    return bool(np.ptp(np.unwrap(np.angle(s))) > 0.5)


def has_s22(S) -> bool:
    return bool(np.max(np.abs(S[:, 1, 1])) > 1e-3)


def detect_notches(f, S, max_n=2, min_prominence_db=15.0, floor_db=-75.0):
    """Out-of-band S21 notches (transmission zeros) in measured data: [(f_Hz, prominence_dB)].
    A notch is a local minimum outside f0 +/- 0.6 BW that is at least min_prominence_db below
    the response 0.25 BW to either side, where that response is above floor_db (so noise-floor
    ripple is not mistaken for a zero)."""
    f = np.asarray(f, float)
    y = db(S[:, 1, 0], 1e-6)
    x = S[:, 1, 0]                                   # trace noise from the data's own roughness
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    sig = 1.4826 * np.median(np.concatenate([np.abs(d2.real), np.abs(d2.imag)])) / np.sqrt(6.0)
    floor_db = max(floor_db, 20 * np.log10(max(5 * sig, 1e-12)))   # ignore dips near the noise
    out = []
    for i in range(1, len(f) - 1):
        if not (y[i] < y[i - 1] and y[i] <= y[i + 1]) or abs(f[i] - F0) <= 0.6 * BW:
            continue
        lo, hi = f[i] - 0.25 * BW, f[i] + 0.25 * BW
        if lo < f[0] or hi > f[-1]:
            continue
        left, right = float(np.interp(lo, f, y)), float(np.interp(hi, f, y))
        prom = min(left, right) - y[i]
        if prom >= min_prominence_db and max(left, right) > floor_db:
            out.append((float(f[i]), float(prom)))
    out.sort(key=lambda t: -t[1])
    keep = []
    for fi, pr in out:                     # one entry per notch (wide dips give several minima)
        if all(abs(fi - fk) > 0.3 * BW for fk, _ in keep):
            keep.append((fi, pr))
    return sorted(keep[:max_n])


def suggest_pairs(n, n_zeros):
    """Cross-coupling pairs (0-based) that can realise one or two zeros at order n."""
    c = n // 2
    if n_zeros >= 2 and n >= 5 and c + 2 <= n - 1:
        return [(c - 2, c), (c, c + 2)]
    return [(max(c - 1, 0), min(c + 1, n - 1))] if n >= 3 else []


def apply_reference_planes(S, f, nuis, refl=1.0):
    """Add port reference-plane delays tau1, tau2 (s) and S21 phase phi (rad).
    refl = -1 flips the sign of S11/S22 (some tools/conventions report an
    out-of-band reflection of -1 where this model gives +1)."""
    t1, t2, phi = nuis
    out = S.copy()
    out[:, 0, 0] = refl * S[:, 0, 0] * np.exp(-4j * np.pi * f * t1)
    out[:, 1, 1] = refl * S[:, 1, 1] * np.exp(-4j * np.pi * f * t2)
    out[:, 1, 0] = S[:, 1, 0] * np.exp(-2j * np.pi * f * (t1 + t2) + 1j * phi)
    out[:, 0, 1] = out[:, 1, 0]
    return out


class _Problem:
    """Residual functions for one data set and one fitting mode."""

    def __init__(self, f, S, mode, use22, db_floor=1e-3, refl=1.0):
        self.f = np.asarray(f, float)
        self.refl = refl
        self.cplx = (mode == "complex")
        self.use22 = use22
        self.fl = db_floor                  # dB-term floor: keep it above the noise
        self.meas = [S[:, 0, 0], S[:, 1, 0]] + ([S[:, 1, 1]] if use22 else [])

    def bounds(self):
        lo, hi = param_bounds()
        if self.cplx:
            t = TAU_MAX_FACTOR
            lo = np.concatenate([lo, [-t, -t, -4.0]])
            hi = np.concatenate([hi, [t, t, 4.0]])
        return lo, hi

    def xscale(self):
        s = 0.1 * np.ones(PARAM_LEN)
        return np.concatenate([s, [0.05, 0.05, 0.3]]) if self.cplx else s

    def nuis(self, x):
        return np.array([x[PARAM_LEN] / BW, x[PARAM_LEN + 1] / BW, x[PARAM_LEN + 2]])

    def terms(self, x):
        S = model(x[:PARAM_LEN], self.f)
        if self.cplx:
            S = apply_reference_planes(S, self.f, self.nuis(x), self.refl)
        t = [S[:, 0, 0], S[:, 1, 0]]
        if self.use22:
            t.append(S[:, 1, 1])
        return t

    def _res(self, x, with_db):
        out = []
        for mod, meas in zip(self.terms(x), self.meas):
            if self.cplx:
                d = mod - meas
                out += [d.real, d.imag]
            else:
                out.append(np.abs(mod) - np.abs(meas))
            if with_db:
                out.append((db(mod, self.fl) - db(meas, self.fl)) / 20)
        return np.concatenate(out)

    def full(self, x):          # fitting residuals (adds a dB term: deep nulls matter)
        return self._res(x, True)

    def lin(self, x):           # purely linear residuals (used for uncertainties)
        return self._res(x, False)


def param_tolerance_abs(p, tol=None):
    """Absolute tolerance of every parameter (used for distances / uncertainty)."""
    tol = tol or Tolerances()
    p = np.asarray(p, float)
    t = np.zeros(PARAM_LEN)
    t[IDX_DIAG] = tol.diagonal_abs
    c = np.abs(p[IDX_COUP])
    t[IDX_COUP] = np.where(c >= tol.zero_target_threshold,
                           tol.coupling_rel * c, tol.coupling_abs_floor)
    t[IDX_R1] = tol.load_rel * abs(p[IDX_R1])
    t[IDX_RN] = tol.load_rel * abs(p[IDX_RN])
    t[IDX_DELTA] = tol.loss_rel * max(abs(p[IDX_DELTA]), 1e-12)
    t[IDX_CROSS] = tol.cross_abs
    return t


def param_distance(a, b, tol=None):
    """Largest parameter difference between two solutions, in tolerance units.
    Couplings are compared by magnitude (sign is a gauge choice)."""
    a, b = np.asarray(a, float)[:PARAM_LEN], np.asarray(b, float)[:PARAM_LEN]
    tl = param_tolerance_abs(b, tol)
    d = np.abs(a - b)
    d[IDX_COUP] = np.abs(np.abs(a[IDX_COUP]) - np.abs(b[IDX_COUP]))
    d[IDX_DELTA] = 0.0                       # loss is judged separately
    return float(np.max(d / np.maximum(tl, 1e-12)))


def _make_starts(n_starts, seed, p0, broad=False):
    """Nominal start + small, wide and structured perturbations.
    broad=True (escalation rounds): uniform samples over the plausible region."""
    rng = np.random.default_rng(seed)
    if broad:
        out = []
        for _ in range(n_starts):
            s = p0.copy()
            s[:N] = rng.uniform(-0.25, 0.25, N)
            s[IDX_COUP] = p0[IDX_COUP] * rng.uniform(0.6, 1.5, N - 1)
            s[IDX_R1:IDX_RN + 1] = p0[IDX_R1:IDX_RN + 1] * rng.uniform(0.5, 2.0, 2)
            s[IDX_DELTA] = p0[IDX_DELTA] * rng.uniform(0.5, 2.0)
            s[IDX_CROSS] = rng.uniform(-0.3, 0.3, len(CROSS_PAIRS))
            out.append(s)
        return out
    out = [p0.copy()]
    k = 0
    while len(out) < n_starts:
        k += 1
        s = p0.copy()
        tier = k % 3
        if tier == 1:                                   # small jitter
            s[:N] += rng.normal(0, 0.05, N)
            s[IDX_COUP] *= 1 + rng.normal(0, 0.08, N - 1)
            s[IDX_R1:IDX_DELTA + 1] *= 1 + rng.normal(0, 0.10, 3)
            s[IDX_CROSS] += rng.normal(0, 0.05, len(CROSS_PAIRS))
        elif tier == 2:                                 # wide jitter
            s[:N] += rng.normal(0, 0.15, N)
            s[IDX_COUP] *= 1 + rng.normal(0, 0.20, N - 1)
            s[IDX_R1:IDX_DELTA + 1] *= 1 + rng.normal(0, 0.30, 3)
            s[IDX_CROSS] += rng.normal(0, 0.15, len(CROSS_PAIRS))
        else:                                           # structured
            s[:N] += (0.1 if (k // 3) % 2 else -0.1) * rng.choice([-1, 1], N)
            s[IDX_DELTA] *= (0.5, 2.0)[(k // 6) % 2]
            s[IDX_R1:IDX_RN + 1] *= (0.8, 1.25)[(k // 12) % 2]
            s[IDX_CROSS] = 0.15 * rng.choice([-1, 1], len(CROSS_PAIRS))
        out.append(s)
    return out[:n_starts]


def _distinct(cands, maxn=6):
    """Keep solutions that differ by more than one tolerance (best first)."""
    out = []
    for c, x in cands:
        if all(param_distance(x, y) > 1.0 for _, y in out):
            out.append((c, x))
        if len(out) >= maxn:
            break
    return out


def _estimate_nuisance(f, S, p, use22):
    """Initial reference-plane delays (grid search) and S21 phase (closed form)."""
    Sm = model(p, f)
    taus = np.linspace(-TAU_MAX_FACTOR / BW, TAU_MAX_FACTOR / BW, 801)
    ph = np.outer(taus, f)

    def best_tau(meas, mod, k):          # maximise Re sum(meas*conj(mod)*e^{j k pi f tau})
        v = meas * np.conj(mod)
        return taus[int(np.argmax(np.real(np.exp(1j * k * np.pi * ph) @ v)))]

    t1 = best_tau(S[:, 0, 0], Sm[:, 0, 0], 4)
    if use22:
        t2 = best_tau(S[:, 1, 1], Sm[:, 1, 1], 4)
    else:                                # S21 only: phase-offset-invariant metric
        v = S[:, 1, 0] * np.conj(Sm[:, 1, 0])
        E = np.exp(1j * 2 * np.pi * (ph + t1 * f[None, :]))
        t2 = taus[int(np.argmax(np.abs(E @ v)))]
    e = np.exp(-2j * np.pi * f * (t1 + t2))
    phi = float(np.angle(np.sum(S[:, 1, 0] * np.conj(Sm[:, 1, 0] * e))))
    return np.array([t1, t2, phi])


def _flip_refl(S, sg):
    """S with S11/S22 multiplied by sg (used so the delay estimator sees the model's sign)."""
    if sg > 0:
        return S
    out = S.copy()
    out[:, 0, 0] = -S[:, 0, 0]
    out[:, 1, 1] = -S[:, 1, 1]
    return out


@dataclass
class FitResult:
    p: np.ndarray                  # physical parameters (same layout as always)
    nuisance: np.ndarray           # [tau1 (s), tau2 (s), phi (rad)]; zeros if magnitude fit
    mode: str                      # "complex" | "magnitude"
    cost: float                    # final-stage cost
    n_resid: int
    starts: list                   # stage-1 results [(cost, p)], best first
    candidates: list               # distinct final-stage solutions [(cost, p)]
    sigma: np.ndarray              # 1-sigma of the physical parameters (nan = not estimated, inf = unidentifiable)
    corr_pairs: list               # [(name_a, name_b, rho)] with |rho| > 0.95
    cond: float                    # column-scaled condition number of the Jacobian
    weakest: list                  # [(name, loading)] of the least-determined combination
    noise: float                   # noise sigma assumed for the uncertainties
    used_s22: bool
    agree_rms: float               # rms of the linear residual per term (in-band)
    ambiguous: bool                # a materially different solution fits about as well
    runner_up: Optional[tuple]     # (cost, distance in tolerances, delta chi2)
    notes: list = field(default_factory=list)
    chi2_red: float = float('nan') # sum(residual^2) / (n * noise^2): ~1 = at the noise level
    refl_sign: float = 1.0         # -1 = measured S11/S22 have the opposite sign to the model


def _identifiability(prob, x, lo, hi, noise_est=NOISE_FLOOR):
    """Jacobian-based uncertainties, correlations and conditioning."""
    names = param_names() + (["tau1 (ref plane 1)", "tau2 (ref plane 2)", "phi (S21 phase)"]
                             if prob.cplx else [])
    span = hi - lo
    free = np.where((x - lo > 1e-3 * span) & (hi - x > 1e-3 * span))[0]
    r0 = prob.lin(x)
    cols = []
    for j in free:
        h = 1e-6 * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        cols.append((prob.lin(xp) - prob.lin(xm)) / (2 * h))
    J = np.array(cols).T
    cn = np.maximum(np.linalg.norm(J, axis=0), 1e-300)
    U, s, Vt = np.linalg.svd(J / cn, full_matrices=False)
    rank = s > 1e-9 * s[0]
    noise = max(float(np.sqrt(np.mean(r0 ** 2))), noise_est, NOISE_FLOOR)
    cov_n = (Vt[rank].T / s[rank] ** 2) @ Vt[rank]
    cov = cov_n / np.outer(cn, cn) * noise ** 2
    sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if (~rank).any():                                  # null directions -> unidentifiable
        null = Vt[~rank]
        for k in range(len(free)):
            if np.any(np.abs(null[:, k]) > 0.3):
                sig[k] = np.inf
    sigma_full = np.full(len(x), np.nan)
    sigma_full[free] = sig
    pairs = []
    for a in range(len(free)):
        for b in range(a + 1, len(free)):
            d = sig[a] * sig[b]
            if np.isfinite(d) and d > 0:
                rho = cov[a, b] / d
                if abs(rho) > 0.95:
                    pairs.append((names[free[a]], names[free[b]], float(rho)))
    pairs.sort(key=lambda t: -abs(t[2]))
    v = Vt[-1]
    order = np.argsort(-np.abs(v))
    weakest = [(names[free[k]], float(v[k])) for k in order[:4] if abs(v[k]) > 0.3]
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return sigma_full, pairs[:5], cond, weakest, noise


def extract_robust(f, S, n_starts=N_STARTS_DEFAULT, seed=1, mode=None) -> FitResult:
    """Global magnitude search -> complex refinement -> identifiability analysis."""
    f = np.asarray(f, float)
    notes: List[str] = []
    req = (mode or FIT_MODE).lower()
    phase = has_phase(f, S)
    if req == "auto":
        use = "complex" if phase else "magnitude"
        if not phase:
            notes.append("Data has no usable phase (all angles ~0): magnitude-only fit.")
    elif req == "complex" and not phase:
        use = "magnitude"
        notes.append("Complex fit requested but the data has no usable phase: "
                     "fell back to magnitude-only.")
    elif req in ("complex", "magnitude"):
        use = req
    else:
        raise ValueError("mode must be 'auto', 'complex' or 'magnitude'")
    use22 = has_s22(S)
    noise_est = estimate_noise(S)
    fl = max(1e-3, 4.0 * noise_est)         # dB terms ignore levels below ~4 sigma
    if not use22:
        notes.append("No usable S22 in the data: S11/S21 only.")
    if CROSS_PAIRS and use == "magnitude":
        notes.append("Cross-couplings with magnitude-only data: the sign of each "
                     "cross-coupling (which side its zero is on) is poorly determined.")

    # ---- stage 1: magnitude multi-start --------------------------------
    pm = _Problem(f, S, "magnitude", use22, fl)
    lo, hi = pm.bounds()
    p0 = nominal_params()
    starts = []
    attempt = 0
    chi_first = None
    while True:
        for st in _make_starts(max(1, n_starts), seed + 7919 * attempt, p0, broad=attempt > 0):
            st = np.clip(st, lo + 1e-9, hi - 1e-9)
            r = least_squares(pm.lin, st, bounds=(lo, hi), x_scale=pm.xscale())
            starts.append((float(r.cost), r.x))
        starts.sort(key=lambda t: t[0])
        # residual vs noise of the best few distinct solutions (linear chi^2)
        top = _distinct(starts, 4)
        chi_red = min(float(np.sum(pm.lin(x) ** 2)) for _, x in top) / (
            len(pm.lin(top[0][1])) * noise_est ** 2)
        chi_first = chi_red if chi_first is None else chi_first
        if chi_red <= CHI2_OK or attempt >= MAX_ESCALATIONS:
            break
        attempt += 1
    if attempt:
        notes.append("Search escalated to %d starts (%d extra round%s): residuals were above "
                     "the noise level (chi2 %.1f -> %.1f)." % (len(starts), attempt,
                                                              "" if attempt == 1 else "s",
                                                              chi_first, chi_red))
    starts.sort(key=lambda t: t[0])
    distinct = _distinct(starts, 6)

    # ---- stage 2: complex refinement of each distinct candidate --------
    refl_best = 1.0
    if use == "complex":
        cands, prob = [], None
        for sg in (1.0, -1.0):                  # normal / sign-flipped reflections
            pr = _Problem(f, S, "complex", use22, fl, refl=sg)
            lo, hi = pr.bounds()
            sg_c = []
            for _, px in distinct + [(0.0, p0)]:
                nu = _estimate_nuisance(f, _flip_refl(S, sg), px, use22)
                x0 = np.clip(np.concatenate([px, [nu[0] * BW, nu[1] * BW, nu[2]]]),
                             lo + 1e-9, hi - 1e-9)
                r = least_squares(pr.lin, x0, bounds=(lo, hi), x_scale=pr.xscale())
                sg_c.append((float(r.cost), r.x))
            sg_c.sort(key=lambda t: t[0])
            if prob is None or sg_c[0][0] < min(c for c, _ in cands):
                cands, prob, refl_best = sg_c, pr, sg
        lo, hi = prob.bounds()
        if refl_best < 0:
            notes.append("Measured S11/S22 have the opposite sign to the model (about -1 "
                         "out of band): sign-flipped reflections were fitted. This is a "
                         "convention difference, not a fault in the filter.")
        final = _distinct(cands, 6)
    else:
        prob = pm
        final = distinct
    # Fits use the purely LINEAR residuals (the maximum-likelihood estimator for
    # Gaussian noise: what the uncertainty analysis assumes). Re-run each candidate to
    # convergence, then rank by linear chi^2; the dB-weighted cost is reported only.
    polished = []
    for _, cx in final:
        rp = least_squares(prob.lin, cx, bounds=(lo, hi), x_scale=prob.xscale())
        polished.append((float(0.5 * np.sum(prob.full(rp.x) ** 2)), rp.x))
    final = sorted(polished, key=lambda cx: float(np.sum(prob.lin(cx[1]) ** 2)))
    best_cost, xbest = final[0]

    nres = len(prob.full(xbest))

    # ---- stage 3: identifiability ---------------------------------------
    sigma_full, pairs, cond, weakest, noise = _identifiability(prob, xbest, lo, hi, noise_est)

    # ---- ambiguity: can the data reject a materially different solution? ----
    # delta chi^2 = (sum r_alt^2 - sum r_best^2) / noise^2 on the linear residuals.
    chi_best = float(np.sum(prob.lin(xbest) ** 2))
    runner, ambiguous = None, False
    for rc, rx in final[1:]:
        dist = param_distance(rx, xbest)
        if dist <= 2.0:                               # same basin, not a competitor
            continue
        dchi = (float(np.sum(prob.lin(rx) ** 2)) - chi_best) / noise ** 2
        if runner is None or dchi < runner[2]:
            runner = (rc, dist, dchi)
    if runner is not None:
        ambiguous = runner[2] < AMBIG_DCHI2
    inband = np.abs(f - F0) <= BW / 2
    lin = prob.lin(xbest)
    nterm = len(prob.meas) * (2 if prob.cplx else 1)
    blocks = lin.reshape(nterm, len(f))
    agree = float(np.sqrt(np.mean(blocks[:, inband] ** 2))) if inband.any() else np.nan

    nuis = prob.nuis(xbest) if prob.cplx else np.zeros(3)
    return FitResult(
        p=xbest[:PARAM_LEN].copy(), nuisance=nuis, mode=prob.cplx and "complex" or "magnitude",
        cost=best_cost, n_resid=nres, starts=[(c, x[:PARAM_LEN]) for c, x in starts],
        candidates=[(c, x[:PARAM_LEN]) for c, x in final],
        sigma=sigma_full[:PARAM_LEN].copy(), corr_pairs=pairs, cond=cond, weakest=weakest,
        noise=noise, used_s22=use22, agree_rms=agree, ambiguous=ambiguous,
        runner_up=runner, notes=notes,
        chi2_red=float(np.sum(lin ** 2) / (len(lin) * noise_est ** 2)),
        refl_sign=refl_best)


def model_with_planes(res: FitResult, f):
    """Fitted response as measured (reference-plane delays / phase applied)."""
    S = model(res.p, f)
    if res.mode == "complex":
        return apply_reference_planes(S, f, res.nuisance, res.refl_sign)
    return S


def fit_summary_lines(res: FitResult) -> List[str]:
    L: List[str] = []
    terms = "S11, S21" + (", S22" if res.used_s22 else "")
    if res.mode == "complex":
        L.append("Fit mode : COMPLEX %s (real+imag residuals; phase used)" % terms)
    else:
        L.append("Fit mode : MAGNITUDE-ONLY |%s| (no phase) - weaker identifiability" % terms.replace(", ", "|, |"))
    L.append("Search   : %d starts -> %d distinct solution(s) after the final stage; cost %.2e"
             % (len(res.starts), len(res.candidates), res.cost))
    if res.runner_up is not None:
        rc, dist, dchi = res.runner_up
        L.append("Runner-up: another solution %.1f tolerances away; delta chi2 = %.0f "
                 "(>%.0f = data rejects it)%s"
                 % (dist, dchi, AMBIG_DCHI2, "  <-- AMBIGUOUS" if res.ambiguous else ""))
    if res.mode == "complex":
        t1, t2, phi = res.nuisance
        L.append("Planes   : delay port1 %+.1f ps, port2 %+.1f ps, S21 phase %+.1f deg "
                 "(reference planes / convention)" % (t1 * 1e12, t2 * 1e12, np.degrees(phi)))
    L.append("Conditioning: scaled Jacobian condition number %.1e (noise assumed %.1e; "
             "residual/noise chi2 = %.1f)" % (res.cond, res.noise, res.chi2_red))
    if res.weakest and res.cond > 1e3:
        L.append("Weakest combination: " + " ".join("%+.2f*%s" % (v, n.split(" ")[0])
                                                    for n, v in res.weakest))
    for a, b, rho in res.corr_pairs[:3]:
        L.append("Correlated: %s ~ %s (rho %+.3f)" % (a.split(" ")[0], b.split(" ")[0], rho))
    names = [n.split(" ")[0] for n in param_names()]
    parts = []
    for n, s in zip(names, res.sigma):
        parts.append("%s %s" % (n, "n/a" if np.isnan(s) else ("inf" if np.isinf(s) else "+/-%.4f" % s)))
    L.append("Uncertainty (1 sigma): " + ", ".join(parts))
    for n in res.notes:
        L.append("Note     : " + n)
    return L


def identifiability_warnings(res: FitResult) -> List[Tuple[str, str]]:
    """Severity-tagged warnings from the fit structure (CRITICAL => unreliable).
    Tolerance-based warnings (parameter uncertainty vs tolerance) are not issued."""
    W: List[Tuple[str, str]] = []
    if res.mode == "magnitude":
        W.append(("WARNING", "Magnitude-only fit: several parameter sets can give similar "
                  "magnitudes. Re-measure/export with phase for a complex fit."))
        if not res.used_s22:
            W.append(("WARNING", "No S22 available: a port-reversed solution cannot be ruled "
                      "out for an asymmetric filter."))
    if res.ambiguous:
        rc, dist, dchi = res.runner_up
        W.append(("CRITICAL", "Ambiguous fit: a different solution (%.1f tolerances away) cannot be "
                  "rejected by the data (delta chi2 = %.0f < %.0f). Advice is not reliable."
                  % (dist, dchi, AMBIG_DCHI2)))
    if np.isfinite(res.chi2_red) and res.chi2_red > 4:
        W.append(("WARNING", "Residuals are %.0fx the data's noise level: model mismatch, "
                  "measurement systematics, or a local minimum. Check the fit plot." % np.sqrt(res.chi2_red)))
    if res.cond > 1e8:
        W.append(("WARNING", "Ill-conditioned fit (condition number %.1e): some parameter "
                  "combinations are barely constrained by the data." % res.cond))
    if res.corr_pairs:
        W.append(("WARNING", "Strongly correlated parameters: " + "; ".join(
            "%s~%s (%+.2f)" % (a.split(" ")[0], b.split(" ")[0], r)
            for a, b, r in res.corr_pairs[:3])))
    if res.mode == "complex":
        tmax = TAU_MAX_FACTOR / BW
        if np.any(np.abs(res.nuisance[:2]) >= 0.95 * tmax):
            W.append(("WARNING", "Reference-plane delay is at its search limit: phase may be "
                      "unreliable (un-de-embedded fixture or wrong phase data)."))
        if np.isfinite(res.agree_rms) and res.agree_rms > 0.05:
            W.append(("CRITICAL", "Poor complex agreement in the passband (rms %.3f): "
                      "the model does not describe the measured phase/magnitude." % res.agree_rms))
    return W


def report(p, path_txt):
    M, r1, rn, delta = unpack(p)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("\nExtracted core coupling matrix M (%dx%d):" % (N, N))
    print(M)
    print("R1 = %.4f   RN = %.4f   (qe = %.3f)" % (r1, rn, 1 / r1))
    print("Fitted Qu = %.1f  (delta = %.5f)" % (1 / (delta * FBW), delta))
    np.savetxt(path_txt, M, fmt="%.6f",
               header="Extracted %dx%d inline coupling matrix | f0=%g MHz BW=%g MHz\n"
                      "R1=%.6f RN=%.6f delta=%.6f Qu=%.2f"
                      % (N, N, F0 / 1e6, BW / 1e6, r1, rn, delta, 1 / (delta * FBW)))


# ---------------- plotting ----------------
def plot_compare(f, S_orig, S_ext, title):
    fm = f / 1e6
    fig, ax = plt.subplots(3, 1, figsize=(9, 11))
    pb = (f > F0 - 0.75 * BW) & (f < F0 + 0.75 * BW)

    ax[0].plot(fm, db(S_orig[:, 0, 0]), "b-", lw=2, label="Original S11")
    ax[0].plot(fm, db(S_ext[:, 0, 0]), "r--", lw=1.5, label="Extracted S11")
    ax[0].set_title("S11 (return loss)")
    ax[0].set_ylim(-60, 0)

    ax[1].plot(fm, db(S_orig[:, 1, 0], 1e-6), "b-", lw=2, label="Original S21")      # -120 dB floor
    ax[1].plot(fm, db(S_ext[:, 1, 0], 1e-6), "r--", lw=1.5, label="Extracted S21")
    ax[1].set_title("S21 (through loss)")
    ax[1].set_ylim(-120, 0)

    ax[2].plot(fm[pb], db(S_orig[pb, 1, 0]), "b-", lw=2, label="Original S21")
    ax[2].plot(fm[pb], db(S_ext[pb, 1, 0]), "r--", lw=1.5, label="Extracted S21")
    ax[2].set_title("S21 passband detail")

    for a in ax:
        a.set_xlabel("Frequency (MHz)")
        a.set_ylabel("Magnitude (dB)")
        a.grid(True, alpha=0.4)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


# ---------------- demo file ----------------
DEMO_CROSS = 0.15            # fallback demo cross-coupling size; signs alternate per pair


def demo_zero_frequencies():
    """Where the demo puts its transmission zeros: +1.6 BW, then -1.4 BW, alternating."""
    return [F0 + 1.6 * BW if k % 2 == 0 else F0 - 1.4 * BW for k in range(len(CROSS_PAIRS))]


def demo_cross_values():
    """Cross-coupling values that put the demo zeros at demo_zero_frequencies() for ANY
    order N (a fixed value only gives a visible zero near N = 5)."""
    K = len(CROSS_PAIRS)
    fz = demo_zero_frequencies()
    oms = [(f / F0 - F0 / f) / FBW for f in fz]
    base = nominal_params()

    def res(c):
        q = base.copy()
        q[IDX_CROSS] = c
        return [_tz_minor(q, om) for om in oms]

    best = None
    for scale in (1.0, 2.0, 0.5):
        c0 = np.array([(0.15 if f > F0 else -0.15) * scale for f in fz])
        try:
            r = least_squares(res, c0, bounds=(-0.6, 0.6), xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
        if r.cost < 1e-14:
            break
    if best is not None and best.cost < 1e-10:
        return best.x
    return DEMO_CROSS * (-1.0) ** np.arange(K)


def make_demo(path):
    rng = np.random.default_rng(7)
    p0 = nominal_params()
    from_synth = (SYNTH_P is not None and SYNTH_KEY == (N, F0, BW, tuple(CROSS_PAIRS))
                  and len(SYNTH_P) == PARAM_LEN)
    if from_synth:                               # a detuned copy of the synthesised target
        p0 = SYNTH_P.copy()
    elif CROSS_PAIRS:
        p0[IDX_CROSS] = demo_cross_values()      # zeros at demo_zero_frequencies()
    p = p0.copy()
    p[:N] += rng.normal(0, 0.03, N)                 # detuned resonators
    p[N:2 * N + 1] *= 1 + rng.normal(0, 0.03, N + 1)    # coupling / load errors
    if CROSS_PAIRS:
        p[IDX_CROSS] += rng.normal(0, 0.01, len(CROSS_PAIRS))
    f = np.linspace(max(F0 - 3 * BW, 0.2 * F0), F0 + 3 * BW, 601)
    S = apply_reference_planes(model(p, f), f, [0.15e-9, -0.10e-9, 0.4])   # fixture delays
    write_s2p(path, f, S, "demo inline filter (perturbed, with reference-plane delays)"
              + (" and cross-couplings " + cross_text() if CROSS_PAIRS else ""))
    print("Demo file written:", path)
    if from_synth:
        print("Demo = the synthesised target with small random detuning (same zero position(s)).")
    if CROSS_PAIRS:
        print("Demo transmission zero(s) near: " + ", ".join(
            "%.0f MHz" % (x / 1e6) for x in find_notches(p0)) + "  (cross values: " + ", ".join(
            "%+.3f" % v for v in p0[IDX_CROSS]) + ")")                                  # matching target = the unperturbed design
        tpath = os.path.splitext(path)[0] + "_target.txt"
        M, r1, rn, delta = unpack(p0)
        np.savetxt(tpath, M, fmt="%.6f",
                   header="Demo target (unperturbed design) | f0=%g MHz BW=%g MHz\n"
                          "R1=%.6f RN=%.6f delta=%.6f" % (F0 / 1e6, BW / 1e6, r1, rn, delta))
        print("Demo target written:", tpath, "(give it at the target prompt)")


# =====================================================================
#  LEVEL 1 TUNING ADVISOR  (model-based electrical advice only)
# =====================================================================
# Parameter vector order (unchanged from the extraction code):
#   p = [M11..M55 (N), M12 M23 M34 M45 (N-1), R1, RN, delta]
#
# Sign / direction conventions, derived from the model
#   A = M + Omega*I - j(R + delta*I),   Omega = (f/f0 - f0/f)/FBW
#   * Resonator i resonates where Omega + M_ii = 0, i.e. Omega_r = -M_ii.
#     Omega ~ 2*(f - f0)/BW, so f_r - f0 ~ -M_ii * BW/2.
#     => extracted M_ii ABOVE target  = resonator frequency BELOW target
#        (raise its frequency); BELOW target = frequency too high (lower it).
#   * R1, RN: external Qe = 1/R in these FBW-normalised units, so a LARGER R
#     means STRONGER input/output coupling.
#   * delta = 1/(Qu*FBW): LARGER delta means MORE modelled loss (LOWER Qu).
PARAM_LEN = 2 * N + 2
IDX_DIAG = slice(0, N)
IDX_COUP = slice(N, 2 * N - 1)
IDX_R1, IDX_RN, IDX_DELTA = 2 * N - 1, 2 * N, 2 * N + 1
IDX_CROSS = slice(2 * N + 2, 2 * N + 2)      # grows when cross-couplings are enabled

# Set these to skip the prompts (leave TARGET_FILE = None to be asked).
TARGET_FILE = None            # e.g. "my_target.txt"
SYNTH_TARGET = None           # set by 'Synthesise target': used as the default target afterwards
SYNTH_P = None                # the synthesised parameter vector (the demo file is built from it)
SYNTH_KEY = None              # design it belongs to: (N, F0, BW, cross pairs)
ALLOW_NOMINAL_FALLBACK = False   # True = never ask, always use the nominal design

ST_OK = "WITHIN TOLERANCE"
ST_ADJ = "ADJUSTMENT SUGGESTED"
ST_NONE = "NO TARGET"

NOMINAL_WARNING = (
    "Target is the NOMINAL starting design from nominal_params() "
    "(ideal Chebyshev synthesis). It is NOT necessarily your final "
    "optimised design - treat all advice relative to an ideal filter only.")


class TargetError(ValueError):
    """Raised when a target design is missing or invalid."""


@dataclass
class Tolerances:
    """Separate tolerances per parameter type (they have different units)."""
    coupling_rel: float = 0.05       # |dk| / |k_target|
    coupling_abs_floor: float = 0.02  # used when |k_target| is ~0 (M units)
    zero_target_threshold: float = 1e-3
    diagonal_abs: float = 0.02       # M_ii units (~ 0.02*BW/2 = 0.5 MHz here)
    load_rel: float = 0.05           # R1, RN relative
    loss_rel: float = 0.20           # delta relative
    cross_abs: float = 0.02          # cross-couplings (signed): absolute, M units


@dataclass
class TuningRow:
    name: str
    kind: str            # resonator | coupling | port | loss
    target: float
    extracted: float
    error: float
    score: float         # |error| / tolerance (dimensionless); NaN = no target
    status: str
    action: str
    note: str = ""
    idx: int = -1        # index into the parameter vector
    sigma: float = float('nan')   # 1-sigma fit uncertainty (nan = unknown)


def param_names() -> List[str]:
    names = ["M%d%d (resonator %d)" % (i + 1, i + 1, i + 1) for i in range(N)]
    names += ["M%d%d (coupling)" % (i + 1, i + 2) for i in range(N - 1)]
    names += ["R1 (input coupling)", "RN (output coupling)", "delta (loss)"]
    names += ["M%d%d (cross)" % (a + 1, b + 1) for a, b in CROSS_PAIRS]
    return names


# ---------------- target loading / validation ----------------
def target_from_numbers(nums: Sequence[float],
                        header: Optional[dict] = None):
    """Build a full-length target vector (NaN = not supplied) + warnings."""
    header = header or {}
    warns: List[str] = []
    nums = np.asarray(nums, dtype=float)
    if not np.all(np.isfinite(nums)):
        raise TargetError("Target contains NaN/inf values.")

    vec = np.full(PARAM_LEN, np.nan)
    if nums.size == N * N:                       # coupling matrix
        M = nums.reshape(N, N)
        if np.max(np.abs(M - M.T)) > 1e-3 * max(1.0, np.max(np.abs(M))):
            raise TargetError("Target matrix is not symmetric.")
        allowed = set(CROSS_PAIRS)
        for i in range(N):
            for j in range(i + 2, N):
                if abs(M[i, j]) > 1e-6 and (i, j) not in allowed:
                    raise TargetError(
                        "Target matrix has a cross-coupling M%d%d that is not "
                        "enabled. Enter %d-%d at the design prompt "
                        "(cross-couplings) to include it." %
                        (i + 1, j + 1, i + 1, j + 1))
        vec[IDX_DIAG] = np.diag(M)
        vec[IDX_COUP] = [M[i, i + 1] for i in range(N - 1)]
        vec[IDX_CROSS] = [M[a, b] for a, b in CROSS_PAIRS]
        for key, idx in (("R1", IDX_R1), ("RN", IDX_RN), ("delta", IDX_DELTA)):
            if key in header:
                vec[idx] = header[key]
    elif nums.size == PARAM_LEN:                 # full parameter vector
        vec[:] = nums
    else:
        raise TargetError(
            "Expected %d numbers (%dx%d matrix) or %d numbers (parameter "
            "vector), found %d." % (N * N, N, N, PARAM_LEN, nums.size))

    for idx, label in ((IDX_R1, "R1"), (IDX_RN, "RN")):
        if np.isfinite(vec[idx]) and vec[idx] <= 0:
            raise TargetError("Target %s must be > 0." % label)
    if np.isfinite(vec[IDX_DELTA]) and vec[IDX_DELTA] < 0:
        raise TargetError("Target delta must be >= 0.")

    # ---- convention / normalisation plausibility checks (warnings) ----
    nominal = nominal_params()
    c = vec[IDX_COUP]
    if np.any(np.abs(c) > 3) or np.any(np.abs(vec[IDX_DIAG]) > 1):
        warns.append("Target values are unusually large for FBW-normalised "
                     "units - normalisation may differ from this program.")
    ratio = np.abs(c) / np.abs(nominal[IDX_COUP])
    if np.any(ratio > 3) or np.any(ratio < 1 / 3):
        warns.append("Target couplings differ >3x from the ideal Chebyshev "
                     "values - check it uses the same FBW normalisation "
                     "(M, not k or MHz).")
    for key, ref in (("f0", F0 / 1e6), ("BW", BW / 1e6)):
        if key in header and abs(header[key] - ref) > 1e-6 * max(ref, 1):
            warns.append("Target file header says %s=%g MHz but this program "
                         "uses %g MHz - normalisation mismatch." %
                         (key, header[key], ref))
    if np.isnan(vec[IDX_R1:]).any():
        warns.append("Target has no R1/RN/delta values: those comparisons "
                     "are skipped (add '# R1=.. RN=.. delta=..' to a matrix "
                     "file, or supply the full %d-value vector)." % PARAM_LEN)
    return vec, warns


def parse_target_text(text: str):
    header: dict = {}
    nums: List[float] = []
    pat = re.compile(r"\b(R1|RN|delta|f0|BW)\s*=\s*([-+0-9.eE]+)")
    for line in text.splitlines():
        data, _, comment = line.partition("#")
        for key, val in pat.findall(comment):
            try:
                header[key] = float(val)
            except ValueError:
                pass
        for tok in data.replace(",", " ").replace(";", " ").split():
            try:
                nums.append(float(tok))
            except ValueError:
                raise TargetError("Cannot read '%s' as a number." % tok)
    return nums, header


def load_target_file(path: str):
    try:
        with open(path, "r") as fh:
            text = fh.read()
    except OSError as exc:
        raise TargetError("Cannot open target file: %s" % exc)
    nums, header = parse_target_text(text)
    return target_from_numbers(nums, header)


def get_target(source: Optional[str], allow_nominal: bool = False):
    """Return (target_vector, warnings, label). Never substitutes silently."""
    if not source:
        if allow_nominal:
            warns, label = [NOMINAL_WARNING], "NOMINAL FALLBACK (not an optimised design)"
            if CROSS_PAIRS:
                warns.append("The nominal design has NO transmission zeros: the advisor and "
                             "Auto-tune will drive cross-coupling %s to 0. Give a target "
                             "file that contains it (the demo writes *_target.txt)." % cross_text())
                label += " - no transmission zeros"
            return nominal_params(), warns, label
        raise TargetError(
            "No target design supplied and nominal fallback is not enabled. "
            "Provide a target file or enable ALLOW_NOMINAL_FALLBACK.")
    vec, warns = load_target_file(source)
    return vec, warns, "file: " + os.path.basename(source)


# ---------------- comparison + ranking ----------------
def _status(score):
    """Within tolerance -> OK, otherwise an adjustment. Tolerances never
    trigger a CHECK / HOLD state."""
    if not np.isfinite(score):
        return ST_NONE
    return ST_OK if score <= 1.0 else ST_ADJ


def compare_to_target(p_ext: np.ndarray, p_tgt: np.ndarray,
                      tol: Optional[Tolerances] = None,
                      sigma: Optional[np.ndarray] = None) -> List[TuningRow]:
    """Compare extracted vs target. Unsorted; see rank_rows()."""
    tol = tol or Tolerances()
    p_ext = np.asarray(p_ext, float)
    p_tgt = np.asarray(p_tgt, float)
    if p_ext.shape != (PARAM_LEN,) or p_tgt.shape != (PARAM_LEN,):
        raise TargetError("Parameter vectors must have length %d." % PARAM_LEN)
    names = param_names()
    rows: List[TuningRow] = []

    def finish(idx, kind, tgt, ext, err, score, action_hi, action_lo, note=""):
        st = _status(score)
        if st == ST_OK:
            act = "No change needed."
        elif st == ST_NONE:
            act = "No target value supplied - not compared."
        else:
            act = action_hi if err > 0 else action_lo
        rows.append(TuningRow(names[idx], kind, tgt, ext, err, score, st,
                              act, note, idx))

    # A. resonator diagonals (signed error, absolute tolerance)
    for i in range(N):
        tgt, ext = p_tgt[i], p_ext[i]
        err = ext - tgt
        score = abs(err) / tol.diagonal_abs if np.isfinite(tgt) else np.nan
        df = abs(err) * BW / 2e6
        finish(i, "resonator", tgt, ext, err, score,
               "Raise resonator %d frequency (~%.2f MHz, approx.)" % (i + 1, df),
               "Lower resonator %d frequency (~%.2f MHz, approx.)" % (i + 1, df),
               "Not unique: resonator shifts also move via coupling")

    # B. adjacent couplings (compare magnitudes)
    for k in range(N - 1):
        idx = N + k
        tgt, ext = p_tgt[idx], p_ext[idx]
        if np.isfinite(tgt):
            err = abs(ext) - abs(tgt)
            if abs(tgt) >= tol.zero_target_threshold:
                score = abs(err) / (tol.coupling_rel * abs(tgt))
                note = "rel err %+.1f%%" % (100 * err / abs(tgt))
            else:                               # ~zero target: absolute tol
                score = abs(err) / tol.coupling_abs_floor
                note = "rel err undefined (target ~0); absolute tol used"
            if np.sign(ext) != np.sign(tgt) and abs(tgt) > 1e-3 and abs(ext) > 1e-3:
                note += "; sign differs (may be a convention difference)"
        else:
            err, score, note = np.nan, np.nan, ""
        finish(idx, "coupling", tgt, ext, err, score,
               "Reduce coupling between resonators %d-%d" % (k + 1, k + 2),
               "Increase coupling between resonators %d-%d" % (k + 1, k + 2),
               note)

    # C. input / output loading (relative tolerance)
    for idx, side in ((IDX_R1, "input"), (IDX_RN, "output")):
        tgt, ext = p_tgt[idx], p_ext[idx]
        if np.isfinite(tgt):
            err = ext - tgt
            score = abs(err) / (tol.load_rel * abs(tgt))
            note = "Qe ext %.3f vs target %.3f (Qe=1/R)" % (1 / ext, 1 / tgt)
        else:
            err, score, note = np.nan, np.nan, ""
        finish(idx, "port", tgt, ext, err, score,
               "Reduce %s coupling (model shows it stronger than target)" % side,
               "Increase %s coupling (model shows it weaker than target)" % side,
               note)

    # D. internal loss
    tgt, ext = p_tgt[IDX_DELTA], p_ext[IDX_DELTA]
    if np.isfinite(tgt):
        err = ext - tgt
        score = abs(err) / (tol.loss_rel * max(abs(tgt), 1e-12))
        qu_e = 1 / (ext * FBW) if ext > 1e-12 else np.inf
        qu_t = 1 / (tgt * FBW) if tgt > 1e-12 else np.inf
        note = "Qu fitted %.0f vs target %.0f" % (qu_e, qu_t)
    else:
        err, score, note = np.nan, np.nan, ""
    finish(IDX_DELTA, "loss", tgt, ext, err, score,
           "Investigate excess internal loss (model-based; cause unproven)",
           "Model shows less loss than target - no loss action suggested",
           note)

    # E. cross-couplings (signed: the sign decides which side the zero is on)
    for m, (a, b) in enumerate(CROSS_PAIRS):
        idx = 2 * N + 2 + m
        tgt, ext = p_tgt[idx], p_ext[idx]
        if np.isfinite(tgt):
            err = ext - tgt
            score = abs(err) / max(tol.coupling_rel * abs(tgt), tol.cross_abs)
            note = "signed; moves the transmission zero (target %+.4f)" % tgt
        else:
            err, score, note = np.nan, np.nan, ""
        finish(idx, "cross", tgt, ext, err, score,
               "Decrease cross-coupling %d-%d (extracted is above target)" % (a + 1, b + 1),
               "Increase cross-coupling %d-%d (extracted is below target)" % (a + 1, b + 1),
               note)
    if sigma is not None:                       # record the 1-sigma for display only
        sigma = np.asarray(sigma, float)
        for r in rows:
            r.sigma = float(sigma[r.idx])
    return rows


def rank_rows(rows: List[TuningRow]) -> List[TuningRow]:
    """Largest normalised discrepancy first; rows without target last."""
    def key(r):
        return (1, 0.0) if not np.isfinite(r.score) else (0, -r.score)
    return sorted(rows, key=key)


# ---------------- next best tuning move ----------------
STEP_FRACTION = 0.5   # correct about half of the model error per iteration


def _describe_move(r: TuningRow, tol: Tolerances) -> str:
    """One concrete, small, model-based electrical step for a row."""
    pc = int(round(100 * STEP_FRACTION))
    if r.kind == "resonator":
        amt = abs(r.error) * BW / 2e6
        return "%s resonator %d frequency by about %.2f MHz (%d%% of the ~%.2f MHz model-estimated error)" % (
            "Raise" if r.error > 0 else "Lower", r.idx + 1, amt * STEP_FRACTION, pc, amt)
    if r.kind == "coupling":
        k = r.idx - N
        verb = "Reduce" if r.error > 0 else "Increase"
        pair = "%d and %d" % (k + 1, k + 2)
        if abs(r.target) >= tol.zero_target_threshold:
            pct = 100 * abs(r.error) / abs(r.target)
            return "%s the coupling between resonators %s by about %.1f%% (%d%% of the %.1f%% error)" % (
                verb, pair, pct * STEP_FRACTION, pc, pct)
        return "%s the coupling between resonators %s by about %.3f in M units (%d%% of the error)" % (
            verb, pair, abs(r.error) * STEP_FRACTION, pc)
    if r.kind == "port":
        side = "input" if r.idx == IDX_R1 else "output"
        pct = 100 * abs(r.error) / abs(r.target)
        return "%s the %s coupling by about %.1f%% (%d%% of the %.1f%% error)" % (
            "Reduce" if r.error > 0 else "Increase", side, pct * STEP_FRACTION, pc, pct)
    if r.kind == "cross":
        a, b = CROSS_PAIRS[r.idx - (2 * N + 2)]
        return ("%s the cross-coupling between resonators %d and %d by about %.3f in M units "
                "(%d%% of the error; it moves the transmission zero)" % (
                    "Decrease" if r.error > 0 else "Increase", a + 1, b + 1,
                    abs(r.error) * STEP_FRACTION, pc))
    return r.action


def next_move_lines(rows: List[TuningRow], tol: Optional[Tolerances] = None) -> List[str]:
    """Plain-text direction for the single next best tuning move.
    Never issues a HOLD because of tolerances."""
    tol = tol or Tolerances()
    scored = [r for r in rank_rows(rows) if np.isfinite(r.score)]
    if not scored:
        return []
    out = [r for r in scored if r.score > 1.0]
    if not out:
        return []
    cand = next((r for r in out if r.kind != "loss"), out[0])
    lines = [">> " + _describe_move(cand, tol)]
    lines.append("   Why: %s has the largest normalised error among adjustable "
                 "parameters (%.2fx tolerance)." % (cand.name.split(" ")[0], cand.score))
    rest = [r for r in out if r is not cand][:3]
    if rest:
        lines.append("   Then, in order: " + ", ".join(
            "%s (%.2fx)" % (r.name.split(" ")[0], r.score) for r in rest))
    if cand.kind != "resonator" and any(r.kind == "resonator" for r in out):
        lines.append("   Note: resonators are also out of tolerance - tune those first; "
                     "coupling estimates are less reliable while they are detuned.")
    lines.append("   Make this ONE change, re-measure and re-extract before the next; "
                 "other parameters will shift too.")
    lines.append("   Model-based electrical direction only - not a mechanical instruction.")
    return lines


# ---------------- fit / data quality diagnostics ----------------
def param_bounds():
    k = len(CROSS_PAIRS)
    lo = np.concatenate([-0.6 * np.ones(N), 1e-3 * np.ones(N - 1), [0.1, 0.1, 0.0],
                         -0.6 * np.ones(k)])
    hi = np.concatenate([0.6 * np.ones(N), 3 * np.ones(N - 1),
                         [8.0, 8.0, max(0.3, 5.0 / (QU * FBW))], 0.6 * np.ones(k)])
    return lo, hi


def at_bound_mask(p: np.ndarray) -> np.ndarray:
    lo, hi = param_bounds()
    span = hi - lo
    return (p - lo <= 1e-3 * span) | (hi - p <= 1e-3 * span)


def diagnose_fit(f, S_meas, S_fit, p, starts, best_cost, mode="magnitude"):
    """Return (warnings, fit_ok). Critical warnings make fit_ok False."""
    warns: List[Tuple[str, str]] = []

    def crit(t): warns.append(("CRITICAL", t))
    def warn(t): warns.append(("WARNING", t))

    # --- data coverage / quality
    if len(f) < 2 or np.any(np.diff(f) <= 0):
        crit("Frequency axis is not strictly increasing.")
    if f.min() > F0 - BW or f.max() < F0 + BW:
        crit("Frequency coverage (%.1f-%.1f MHz) does not span f0 +/- BW "
             "(%.1f-%.1f MHz): skirts not constrained." %
             (f.min() / 1e6, f.max() / 1e6, (F0 - BW) / 1e6, (F0 + BW) / 1e6))
    inband = np.abs(f - F0) <= BW / 2
    if inband.sum() < 15:
        crit("Only %d points inside the passband (need >= 15): data too "
             "sparse." % int(inband.sum()))
    peak = max(np.abs(S_meas[:, 0, 0]).max(), np.abs(S_meas[:, 1, 0]).max())
    if peak > 1.05:
        warn("Measured |S| reaches %.3f (>1): check calibration / reference "
             "plane / data format." % peak)

    # --- agreement between measurement and model
    if inband.any():
        d21 = db(S_meas[:, 1, 0]) - db(S_fit[:, 1, 0])
        d11 = np.abs(S_meas[:, 0, 0]) - np.abs(S_fit[:, 0, 0])
        rms21 = float(np.sqrt(np.mean(d21[inband] ** 2)))
        rms11 = float(np.sqrt(np.mean(d11[inband] ** 2)))
        if rms21 > 0.3 or rms11 > 0.03:
            crit("Poor model agreement in passband (S21 rms %.2f dB, S11 rms "
                 "%.3f linear). Model may not describe this filter." %
                 (rms21, rms11))
    skirt = np.abs(f - F0) <= BW
    if skirt.any():
        rms_sk = float(np.sqrt(np.mean(
            (db(S_meas[:, 1, 0]) - db(S_fit[:, 1, 0]))[skirt] ** 2)))
        if rms_sk > 1.0:
            crit("Poor S21 agreement over f0 +/- BW (rms %.2f dB)." % rms_sk)

    # --- parameters at bounds / loss plausibility
    mask = at_bound_mask(p)
    if mask.any():
        hit = [param_names()[i] for i in np.where(mask)[0]]
        warn("Parameters at fit bounds (not converged / outside range): "
             + ", ".join(hit))
    delta = p[IDX_DELTA]
    qu_fit = 1 / (delta * FBW) if delta > 1e-12 else np.inf
    if qu_fit > 10000 or qu_fit < 20:
        warn("Fitted Qu = %s is implausible; loss value may be nonphysical "
             "or poorly determined (S21 depth, not just shape, sets loss)." %
             ("inf" if np.isinf(qu_fit) else "%.0f" % qu_fit))

    # --- non-uniqueness / local minima
    similar = [x for c, x in starts if c <= max(1.5 * best_cost, best_cost + 1e-6)]
    spread = 0.0
    for x in similar:
        spread = max(spread,
                     float(np.max(np.abs(x[IDX_DIAG] - p[IDX_DIAG]))),
                     float(np.max(np.abs(np.abs(x[IDX_COUP]) - np.abs(p[IDX_COUP])))),
                     float(np.max(np.abs(x[IDX_CROSS] - p[IDX_CROSS]), initial=0.0)))
    if spread > 0.05:
        warn("Other starting points reached an equally good fit with "
             "different parameters (spread %.3f). Magnitude-only fitting is "
             "not unique (e.g. reversed/detuned solutions)." % spread)
    elif len(similar) == 1 and len(starts) > 1:
        warn("Only 1 of %d starts reached the best fit: possible local "
             "minimum - repeat with other start values to confirm." % len(starts))

    if mode != "complex":
        warn("Fit uses magnitudes only (no phase): several parameter "
             "combinations can give similar magnitude responses.")
    fit_ok = not any(sev == "CRITICAL" for sev, _ in warns)
    return warns, fit_ok


# ---------------- iteration tracking ----------------
def iteration_changes(p_prev: np.ndarray, p_now: np.ndarray):
    """Per-parameter change since the previous iteration + large-change flags."""
    out, flagged = [], []
    names = param_names()
    for i in range(PARAM_LEN):
        d = p_now[i] - p_prev[i]
        lim = (0.1 if i < N else 0.15 if i < 2 * N - 1 else
               0.1 if i >= 2 * N + 2 else 0.25 * max(abs(p_prev[i]), 1e-9))
        big = abs(d) > lim
        out.append((names[i], p_prev[i], p_now[i], d, big))
        if big:
            flagged.append(names[i])
    return out, flagged


# ---------------- report formatting ----------------
NEXT_STEPS = [
    "Review the highest-ranked electrical discrepancies above.",
    "Select only a few adjustments for the next iteration (not all at once).",
    "Have a qualified operator adjust the filter using the proper hardware "
    "procedure for this topology.",
    "Re-measure with a consistent calibration and measurement setup.",
    "Run the extraction again.",
    "Compare the new matrix and discrepancies with the previous iteration "
    "(enter the previous *_matrix.txt when prompted).",
    "Keep or revise the approach based on the measured result; record every "
    "measurement and tuning decision.",
]

DISCLAIMER = (
    "These are MODEL-BASED electrical suggestions, not verified physical "
    "tuning instructions. A coupling-matrix error does not uniquely identify "
    "a mechanical adjustment, and changing one element generally moves "
    "others. General tendency only (verify for your hardware): tuning "
    "screws/elements mainly shift resonator frequency; iris/aperture/gap "
    "changes mainly alter coupling.")


def format_report(rows, label, tol, target_warns, fit_warns, fit_ok,
                  changes=None, flagged=None, fit_lines=None) -> str:
    L: List[str] = []
    L.append("=" * 72)
    L.append("LEVEL 1 TUNING ADVISOR  (model-based, electrical only)")
    L.append("=" * 72)
    L.append("Target : " + label)
    L.append("Design : N=%d f0=%g MHz BW=%g MHz (FBW=%.3f) cross=%s" %
             (N, F0 / 1e6, BW / 1e6, FBW, cross_text()))
    L.append("Tol    : coupling %.0f%%  diag %.3f  R1/RN %.0f%%  delta %.0f%%" %
             (100 * tol.coupling_rel, tol.diagonal_abs, 100 * tol.load_rel,
              100 * tol.loss_rel))
    L.append("Fit reliability: " + ("OK" if fit_ok else
             "NOT RELIABLE - treat advice below as unverified"))
    for w in target_warns:
        L.append("  TARGET NOTE: " + w)
    for sev, w in fit_warns:
        L.append("  %s: %s" % (sev, w))
    L.append("")
    move = next_move_lines(rows, tol)
    if move:
        L.append("NEXT BEST MOVE")
        L.extend("  " + t for t in move)
        L.append("")
    L.append("Ranked by normalised error = |error| / tolerance  (>1 = outside tolerance)")
    L.append("-" * 72)
    hdr = "%2s %-24s %9s %9s %8s %9s %6s" % ("#", "Parameter", "Target", "Extracted",
                                              "+/-1sig", "Error", "x tol")
    L.append(hdr)
    for n, r in enumerate(rank_rows(rows), 1):
        tgt = "%+9.4f" % r.target if np.isfinite(r.target) else "%9s" % "n/a"
        err = "%+9.4f" % r.error if np.isfinite(r.error) else "%9s" % "n/a"
        sc = "%6.2f" % r.score if np.isfinite(r.score) else "%6s" % "n/a"
        sg = ("%8s" % "n/a" if np.isnan(r.sigma) else
              "%8s" % "inf" if np.isinf(r.sigma) else "%8.4f" % r.sigma)
        L.append("%2d %-24s %s %+9.4f %s %s %s" % (n, r.name[:24], tgt, r.extracted, sg, err, sc))
        L.append("     [%s]" % r.status)
        L.append("     -> " + r.action)
        if r.note:
            L.append("     (%s)" % r.note)
    if fit_lines:
        L.append("")
        L.append("FIT METHOD AND IDENTIFIABILITY")
        L.extend("  " + t for t in fit_lines)
    if changes is not None:
        L.append("")
        L.append("Change since previous iteration:")
        for name, a, b, d, big in changes:
            L.append("  %-24s %+9.4f -> %+9.4f  (%+.4f)%s" %
                     (name[:24], a, b, d, "  LARGE" if big else ""))
        if flagged:
            L.append("  WARNING: large/unstable changes in: " + ", ".join(flagged) +
                     ". Check measurement repeatability and fit before trusting.")
    L.append("")
    L.append("NEXT STEPS")
    for i, s in enumerate(NEXT_STEPS, 1):
        L.append(" %d. %s" % (i, s))
    L.append("")
    L.append("NOTE: " + DISCLAIMER)
    return "\n".join(L)


def save_tuning_csv(path: str, rows: List[TuningRow]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "parameter", "kind", "target", "extracted",
                    "error", "sigma", "score_x_tol", "status", "action", "note"])
        for n, r in enumerate(rank_rows(rows), 1):
            w.writerow([n, r.name, r.kind, r.target, r.extracted, r.error,
                        r.sigma, r.score, r.status, r.action, r.note])


def run_tuning_advisor(f, S, S_fit, p, starts, best_cost, target_source,
                       allow_nominal=False, tol=None, previous=None,
                       out_base=None, fit=None):
    """Full advisor pass. Prints and (optionally) saves the report.
    Pass fit=FitResult (from extract_robust) to include identifiability."""
    tol = tol or Tolerances()
    try:
        p_tgt, t_warns, label = get_target(target_source, allow_nominal)
    except TargetError as exc:
        print("\nTuning advisor not run: %s" % exc)
        return None
    f_warns, fit_ok = diagnose_fit(f, S, S_fit, p, starts if fit is None else [],
                                   best_cost, "magnitude" if fit is None else fit.mode)
    sigma = fit_lines = None
    if fit is not None:
        f_warns = list(f_warns) + identifiability_warnings(fit)
        fit_ok = not any(sev == "CRITICAL" for sev, _ in f_warns)
        sigma, fit_lines = fit.sigma, fit_summary_lines(fit)
    rows = compare_to_target(p, p_tgt, tol, sigma)
    changes = flagged = None
    if previous is not None:
        changes, flagged = iteration_changes(previous, p)
    text = format_report(rows, label, tol, t_warns, f_warns, fit_ok,
                         changes, flagged, fit_lines)
    print("\n" + text)
    if out_base:
        with open(out_base + "_tuning.txt", "w") as fh:
            fh.write(text + "\n")
        save_tuning_csv(out_base + "_tuning.csv", rows)
        print("\nSaved:", out_base + "_tuning.txt")
        print("Saved:", out_base + "_tuning.csv")
    return rows


# =====================================================================
#  SYNTHESISE TARGET  (matched design for the chosen cross-couplings)
# =====================================================================
# Solves for the main-line couplings, resonator offsets and R1/RN so that the
# passband (f0 +/- BW/2) meets a return-loss level (loss from Qu included) while the
# near-stopband rejection (0.7..1.5 BW either side of f0) is as high as possible.
# Each cross-coupling is either held at a value or solved so that a transmission
# zero lands at a chosen frequency (exact condition: the minor of A that gives
# inv(A)[N-1,0] vanishes there; the lossless form is used for it).
SYNTH_PASS_POINTS = 81
SYNTH_STOP_OFFSETS = (0.7, 1.5, 17)      # (start, stop, count) in units of BW
SYNTH_MIN_RL = 10.0                      # never relax below this return loss (dB)


def _tz_minor(p, om):
    """Real determinant that vanishes at a transmission zero (lossless form)."""
    M = unpack(p)[0]
    return float(np.linalg.det((M + om * np.eye(N))[1:, :-1]))


def synthesize_target(rl_db, specs, seeds=10):
    """specs: one ('value', v) or ('freq', f_hz) per cross-coupling pair.
    Returns (p, info) or (None, info). If rl_db cannot be met, it is relaxed in 1 dB
    steps (loss from Qu limits the best possible return loss)."""
    K = len(CROSS_PAIRS)
    if len(specs) != K:
        raise ValueError("need one specification per cross-coupling")
    p0 = nominal_params()
    fb = np.linspace(F0 - BW / 2, F0 + BW / 2, SYNTH_PASS_POINTS)
    a, b, n = SYNTH_STOP_OFFSETS
    off = np.linspace(a, b, n)
    fs = np.concatenate([F0 - off * BW, F0 + off * BW])
    zeros = [v for kind, v in specs if kind == "freq"]
    zom = [(fz / F0 - F0 / fz) / FBW for fz in zeros]
    nx = 2 * N + 1
    nfree = len(zeros)

    def build(v):
        p = p0.copy()
        p[:N] = v[:N]
        p[N:2 * N - 1] = v[N:2 * N - 1]
        p[IDX_R1], p[IDX_RN] = v[2 * N - 1], v[2 * N]
        c, j = np.zeros(K), nx
        for k, (kind, val) in enumerate(specs):
            if kind == "value":
                c[k] = val
            else:
                c[k] = v[j]
                j += 1
        p[IDX_CROSS] = c
        return p

    bounds = ([(-0.5, 0.5)] * N + [(0.05, 3.0)] * (N - 1) + [(0.3, 3.0)] * 2
              + [(-0.6, 0.6)] * nfree + [(0.0, 1.0)])
    x0 = np.concatenate([p0[:N], p0[N:2 * N - 1], [p0[IDX_R1], p0[IDX_RN]]])
    info = {"relaxed": False, "rl_requested": rl_db}
    rl = float(rl_db)
    while rl >= SYNTH_MIN_RL - 1e-9:
        lim = 10 ** (-rl / 20)

        def ineq(v):
            p = build(v)
            return np.concatenate([lim - np.abs(model(p, fb)[:, 0, 0]),
                                   v[-1] - np.abs(model(p, fs)[:, 1, 0])])

        cons = [{"type": "ineq", "fun": ineq}]
        if zom:
            cons.append({"type": "eq", "fun": lambda v: np.array(
                [_tz_minor(build(v), om) for om in zom])})
        best = None
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            v0 = np.concatenate([x0 + rng.normal(0, 0.02, len(x0)),
                                 [(0.15 if fz > F0 else -0.15) * (1 if seed == 0 else rng.choice([-1, 1]))
                                  for fz in zeros],
                                 [0.1]])
            try:
                r = minimize(lambda v: v[-1], v0, method="SLSQP", bounds=bounds,
                             constraints=cons, options={"maxiter": 500, "ftol": 1e-10})
            except Exception:
                continue
            ok = np.min(ineq(r.x)) > -1e-5
            if zom:
                ok = ok and np.max(np.abs([_tz_minor(build(r.x), om) for om in zom])) < 1e-5
            if ok and (best is None or r.fun < best.fun):
                best = r
        if best is not None:
            p = build(best.x)
            info["rl_used"] = rl
            info["relaxed"] = rl < rl_db - 1e-9
            return p, info
        rl -= 1.0
    return None, info


def find_notches(p, floor_db=-60.0):
    """Frequencies (Hz) of deep S21 notches outside the passband (transmission zeros)."""
    f = np.linspace(F0 - 3 * BW, F0 + 3 * BW, 4001)
    y = np.abs(model(p, f)[:, 1, 0])
    return [f[i] for i in range(1, len(f) - 1)
            if y[i] < y[i - 1] and y[i] < y[i + 1] and abs(f[i] - F0) > 0.6 * BW
            and 20 * np.log10(max(y[i], 1e-12)) < floor_db]


def write_target_file(path, p, label):
    M, r1, rn, delta = unpack(p)
    np.savetxt(path, M, fmt="%.6f",
               header="%s | f0=%g MHz BW=%g MHz\nR1=%.6f RN=%.6f delta=%.6f"
                      % (label, F0 / 1e6, BW / 1e6, r1, rn, delta))


def parse_zero_spec(text: str):
    """'0.15' -> ('value', 0.15);  '580 MHz' or '580' -> ('freq', 580e6)."""
    t = text.strip().lower().replace(" ", "")
    if t.endswith("hz"):
        kind, v = "freq", parse_freq(text)
    else:
        num = parse_number(text)
        kind, v = ("value", num) if abs(num) < 5 else ("freq", parse_freq(text))
    if kind == "freq":
        if v <= 0 or abs(v - F0) < 0.55 * BW:
            raise ValueError("a transmission zero must lie outside the passband "
                             "(f0 +/- %s)" % fmt_freq(0.55 * BW))
    elif abs(v) > 0.6:
        raise ValueError("cross-coupling value must be within +/-0.6")
    return kind, v


def synthesize_interactive():
    """Ask for the specification, synthesise, print and save. Returns the path or None."""
    print("\nSYNTHESISE TARGET  (design: " + design_summary() + ")")
    rl_def = -10 * np.log10(1 - 10 ** (-RIPPLE_DB / 10))
    rl = _ask("Return loss in the passband (dB)", "%.1f, from the ripple" % rl_def, parse_number)
    rl = rl_def if rl is None else rl
    specs = []
    for a, b in CROSS_PAIRS:
        while True:
            raw = input("  Cross-coupling %d-%d: value (e.g. 0.15) or zero frequency "
                        "(e.g. 580 MHz) [0.15]: " % (a + 1, b + 1)).strip()
            if not raw:
                specs.append(("value", 0.15))
                break
            try:
                specs.append(parse_zero_spec(raw))
                break
            except ValueError as exc:
                print("    ! %s" % exc)
    if not CROSS_PAIRS:
        print("No cross-couplings enabled: the target is the plain Chebyshev design.")
    print("Synthesising ...")
    if CROSS_PAIRS:
        p, info = synthesize_target(rl, specs)
    else:
        p, info = nominal_params(), {"relaxed": False, "rl_used": rl}
    if p is None:
        print("Synthesis failed: no design meets even %.0f dB return loss with these "
              "settings. Try a zero farther from the passband or a different pair." % SYNTH_MIN_RL)
        return None
    M, r1, rn, delta = unpack(p)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print("\nSynthesised coupling matrix M:")
    print(M)
    print("R1 = %.4f   RN = %.4f   delta = %.5f (Qu = %.0f)" % (r1, rn, delta, 1 / (delta * FBW)))
    fb = np.linspace(F0 - BW / 2, F0 + BW / 2, 201)
    print("Worst in-band return loss: %.1f dB" % (-db(model(p, fb)[:, 0, 0]).max()))
    if info.get("relaxed"):
        print("NOTE: %.1f dB was not reachable (loss from Qu limits it); %.1f dB is the "
              "best found." % (info["rl_requested"], info["rl_used"]))
    notches = find_notches(p)
    print("Transmission zeros (notches): " + (", ".join("%.1f MHz" % (x / 1e6) for x in notches)
                                              or "none below -60 dB"))
    path = os.path.join(search_dirs()[0], "target_synth_%s.txt" % cross_text().replace(",", "_"))
    write_target_file(path, p, "Synthesised target (cross %s, RL >= %.1f dB)" % (
        cross_text(), info.get("rl_used", rl)))
    print("Saved target file:", path)
    global SYNTH_TARGET, SYNTH_P, SYNTH_KEY
    SYNTH_TARGET, SYNTH_P = path, np.array(p, float)
    SYNTH_KEY = (N, F0, BW, tuple(CROSS_PAIRS))
    return path


# =====================================================================
#  LIVE EDITABLE-MATRIX EDITOR  (real-time response, runs in a WebView)
# =====================================================================
# Same model as model(): tridiagonal A = M + Omega*I - j(R + delta*I).
# Edit any diagonal / adjacent-coupling cell, R1, RN or Qu and the S11/S21
# response (red) updates instantly against the measured data (blue) and the
# target response (grey).

# Auto-tune speed: delay between moves in the live editor, in milliseconds.
# Smaller = faster (e.g. 30), larger = slower so you can watch each move (e.g. 500).
AUTOTUNE_DELAY_MS = 840
EDITOR_PRESENTATION = "fullscreen"   # basic backend only; "sheet" can be swiped down
EDITOR_BACKEND = "uikit"             # "uikit" = native view with a Done button and a page<->Python bridge;
                                     # "basic" = the plain pyto_ui.WebView (used automatically if uikit fails)
EDITOR_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Live coupling-matrix editor</title>
<style>
:root{--bg:#fff;--fg:#14161a;--mut:#69707d;--line:#d5d9e0;--card:#f4f5f8;--acc:#1f6feb;
--ok:#1a7f37;--warn:#b26a00;--bad:#c62828;--in:#fff}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#eceef2;--mut:#98a0ad;--line:#2d323b;
--card:#1a1d23;--acc:#5aa2ff;--ok:#3fb950;--warn:#e3a13a;--bad:#ff6b6b;--in:#0f1115}}
*{box-sizing:border-box}
body{margin:0;padding:10px;padding-top:max(34px,env(safe-area-inset-top));
padding-bottom:max(10px,env(safe-area-inset-bottom));background:var(--bg);color:var(--fg);
font:15px -apple-system,system-ui,sans-serif}
h1{font-size:17px;margin:2px 0 8px}
.card{background:var(--card);border-radius:12px;padding:10px;margin-bottom:10px}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.mut{color:var(--mut)} .small{font-size:12px}
canvas{width:100%;display:block;margin-top:6px}
table.mx{border-collapse:separate;border-spacing:4px;margin:6px 0}
table.mx td{padding:0;text-align:center}
table.mx .hd{color:var(--mut);font-size:12px;width:34px}
input[type=number]{width:84px;padding:9px 4px;font-size:16px;text-align:center;border:1px solid var(--line);
border-radius:8px;background:var(--in);color:var(--fg)}
input.sel{border-color:var(--acc);box-shadow:0 0 0 2px var(--acc)}
.dis{width:84px;padding:9px 4px;color:var(--mut);opacity:.5}
button{font-size:15px;padding:9px 12px;border-radius:9px;border:1px solid var(--line);
background:var(--in);color:var(--fg)}
button.on,button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
button:disabled{opacity:.4}
.seg{display:flex;gap:0}.seg button{border-radius:0}.seg button:first-child{border-radius:9px 0 0 9px}
.seg button:last-child{border-radius:0 9px 9px 0}
table.tt{width:100%;border-collapse:collapse;font-size:13px}
table.tt th{font-weight:600;color:var(--mut);text-align:right;padding:3px 4px}
table.tt td{padding:5px 4px;text-align:right;border-top:1px solid var(--line)}
table.tt td:first-child,table.tt th:first-child,table.tt td:last-child,table.tt th:last-child{text-align:left}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
textarea{width:100%;font:12px ui-monospace,Menlo,monospace;background:var(--in);color:var(--fg);
border:1px solid var(--line);border-radius:8px;padding:6px}
</style></head><body>
<div class="row" style="justify-content:space-between;align-items:center"><h1 id="ttl">Live coupling-matrix editor</h1><button id="b_exit_top">Exit</button></div>
<div class="card" id="exitmsg" style="display:none;position:fixed;left:10px;right:10px;top:48px;z-index:50;border:2px solid var(--acc)"></div>

<div class="card" id="nm"></div>

<div class="card">
 <div class="row" style="justify-content:space-between"><b>Response</b>
  <span class="seg"><button id="vw" class="on">Wide</button><button id="vz">Passband zoom</button></span></div>
 <canvas id="cv"></canvas>
 <div id="ro" class="mut small"></div>
</div>

<div class="card">
 <b>Coupling matrix M</b> <span class="mut small">tap a cell, type a value or nudge (main-line couplings are positive - the sign is a gauge choice; cross-couplings are signed and set which side the transmission zero is on)</span>
 <div id="mx"></div>
 <div class="row" id="ports"></div>
 <div class="row" id="nudge" style="margin-top:8px"></div>
</div>

<div class="card row">
 <button id="b_ext">Reset to extracted</button>
 <button id="b_tgt">Reset to target</button>
 <button id="b_step" class="pri">Apply next move</button>
 <button id="b_auto" class="pri">Auto-tune</button>
 <button id="b_save" class="pri">Save tuned.s2p</button>
 <button id="b_exit">Exit</button>
</div>

<div class="card" id="tcard">
 <b>Live comparison to target</b> <span id="tlabel" class="mut small"></span>
 <div id="tt" style="margin-top:6px"></div>
 <p class="mut small">Ranked by |error| / tolerance (&gt;1 = outside). Model-based electrical advice only:
 the physical adjustment depends on your hardware and one change usually moves several elements.</p>
</div>

<div class="card" id="s2pcard" style="display:none">
 <div class="row" style="justify-content:space-between"><b id="s2ptitle">tuned.s2p</b>
 <button id="b_s2pcopy">Copy</button></div>
 <div id="s2pmsg" class="mut small" style="margin:6px 0"></div>
 <textarea id="s2pbox" rows="8" readonly></textarea>
</div>

<div class="card">
 <div class="row" style="justify-content:space-between"><b>Matrix text</b>
 <button id="b_copy">Copy</button></div>
 <textarea id="tx" rows="10" readonly></textarea>
 <div class="mut small">Save as a .txt and use it as a target or previous-iteration file.</div>
</div>

<script>
const D = __DATA__;
const N = D.N, FBW = D.BW / D.F0, NP = 2 * N + 2;
const IR1 = 2 * N - 1, IRN = 2 * N, IDE = 2 * N + 1;
const NAMES = [];
for (let i = 0; i < N; i++) NAMES.push('M' + (i + 1) + (i + 1));
for (let i = 0; i < N - 1; i++) NAMES.push('M' + (i + 1) + (i + 2));
NAMES.push('R1', 'RN', 'Qu');
const CR = D.cross || [], IX0 = 2 * N + 2;            // cross-coupling pairs (0-based) and first index
CR.forEach(pr => NAMES.push('M' + (pr[0] + 1) + (pr[1] + 1)));

// ==CORE==
// Linear |S11|, |S21| at frequencies fs (Hz) for parameter vector P.
// P = [M11..MNN, M12..M(N-1)N, R1, RN, delta]  (same order as the Python code)
function respond(P, fs) {
  const n = N, c = P.slice(n, 2 * n - 1), r1 = P[2 * n - 1], rn = P[2 * n], de = P[2 * n + 1];
  const s11 = new Float64Array(fs.length), s21 = new Float64Array(fs.length);
  if (CR.length) return respondDense(P, fs, n, r1, rn, de, s11, s21);
  const br = new Float64Array(n), bi = new Float64Array(n);
  const rr = new Float64Array(n), ri = new Float64Array(n);
  const xr = new Float64Array(n), xi = new Float64Array(n);
  for (let k = 0; k < fs.length; k++) {
    const om = (fs[k] / D.F0 - D.F0 / fs[k]) / FBW;
    for (let i = 0; i < n; i++) {
      br[i] = om + P[i];
      bi[i] = -(de + (i === 0 ? r1 : 0) + (i === n - 1 ? rn : 0));
      rr[i] = i === 0 ? 1 : 0; ri[i] = 0;
    }
    for (let i = 1; i < n; i++) {                       // Thomas algorithm (complex)
      const den = br[i - 1] * br[i - 1] + bi[i - 1] * bi[i - 1];
      const wr = c[i - 1] * br[i - 1] / den, wi = -c[i - 1] * bi[i - 1] / den;
      br[i] -= wr * c[i - 1]; bi[i] -= wi * c[i - 1];
      const tr = wr * rr[i - 1] - wi * ri[i - 1], ti = wr * ri[i - 1] + wi * rr[i - 1];
      rr[i] -= tr; ri[i] -= ti;
    }
    {
      const den = br[n - 1] * br[n - 1] + bi[n - 1] * bi[n - 1];
      xr[n - 1] = (rr[n - 1] * br[n - 1] + ri[n - 1] * bi[n - 1]) / den;
      xi[n - 1] = (ri[n - 1] * br[n - 1] - rr[n - 1] * bi[n - 1]) / den;
    }
    for (let i = n - 2; i >= 0; i--) {
      const nr = rr[i] - c[i] * xr[i + 1], ni = ri[i] - c[i] * xi[i + 1];
      const den = br[i] * br[i] + bi[i] * bi[i];
      xr[i] = (nr * br[i] + ni * bi[i]) / den;
      xi[i] = (ni * br[i] - nr * bi[i]) / den;
    }
    s11[k] = Math.hypot(1 - 2 * r1 * xi[0], 2 * r1 * xr[0]);          // 1 + 2j R1 [A^-1]11
    s21[k] = 2 * Math.sqrt(r1 * rn) * Math.hypot(xr[n - 1], xi[n - 1]); // |-2j sqrt(R1 RN) [A^-1]N1|
  }
  return { s11: s11, s21: s21 };
}
// Dense complex solve (Gaussian elimination, partial pivoting): used when cross-couplings exist.
function respondDense(P, fs, n, r1, rn, de, s11, s21) {
  const Ar = new Float64Array(n * n), Ai = new Float64Array(n * n);
  const br = new Float64Array(n), bi = new Float64Array(n);
  const xr = new Float64Array(n), xi = new Float64Array(n);
  for (let k = 0; k < fs.length; k++) {
    const om = (fs[k] / D.F0 - D.F0 / fs[k]) / FBW;
    Ar.fill(0); Ai.fill(0); br.fill(0); bi.fill(0); br[0] = 1;
    for (let i = 0; i < n; i++) {
      Ar[i * n + i] = om + P[i];
      Ai[i * n + i] = -(de + (i === 0 ? r1 : 0) + (i === n - 1 ? rn : 0));
      if (i < n - 1) { Ar[i * n + i + 1] = P[n + i]; Ar[(i + 1) * n + i] = P[n + i]; }
    }
    CR.forEach((pr, m) => { Ar[pr[0] * n + pr[1]] = P[IX0 + m]; Ar[pr[1] * n + pr[0]] = P[IX0 + m]; });
    for (let col = 0; col < n; col++) {
      let piv = col, best = Ar[col * n + col] * Ar[col * n + col] + Ai[col * n + col] * Ai[col * n + col];
      for (let r = col + 1; r < n; r++) {
        const v = Ar[r * n + col] * Ar[r * n + col] + Ai[r * n + col] * Ai[r * n + col];
        if (v > best) { best = v; piv = r; }
      }
      if (piv !== col) {
        for (let j = 0; j < n; j++) {
          let t = Ar[col * n + j]; Ar[col * n + j] = Ar[piv * n + j]; Ar[piv * n + j] = t;
          t = Ai[col * n + j]; Ai[col * n + j] = Ai[piv * n + j]; Ai[piv * n + j] = t;
        }
        let t = br[col]; br[col] = br[piv]; br[piv] = t;
        t = bi[col]; bi[col] = bi[piv]; bi[piv] = t;
      }
      const pr_ = Ar[col * n + col], pi_ = Ai[col * n + col], den = pr_ * pr_ + pi_ * pi_;
      for (let r = col + 1; r < n; r++) {
        const fr = (Ar[r * n + col] * pr_ + Ai[r * n + col] * pi_) / den;
        const fi = (Ai[r * n + col] * pr_ - Ar[r * n + col] * pi_) / den;
        for (let j = col; j < n; j++) {
          Ar[r * n + j] -= fr * Ar[col * n + j] - fi * Ai[col * n + j];
          Ai[r * n + j] -= fr * Ai[col * n + j] + fi * Ar[col * n + j];
        }
        br[r] -= fr * br[col] - fi * bi[col];
        bi[r] -= fr * bi[col] + fi * br[col];
      }
    }
    for (let i = n - 1; i >= 0; i--) {
      let sr = br[i], si = bi[i];
      for (let j = i + 1; j < n; j++) {
        sr -= Ar[i * n + j] * xr[j] - Ai[i * n + j] * xi[j];
        si -= Ar[i * n + j] * xi[j] + Ai[i * n + j] * xr[j];
      }
      const dr = Ar[i * n + i], di = Ai[i * n + i], dd = dr * dr + di * di;
      xr[i] = (sr * dr + si * di) / dd; xi[i] = (si * dr - sr * di) / dd;
    }
    s11[k] = Math.hypot(1 - 2 * r1 * xi[0], 2 * r1 * xr[0]);
    s21[k] = 2 * Math.sqrt(r1 * rn) * Math.hypot(xr[n - 1], xi[n - 1]);
  }
  return { s11: s11, s21: s21 };
}
// ==END CORE==

const dbf = x => 20 * Math.log10(Math.max(x, 1e-6));
const dbc = x => 20 * Math.log10(Math.max(x, 1e-3));          // same -60 dB floor as the Python fit
const lin = v => Math.pow(10, v / 20);
const M11 = D.m11.map(lin), M21 = D.m21.map(lin);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const pass = D.f.map(f => Math.abs(f - D.F0) <= D.BW / 2);

let P = D.p.slice();
const hasTgt = !!D.tgt;
const tgtFull = hasTgt && D.tgt.every(v => v !== null);
const TG = tgtFull ? respond(D.tgt, D.f) : null;
let sel = 0, skip = -1, pending = false, view = 'wide';
const $ = id => document.getElementById(id);
$('ttl').textContent = D.title || 'Live coupling-matrix editor';

// ---------- cost (identical weighting to the Python residuals) ----------
function cost(Q) {
  const r = respond(Q, D.f); let s = 0;
  for (let k = 0; k < D.f.length; k++) {
    const a = r.s11[k] - M11[k], b = r.s21[k] - M21[k];
    const c = (dbc(r.s11[k]) - dbc(M11[k])) / 20, d = (dbc(r.s21[k]) - dbc(M21[k])) / 20;
    s += a * a + b * b + c * c + d * d;
  }
  return 0.5 * s;
}

// ---------- editable inputs ----------
const inputs = {};
const fmt = i => i === IDE ? (1 / (P[IDE] * FBW)).toFixed(1) : P[i].toFixed(4);
function mk(i) {
  const e = document.createElement('input');
  e.type = 'number'; e.step = 'any';
  e.addEventListener('focus', () => select(i));
  e.addEventListener('input', () => {
    const v = parseFloat(e.value);
    if (isFinite(v)) { setParam(i, v); inputs[i].forEach(o => { if (o !== e) o.value = fmt(i); }); schedule(i); }
  });
  (inputs[i] = inputs[i] || []).push(e); return e;
}
function setParam(i, v) {
  if (i === IDE) { if (v < 5) return; v = 1 / (v * FBW); }
  P[i] = clamp(v, D.lo[i], D.hi[i]);
}
(function build() {
  const t = document.createElement('table'); t.className = 'mx';
  let h = document.createElement('tr'); h.appendChild(document.createElement('td'));
  for (let j = 0; j < N; j++) { const c = document.createElement('td'); c.className = 'hd'; c.textContent = j + 1; h.appendChild(c); }
  t.appendChild(h);
  for (let i = 0; i < N; i++) {
    const tr = document.createElement('tr');
    const hd = document.createElement('td'); hd.className = 'hd'; hd.textContent = i + 1; tr.appendChild(hd);
    for (let j = 0; j < N; j++) {
      const td = document.createElement('td');
      if (i === j) td.appendChild(mk(i));
      else if (Math.abs(i - j) === 1) td.appendChild(mk(N + Math.min(i, j)));
      else if (crossIdx(i, j) >= 0) td.appendChild(mk(crossIdx(i, j)));
      else { const d = document.createElement('div'); d.className = 'dis'; d.textContent = '0'; td.appendChild(d); }
      tr.appendChild(td);
    }
    t.appendChild(tr);
  }
  $('mx').appendChild(t);
  [[IR1, 'R1 (input)'], [IRN, 'RN (output)'], [IDE, 'Qu']].forEach(([i, l]) => {
    const w = document.createElement('label'); w.className = 'row';
    w.append(l + ' '); w.appendChild(mk(i)); $('ports').appendChild(w);
  });
})();
function crossIdx(i, j) {
  const a = Math.min(i, j), b = Math.max(i, j), m = CR.findIndex(pr => pr[0] === a && pr[1] === b);
  return m < 0 ? -1 : IX0 + m;
}
function refresh() { for (const k in inputs) if (+k !== skip) inputs[k].forEach(e => { e.value = fmt(+k); }); }
function select(i) {
  sel = i;
  for (const k in inputs) inputs[k].forEach(e => e.classList.toggle('sel', +k === i));
  const nb = $('nudge'); nb.innerHTML = '';
  const steps = i === IDE ? [-50, -5, 5, 50] : [-0.01, -0.001, 0.001, 0.01];
  const lab = document.createElement('span'); lab.className = 'mut small'; lab.textContent = NAMES[i] + ':'; nb.appendChild(lab);
  steps.forEach(s => {
    const b = document.createElement('button'); b.textContent = (s > 0 ? '+' : '\u2212') + Math.abs(s);
    b.onclick = () => { const cur = i === IDE ? 1 / (P[IDE] * FBW) : P[i]; setParam(i, cur + s); skip = -1; refresh(); schedule(-1); };
    nb.appendChild(b);
  });
}

// ---------- plotting ----------
function ticks(a, b, n) {
  const raw = (b - a) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / mag;
  const st = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(v);
  return out;
}
function panel(ctx, x0, y0, w, h, title, xr, yr, series) {
  const cs = getComputedStyle(document.documentElement);
  const mut = cs.getPropertyValue('--mut'), line = cs.getPropertyValue('--line'), fg = cs.getPropertyValue('--fg');
  const L = 44, R = 8, T = 18, B = 22, px = x0 + L, py = y0 + T, pw = w - L - R, ph = h - T - B;
  const X = v => px + (v - xr[0]) / (xr[1] - xr[0]) * pw, Y = v => py + (yr[1] - v) / (yr[1] - yr[0]) * ph;
  ctx.save(); ctx.font = '11px -apple-system,sans-serif'; ctx.lineWidth = 1;
  ctx.fillStyle = fg; ctx.textAlign = 'left'; ctx.fillText(title, px, y0 + 12);
  ctx.strokeStyle = line; ctx.fillStyle = mut;
  ticks(yr[0], yr[1], 5).forEach(v => {
    ctx.beginPath(); ctx.moveTo(px, Y(v)); ctx.lineTo(px + pw, Y(v)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(v.toFixed(0), px - 4, Y(v) + 4);
  });
  ticks(xr[0], xr[1], 6).forEach(v => {
    ctx.beginPath(); ctx.moveTo(X(v), py); ctx.lineTo(X(v), py + ph); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText(v.toFixed(0), X(v), py + ph + 14);
  });
  ctx.strokeRect(px, py, pw, ph);
  ctx.beginPath(); ctx.rect(px, py, pw, ph); ctx.clip();
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.w; ctx.setLineDash(s.dash); ctx.beginPath();
    for (let k = 0; k < D.f.length; k++) {
      const x = X(D.f[k] / 1e6), y = Y(clamp(s.y[k], yr[0] - 5, yr[1] + 5));
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  });
  ctx.restore();
}
function draw(s11, s21) {
  const cv = $('cv'), ctx = cv.getContext('2d'), dpr = window.devicePixelRatio || 1;
  const w = cv.parentElement.clientWidth - 20, h = Math.round(clamp(w * 0.95, 360, 560));
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.style.width = w + 'px'; cv.style.height = h + 'px'; cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const zoom = view === 'zoom';
  const xr = zoom ? [(D.F0 - 0.75 * D.BW) / 1e6, (D.F0 + 0.75 * D.BW) / 1e6] : [D.f[0] / 1e6, D.f[D.f.length - 1] / 1e6];
  const mk2 = (y, color, w2, dash) => ({ y: y, color: color, w: w2, dash: dash });
  const set = (me, mo, tg) => {
    const a = [mk2(me, '#1f6feb', 2.2, [])];
    if (tg) a.push(mk2(tg, '#8a8f98', 1.5, [2, 3]));
    a.push(mk2(mo, '#e5484d', 1.6, [6, 3])); return a;
  };
  const m21 = D.m21, m11 = D.m11;
  panel(ctx, 0, 0, w, h / 2, 'S21 (dB)   blue: measured   red: model   grey: target', xr, zoom ? [-4, 0] : [-120, 0],
    set(m21, s21, TG ? Array.from(TG.s21, dbf) : null));
  panel(ctx, 0, h / 2, w, h / 2, 'S11 (dB)', xr, [-40, 0], set(m11, s11, TG ? Array.from(TG.s11, dbf) : null));
}

// ---------- live comparison to target (same rules as the Python advisor) ----------
const TOL = { diag: 0.02, cpl: 0.05, cplFloor: 0.02, zero: 1e-3, load: 0.05, loss: 0.2, cross: 0.02 };
function rows() {
  if (!hasTgt) return [];
  const out = [];
  const add = (name, t, e, err, score, act, kind, i) => out.push({ name, t, e, err, score, act, kind, i });
  for (let i = 0; i < N; i++) {
    const t = D.tgt[i]; if (t === null) continue;
    const err = P[i] - t, sc = Math.abs(err) / TOL.diag, df = Math.abs(err) * D.BW / 2e6;
    add(NAMES[i], t, P[i], err, sc, sc <= 1 ? 'ok' : (err > 0 ? 'raise res.' + (i + 1) + ' freq ~' + df.toFixed(2) + ' MHz' : 'lower res.' + (i + 1) + ' freq ~' + df.toFixed(2) + ' MHz'), 'resonator', i);
  }
  for (let k = 0; k < N - 1; k++) {
    const t = D.tgt[N + k]; if (t === null) continue;
    const e = P[N + k], err = Math.abs(e) - Math.abs(t);
    const sc = Math.abs(t) >= TOL.zero ? Math.abs(err) / (TOL.cpl * Math.abs(t)) : Math.abs(err) / TOL.cplFloor;
    add(NAMES[N + k], t, e, err, sc, sc <= 1 ? 'ok' : (err > 0 ? 'reduce coupling' : 'increase coupling'), 'coupling', N + k);
  }
  [[IR1, 'input'], [IRN, 'output']].forEach(([i, s]) => {
    const t = D.tgt[i]; if (t === null) return;
    const err = P[i] - t, sc = Math.abs(err) / (TOL.load * Math.abs(t));
    add(NAMES[i], t, P[i], err, sc, sc <= 1 ? 'ok' : (err > 0 ? 'reduce ' + s + ' coupling' : 'increase ' + s + ' coupling'), 'port', i);
  });
  CR.forEach((pr, m) => {
    const i = IX0 + m, t = D.tgt[i]; if (t === null) return;
    const err = P[i] - t, sc = Math.abs(err) / Math.max(TOL.cpl * Math.abs(t), TOL.cross);
    add(NAMES[i], t, P[i], err, sc, sc <= 1 ? 'ok' : (err > 0 ? 'decrease cross-coupling' : 'increase cross-coupling'), 'cross', i);
  });
  const td = D.tgt[IDE];
  if (td !== null) {
    const err = P[IDE] - td, sc = Math.abs(err) / (TOL.loss * Math.max(Math.abs(td), 1e-12));
    add('Qu', 1 / (td * FBW), 1 / (P[IDE] * FBW), err, sc, sc <= 1 ? 'ok' : (err > 0 ? 'excess loss: investigate' : 'less loss than target'), 'loss', IDE);
  }
  return out.sort((a, b) => b.score - a.score);
}

// ---------- next best tuning move (same logic as the Python advisor) ----------
const HALF = 0.5, PC = Math.round(100 * HALF);
function describe(r) {
  if (r.kind === 'resonator') {
    const amt = Math.abs(r.err) * D.BW / 2e6;
    return (r.err > 0 ? 'Raise' : 'Lower') + ' resonator ' + (r.i + 1) + ' frequency by about ' + (amt * HALF).toFixed(2) +
      ' MHz (' + PC + '% of the ~' + amt.toFixed(2) + ' MHz model-estimated error)';
  }
  if (r.kind === 'coupling') {
    const k = r.i - N, verb = r.err > 0 ? 'Reduce' : 'Increase', pair = (k + 1) + ' and ' + (k + 2);
    if (Math.abs(r.t) >= TOL.zero) {
      const pct = 100 * Math.abs(r.err) / Math.abs(r.t);
      return verb + ' the coupling between resonators ' + pair + ' by about ' + (pct * HALF).toFixed(1) + '% (' + PC + '% of the ' + pct.toFixed(1) + '% error)';
    }
    return verb + ' the coupling between resonators ' + pair + ' by about ' + (Math.abs(r.err) * HALF).toFixed(3) + ' in M units';
  }
  if (r.kind === 'port') {
    const pct = 100 * Math.abs(r.err) / Math.abs(r.t);
    return (r.err > 0 ? 'Reduce' : 'Increase') + ' the ' + (r.i === IR1 ? 'input' : 'output') + ' coupling by about ' + (pct * HALF).toFixed(1) + '% (' + PC + '% of the ' + pct.toFixed(1) + '% error)';
  }
  if (r.kind === 'cross') {
    const pr = CR[r.i - IX0];
    return (r.err > 0 ? 'Decrease' : 'Increase') + ' the cross-coupling between resonators ' + (pr[0] + 1) + ' and ' + (pr[1] + 1) +
      ' by about ' + (Math.abs(r.err) * HALF).toFixed(3) + ' in M units (' + PC + '% of the error; it moves the transmission zero)';
  }
  if (r.kind === 'loss') return (r.err > 0 ? 'Investigate excess internal loss' : 'Loss is below target - no loss action') +
    ' (Qu ' + r.e.toFixed(0) + ' vs target ' + r.t.toFixed(0) + '); model-based, cause unproven';
  return r.act;
}
function nextMove(rs) {
  if (!hasTgt || !rs.length) return null;
  const bad = rs.filter(r => r.score > 1);
  const out = bad.length ? bad : rs;          // always give a move: largest remaining error
  const cand = out.find(r => r.kind !== 'loss') || out[0];
  const more = ['Why: ' + cand.name.split(' ')[0] + ' has the largest normalised error among adjustable parameters (' + cand.score.toFixed(3) + 'x tolerance).'];
  const rest = out.filter(r => r !== cand).slice(0, 3);
  if (rest.length) more.push('Then, in order: ' + rest.map(r => r.name.split(' ')[0] + ' (' + r.score.toFixed(3) + 'x)').join(', '));
  if (cand.kind !== 'resonator' && bad.some(r => r.kind === 'resonator')) more.push('Note: resonators are also out of tolerance - tune those first; coupling estimates are less reliable while they are detuned.');
  more.push('Make this ONE change, re-measure and re-extract before the next; other parameters will shift too.');
  more.push('Model-based electrical direction only - not a mechanical instruction.');
  return { cls: 'warn', head: describe(cand), more: more };
}
function renderNextMove(rs) {
  if (!hasTgt) {
    $('nm').style.display = '';
    $('nm').innerHTML = '<b>No target loaded</b><div class="mut small">Auto-tune and Apply next move need a target. Restart and give a target file at the target prompt (Enter = nominal design).</div>';
    return;
  }
  const m = nextMove(rs);
  $('nm').style.display = m ? '' : 'none';
  if (!m) { $('nm').innerHTML = ''; return; }
  $('nm').innerHTML = '<b>Next best move</b><div class="' + m.cls + '" style="font-size:16px;margin:6px 0"><b>' + m.head + '</b></div>' +
    m.more.map(t => '<div class="mut small">' + t + '</div>').join('');
}
function renderTable() {
  if (!hasTgt) { $('tt').innerHTML = '<span class="mut">No target supplied.</span>'; return; }
  const rs = rows();
  let h = '<table class="tt"><tr><th>Param</th><th>Target</th><th>Now</th><th>\u00d7 tol</th><th>Action</th></tr>';
  rs.forEach(r => {
    const cl = r.score <= 1 ? 'ok' : 'warn';
    const d = r.name === 'Qu' ? 1 : 4;
    h += '<tr><td>' + r.name + '</td><td>' + r.t.toFixed(d) + '</td><td>' + r.e.toFixed(d) + '</td><td class="' + cl + '">' +
      r.score.toFixed(3) + '</td><td>' + (r.act === 'ok' ? '\u2713' : r.act) + '</td></tr>';
  });
  $('tt').innerHTML = h + '</table>';
}
function exportText() {
  let s = '# Live-edited ' + N + 'x' + N + ' inline coupling matrix\n# f0=' + (D.F0 / 1e6) + ' MHz BW=' + (D.BW / 1e6) + ' MHz\n';
  s += '# R1=' + P[IR1].toFixed(6) + ' RN=' + P[IRN].toFixed(6) + ' delta=' + P[IDE].toFixed(6) +
    ' Qu=' + (1 / (P[IDE] * FBW)).toFixed(2) + '\n';
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < N; j++) row.push((i === j ? P[i] : Math.abs(i - j) === 1 ? P[N + Math.min(i, j)] : crossIdx(i, j) >= 0 ? P[crossIdx(i, j)] : 0).toFixed(6));
    s += row.join(' ') + '\n';
  }
  $('tx').value = s;
}

// ---------- main update loop ----------
function update() {
  const r = respond(P, D.f);
  const s21 = Array.from(r.s21, dbf), s11 = Array.from(r.s11, dbf);
  draw(s11, s21);
  let a = 0, b = 0, n = 0;
  for (let k = 0; k < D.f.length; k++) if (pass[k]) { a += Math.pow(s21[k] - D.m21[k], 2); b += Math.pow(r.s11[k] - M11[k], 2); n++; }
  const q = n ? 'passband rms: S21 ' + Math.sqrt(a / n).toFixed(3) + ' dB, |S11| ' + Math.sqrt(b / n).toFixed(4) +
    '   fit cost ' + cost(P).toExponential(2) : 'no points inside passband';
  $('ro').textContent = q + '   (Qu ' + (1 / (P[IDE] * FBW)).toFixed(0) + ')';
  renderNextMove(rows());
  renderTable(); exportText();
}
function schedule(s) {
  skip = s;
  if (pending) return; pending = true;
  requestAnimationFrame(() => { pending = false; update(); skip = -1; });
}

// ---------- apply the next best move, one parameter at a time ----------
let autoTimer = null;
function applyNextMove() {
  if (!hasTgt) return false;
  const rs = rows();
  if (!rs.length || rs[0].score < 0.0005) return false;      // every x tol reads 0.000
  const bad = rs.filter(r => r.score > 1);
  const pool = bad.length ? bad : rs;
  const cand = pool.find(r => r.kind !== 'loss') || pool[0]; // same pick as the Next best move card
  const frac = cand.score <= 1 ? 1.0 : HALF;                 // inside tolerance: finish exactly
  const i = cand.i;
  P[i] = clamp(P[i] - frac * cand.err, D.lo[i], D.hi[i]);
  skip = -1; refresh(); update();
  return true;
}
function stopAuto() {
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  $('b_auto').textContent = 'Auto-tune';
}

// ---------- buttons ----------
$('b_ext').onclick = () => { stopAuto(); P = D.p.slice(); skip = -1; refresh(); update(); };
$('b_tgt').onclick = () => { if (hasTgt) { stopAuto(); P = D.tgt.map((v, i) => v === null ? P[i] : (i >= N && i < 2 * N - 1) ? Math.abs(v) : v); skip = -1; refresh(); update(); } };
$('b_tgt').disabled = !hasTgt;
$('b_step').onclick = () => { stopAuto(); applyNextMove(); };
$('b_auto').onclick = () => {
  if (autoTimer) { stopAuto(); return; }
  $('b_auto').textContent = 'Stop';
  let n = 0;
  autoTimer = setInterval(() => { if (!applyNextMove() || ++n > 500) stopAuto(); }, D.autoMs);
};
$('b_step').disabled = $('b_auto').disabled = !hasTgt;
$('b_copy').onclick = () => { const t = $('tx'); t.removeAttribute('readonly'); t.select(); try { document.execCommand('copy'); } catch (e) {} t.setAttribute('readonly', ''); };
$('vw').onclick = () => { view = 'wide'; $('vw').className = 'on'; $('vz').className = ''; update(); };
$('vz').onclick = () => { view = 'zoom'; $('vz').className = 'on'; $('vw').className = ''; update(); };
const nMiss = hasTgt ? D.tgt.filter(v => v === null).length : 0;
$('tlabel').textContent = hasTgt ? '(' + D.tgtLabel + (nMiss ? ' - partial: ' + nMiss + ' value(s) missing, those are skipped' : '') + ')' : '';
$('tcard').style.display = hasTgt ? '' : 'none';
window.addEventListener('resize', () => update());
// ---------- tuned.s2p export: complex S-parameters of the live-edited matrix ----------
function gaussSolve(Ar, Ai, br, bi, n, xr, xi) {                // complex, partial pivoting; A destroyed
  for (let col = 0; col < n; col++) {
    let piv = col, best = Ar[col * n + col] * Ar[col * n + col] + Ai[col * n + col] * Ai[col * n + col];
    for (let r = col + 1; r < n; r++) {
      const v = Ar[r * n + col] * Ar[r * n + col] + Ai[r * n + col] * Ai[r * n + col];
      if (v > best) { best = v; piv = r; }
    }
    if (piv !== col) {
      for (let j = 0; j < n; j++) {
        let t = Ar[col * n + j]; Ar[col * n + j] = Ar[piv * n + j]; Ar[piv * n + j] = t;
        t = Ai[col * n + j]; Ai[col * n + j] = Ai[piv * n + j]; Ai[piv * n + j] = t;
      }
      let t = br[col]; br[col] = br[piv]; br[piv] = t;
      t = bi[col]; bi[col] = bi[piv]; bi[piv] = t;
    }
    const pr_ = Ar[col * n + col], pi_ = Ai[col * n + col], den = pr_ * pr_ + pi_ * pi_;
    for (let r = col + 1; r < n; r++) {
      const fr = (Ar[r * n + col] * pr_ + Ai[r * n + col] * pi_) / den;
      const fi = (Ai[r * n + col] * pr_ - Ar[r * n + col] * pi_) / den;
      for (let j = col; j < n; j++) {
        Ar[r * n + j] -= fr * Ar[col * n + j] - fi * Ai[col * n + j];
        Ai[r * n + j] -= fr * Ai[col * n + j] + fi * Ar[col * n + j];
      }
      br[r] -= fr * br[col] - fi * bi[col];
      bi[r] -= fr * bi[col] + fi * br[col];
    }
  }
  for (let i = n - 1; i >= 0; i--) {
    let sr = br[i], si = bi[i];
    for (let j = i + 1; j < n; j++) {
      sr -= Ar[i * n + j] * xr[j] - Ai[i * n + j] * xi[j];
      si -= Ar[i * n + j] * xi[j] + Ai[i * n + j] * xr[j];
    }
    const dr = Ar[i * n + i], di = Ai[i * n + i], dd = dr * dr + di * di;
    xr[i] = (sr * dr + si * di) / dd; xi[i] = (si * dr - sr * di) / dd;
  }
}
function buildS2P() {
  const n = N, r1 = P[2 * n - 1], rn = P[2 * n], de = P[2 * n + 1], pl = D.planes || [0, 0, 0, 1];
  const t1 = pl[0], t2 = pl[1], phi = pl[2], refl = pl[3];
  const Ar = new Float64Array(n * n), Ai = new Float64Array(n * n), br = new Float64Array(n), bi = new Float64Array(n);
  const xr = new Float64Array(n), xi = new Float64Array(n);
  const solve = (om, rhs) => {
    Ar.fill(0); Ai.fill(0); br.fill(0); bi.fill(0); br[rhs] = 1;
    for (let i = 0; i < n; i++) {
      Ar[i * n + i] = om + P[i];
      Ai[i * n + i] = -(de + (i === 0 ? r1 : 0) + (i === n - 1 ? rn : 0));
      if (i < n - 1) { Ar[i * n + i + 1] = P[n + i]; Ar[(i + 1) * n + i] = P[n + i]; }
    }
    CR.forEach((pr, m) => { Ar[pr[0] * n + pr[1]] = P[IX0 + m]; Ar[pr[1] * n + pr[0]] = P[IX0 + m]; });
    gaussSolve(Ar, Ai, br, bi, n, xr, xi);
  };
  const num = v => String(Number(v.toPrecision(9)));
  const rot = (re, im, ang) => [re * Math.cos(ang) - im * Math.sin(ang), re * Math.sin(ang) + im * Math.cos(ang)];
  const out = [];
  out.push('! Tuned response of the live-edited matrix (model; ' + (D.planes ? 'fitted reference planes applied' : 'no reference planes')
    + '), N=' + N + (CR.length ? ', cross ' + CR.map(pr => (pr[0] + 1) + '-' + (pr[1] + 1)).join(',') : ''));
  out.push('! R1=' + P[IR1].toFixed(6) + ' RN=' + P[IRN].toFixed(6) + ' Qu=' + (1 / (P[IDE] * FBW)).toFixed(2));
  out.push('# MHZ S RI R ' + (D.z0 || 50));
  for (let k = 0; k < D.f.length; k++) {
    const f = D.f[k], om = (f / D.F0 - D.F0 / f) / FBW;
    solve(om, 0);
    const x0r = xr[0], x0i = xi[0], xNr = xr[n - 1], xNi = xi[n - 1];
    solve(om, n - 1);
    const yr = xr[n - 1], yi = xi[n - 1], g = 2 * Math.sqrt(r1 * rn);
    let s11 = [1 - 2 * r1 * x0i, 2 * r1 * x0r];
    let s22 = [1 - 2 * rn * yi, 2 * rn * yr];
    let s21 = [g * xNi, -g * xNr];
    s11 = rot(refl * s11[0], refl * s11[1], -4 * Math.PI * f * t1);
    s22 = rot(refl * s22[0], refl * s22[1], -4 * Math.PI * f * t2);
    s21 = rot(s21[0], s21[1], -2 * Math.PI * f * (t1 + t2) + phi);
    out.push([f / 1e6, s11[0], s11[1], s21[0], s21[1], s21[0], s21[1], s22[0], s22[1]].map((v, i) => i ? num(v) : String(Number((v).toPrecision(12)))).join(' '));
  }
  return out.join('\n') + '\n';
}
async function saveTuned() {
  stopAuto();
  const text = buildS2P(), name = D.saveName || 'tuned.s2p', msg = $('s2pmsg'), notes = [];
  const show = () => { msg.innerHTML = notes.join('<br>'); };
  $('s2pcard').style.display = ''; $('s2ptitle').textContent = name; $('s2pbox').value = text;
  $('s2pcard').scrollIntoView();
  window.__pytoSaveText = text; window.__pytoSavedTo = ''; window.__pytoSaveReq = true;   // Pyto writes it next to the script
  nativePost('save:' + name + '\n' + text);                                                // native bridge
  notes.push('Asking Pyto to write ' + name + ' next to the script...'); show();
  try {
    const file = new File([text], name, { type: 'text/plain' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: name });
      notes.push('Share sheet: choose Save to Files.');
    } else { notes.push('No share sheet here: use Copy below.'); }
  } catch (e) { notes.push(e && e.name === 'AbortError' ? 'Share sheet closed.' : 'Share sheet unavailable: use Copy below.'); }
  show();
  setTimeout(() => {
    window.__pytoSaveReq = false;
    notes[0] = window.__pytoSavedTo ? 'Saved by Pyto: ' + window.__pytoSavedTo
      : 'Pyto did not confirm writing the file - use the share sheet or Copy.';
    show();
  }, 1500);
}
$('b_save').onclick = saveTuned;
$('b_s2pcopy').onclick = () => { const t = $('s2pbox'); t.removeAttribute('readonly'); t.select(); try { document.execCommand('copy'); } catch (e) {} t.setAttribute('readonly', ''); };
$('b_save').disabled = false;

function nativePost(msg) {                        // native bridge (Pyto UIKit backend), if present
  try { window.webkit.messageHandlers.pyto.postMessage(msg); return true; } catch (e) { return false; }
}
function doExit() {
  stopAuto();
  const m = $('exitmsg'); m.style.display = '';
  m.innerHTML = '<b>Exit pressed</b><div class="mut small">Asking Pyto to close the editor...</div>';
  nativePost('exit');
  window.__pytoExit = true;                       // polled by Pyto (evaluate_js)
  try { if (window.webkit) location.href = 'pytoexit://close'; else window.close(); } catch (e) {}   // caught by on_redirect
  setTimeout(() => {
    window.__pytoExit = false;
    m.innerHTML = '<b>Still open?</b><div class="mut small">Pyto did not close this page. Use the Done button at the top right if it is shown, swipe down if it is a sheet, or press Stop in the Pyto console (your results are already saved).</div>';
  }, 2500);
}
$('b_exit').onclick = $('b_exit_top').onclick = doExit;
select(0); refresh(); update();
</script></body></html>
"""


def editor_template() -> str:
    """The live-editor page (embedded below; __DATA__ is replaced with the measured data)."""
    return EDITOR_HTML


def build_editor_html(f, S, p, p_tgt=None, tgt_label="", title="", planes=None, z0=Z0,
                      save_name=""):
    """Embed measured data + extracted parameters into the editor page."""
    f = np.asarray(f, float)
    if len(f) > 801:                                    # keep the page light
        idx = np.linspace(0, len(f) - 1, 801).astype(int)
        f, S = f[idx], S[idx]

    def clean(a, nd=6):
        return [None if not np.isfinite(v) else round(float(v), nd)
                for v in np.asarray(a, float).ravel()]

    def dbm(x):
        return 20 * np.log10(np.maximum(np.abs(x), 1e-6))

    lo, hi = param_bounds()
    data = {
        "N": N, "F0": F0, "BW": BW, "title": title,
        "f": clean(f, 3),
        "m11": clean(dbm(S[:, 0, 0]), 4), "m21": clean(dbm(S[:, 1, 0]), 4),
        "p": clean(p, 8), "lo": clean(lo), "hi": clean(hi),
        "tgt": None if p_tgt is None else clean(p_tgt, 8),
        "tgtLabel": tgt_label,
        "autoMs": int(max(10, AUTOTUNE_DELAY_MS)),
        "cross": [[a, b] for a, b in CROSS_PAIRS],
        "planes": None if planes is None else [float(x) for x in planes],   # tau1, tau2, phi, refl
        "z0": float(z0), "saveName": save_name,
    }
    return editor_template().replace("__DATA__", json.dumps(data))


def _launch_uikit(html: str, save_path: Optional[str]) -> None:
    """Show the page in a native UIKit view controller (the pattern in Pyto's pyto_ui docs):
    a WKWebView under a navigation bar with a Done button, plus a WKScriptMessageHandler so
    the page's Exit and Save buttons call Python directly. Blocks until the view is closed.
    Raises before presenting if anything is unavailable (the caller then falls back)."""
    import threading
    import time
    import types
    from UIKit import UIViewController, UINavigationController, UIBarButtonItem, UIColor
    from WebKit import WKWebView, WKWebViewConfiguration
    from Foundation import NSObject
    from rubicon.objc import ObjCInstance, ObjCProtocol, objc_method, SEL
    from mainthread import mainthread
    import pyto_ui as ui

    state = {"closed": threading.Event(), "vc": None, "web": None, "error": None}

    def finish():
        state["closed"].set()
        try:
            state["vc"].dismissViewControllerAnimated(True, completion=None)
        except Exception as exc:
            print("[editor] dismiss failed: %s" % exc)

    def on_message(body):
        if body == "exit":
            print("[editor] closing the editor (page Exit button)")
            finish()
        elif body.startswith("save:") and save_path:
            _name, _, text = body[5:].partition("\n")
            try:
                with open(save_path, "w") as fh:
                    fh.write(text if text.endswith("\n") else text + "\n")
                print("[editor] saved tuned file: %s" % save_path)
                state["web"].evaluateJavaScript(
                    "window.__pytoSavedTo = %s;" % json.dumps(save_path), completionHandler=None)
            except Exception as exc:
                print("[editor] could not write %s: %s" % (save_path, exc))

    def close(self):                                        # the Done bar button
        print("[editor] closing the editor (Done button)")
        finish()

    def userContentController_didReceiveScriptMessage_(self, controller: ObjCInstance,
                                                       message: ObjCInstance) -> None:
        try:
            on_message(str(message.body))
        except Exception as exc:
            print("[editor] message handler error: %s" % exc)

    methods = {"close": objc_method(close),
               "userContentController_didReceiveScriptMessage_":
                   objc_method(userContentController_didReceiveScriptMessage_)}
    stamp = int(time.time() * 1000) % 10 ** 9              # ObjC class names must be unique
    handler_cls, last = None, None
    for k, kw in enumerate(({"protocols": [ObjCProtocol("WKScriptMessageHandler")]}, {})):
        try:
            handler_cls = types.new_class("FilterEditorHandler%d_%d" % (stamp, k), (NSObject,), kw,
                                          lambda ns: ns.update(methods))
            break
        except Exception as exc:
            last = exc
    if handler_cls is None:
        raise last
    handler = handler_cls.alloc().init()
    state["handler"] = handler

    @mainthread
    def present():
        try:
            config = WKWebViewConfiguration.alloc().init()
            config.userContentController.addScriptMessageHandler(handler, name="pyto")
            vc = UIViewController.alloc().init()
            vc.title = "Live coupling-matrix editor"
            vc.view.backgroundColor = UIColor.systemBackgroundColor()
            web = WKWebView.alloc().initWithFrame(vc.view.bounds, configuration=config)
            web.autoresizingMask = 18                       # flexible width + height
            vc.view.addSubview(web)
            web.loadHTMLString(html, baseURL=None)
            done = UIBarButtonItem.alloc().initWithBarButtonSystemItem(
                0, target=handler, action=SEL("close"))     # 0 = Done
            vc.navigationItem.rightBarButtonItems = [done]
            state["vc"], state["web"] = vc, web
            nav = UINavigationController.alloc().initWithRootViewController(vc)
            ui.show_view_controller(nav)
        except Exception as exc:
            state["error"] = exc

    present()
    t0, seen = time.time(), False
    while not state["closed"].is_set():
        time.sleep(0.3)
        if state["error"] is not None:
            raise state["error"]
        try:
            on_screen = state["vc"].view.window is not None
        except Exception:
            on_screen = seen
        if on_screen:
            seen = True
        elif seen:
            break                                           # dismissed by a swipe
        elif time.time() - t0 > 8:
            raise RuntimeError("the editor view never appeared")
    print("[editor] closed")


def launch_live_editor(html: str, html_path: str, save_path: Optional[str] = None) -> bool:
    """Save the page and show it in a Pyto WebView (falls back to the file).

    The page's Exit button asks Python to close the view in several independent ways
    (one working path is enough); every step is printed so the console shows which
    one worked:
      1. the page navigates to pytoexit://close -> WebView.on_redirect / on_fail_loading
      2. the page sets window.__pytoExit -> a background thread polls it with evaluate_js
    The page's "Save tuned.s2p" button also sets window.__pytoSaveReq and window.__pytoSaveText;
    the same thread then writes the text to save_path and reports back in
    window.__pytoSavedTo (the page also offers the share sheet and Copy as backups)."""
    with open(html_path, "w") as fh:
        fh.write(html)
    if EDITOR_BACKEND == "uikit":
        try:
            _launch_uikit(html, save_path)
            return True
        except Exception as exc:
            print("[editor] native view unavailable (%s: %s); using the basic page."
                  % (type(exc).__name__, exc))
    try:
        import pyto_ui as ui                            # Pyto only
        import threading
        import time
        web = ui.WebView()
        state = {"done": False, "warned": False}
        print("[editor] WebView support: " + ", ".join(
            "%s=%s" % (n, hasattr(web, n)) for n in
            ("close", "evaluate_js", "on_redirect", "on_fail_loading", "load_html")))

        def close_view(why):
            print("[editor] closing the editor (%s)" % why)
            try:
                from mainthread import mainthread       # UI calls belong on the main thread
                mainthread(web.close)()
            except Exception:
                try:
                    web.close()
                except Exception as exc:
                    print("[editor] close() failed: %s" % exc)

        def on_redirect(*args):                         # (web_view, url) -> allow navigation?
            if "pytoexit:" in " ".join(str(a) for a in args):
                close_view("on_redirect")
                return False
            return True

        def on_fail(*args):                             # a custom-scheme navigation may land here
            if "pytoexit" in " ".join(str(a) for a in args):
                close_view("on_fail_loading")

        def js_eval(code):
            try:
                return web.evaluate_js(code)
            except Exception as exc1:
                try:
                    from mainthread import mainthread
                    return mainthread(web.evaluate_js)(code)
                except Exception:
                    if not state["warned"]:
                        state["warned"] = True
                        print("[editor] evaluate_js failed: %s" % exc1)
                    return None

        def save_tuned():
            text = js_eval("window.__pytoSaveText")
            if isinstance(text, str) and text[:1] == '"':
                try:
                    text = json.loads(text)
                except ValueError:
                    pass
            if not (isinstance(text, str) and "# MHZ" in text):
                print("[editor] save request: no text received from the page")
                js_eval("window.__pytoSaveReq = false; ''")
                return
            try:
                with open(save_path, "w") as fh:
                    fh.write(text if text.endswith("\n") else text + "\n")
                print("[editor] saved tuned file: %s" % save_path)
                js_eval("window.__pytoSaveReq = false; window.__pytoSavedTo = %s; ''"
                        % json.dumps(save_path))
            except OSError as exc:
                print("[editor] could not write %s: %s" % (save_path, exc))
                js_eval("window.__pytoSaveReq = false; ''")

        def poll_exit():
            while not state["done"]:
                time.sleep(0.4)
                r = js_eval("window.__pytoExit === true ? 'EXIT' : "
                            "(window.__pytoSaveReq === true ? 'SAVE' : '')")
                if r and "EXIT" in str(r):
                    close_view("evaluate_js poll")
                    return
                if r and "SAVE" in str(r) and save_path:
                    save_tuned()

        web.on_redirect = on_redirect
        web.on_fail_loading = on_fail
        threading.Thread(target=poll_exit, daemon=True).start()
        web.load_html(html)
        mode = (ui.PRESENTATION_MODE_SHEET if EDITOR_PRESENTATION == "sheet"
                else ui.PRESENTATION_MODE_FULLSCREEN)
        ui.show_view(web, mode)                         # returns when the view is closed
        state["done"] = True
        print("[editor] closed")
        return True
    except Exception as exc:                            # not Pyto, or API differs
        print("Could not open the Pyto WebView (%s)." % exc)
    try:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(html_path))
    except Exception:
        pass
    print("Live editor saved to:\n  %s\nOpen that .html file in Safari "
          "(Files app > Share > Open in Safari / Quick Look)." % html_path)
    return False


def open_live_editor(f, S, p, target_source=None, allow_nominal=False,
                     base="live", planes=None, z0=Z0):
    """Resolve an optional target, build the page and show it."""
    p_tgt, label = None, ""
    if target_source or allow_nominal:
        try:
            p_tgt, _, label = get_target(target_source, allow_nominal)
        except TargetError as exc:
            print("Live editor: no target comparison (%s)" % exc)
    try:
        html = build_editor_html(f, S, p, p_tgt, label,
                                 "Live editor - " + os.path.basename(base),
                                 planes=planes, z0=z0,
                                 save_name=os.path.basename(base) + "_tuned.s2p")
    except FileNotFoundError as exc:
        print("Live editor not opened: %s" % exc)
        return
    launch_live_editor(html, base + "_live_editor.html", base + "_tuned.s2p")


# =====================================================================
#  file location (Pyto / iPad)
# =====================================================================
def search_dirs():
    """Folders to look in: the script's folder, Pyto's working dir, Documents."""
    dirs = []
    try:
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        pass
    dirs.append(os.getcwd())
    dirs.append(os.path.expanduser("~/Documents"))
    out = []
    for d in dirs:
        if os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def find_s2p(max_depth=3):
    """Return list of .s2p files found in the folders above (and subfolders)."""
    found = []
    for root in search_dirs():
        base_depth = root.rstrip(os.sep).count(os.sep)
        for cur, subdirs, files in os.walk(root):
            if cur.count(os.sep) - base_depth >= max_depth:
                subdirs[:] = []
            subdirs[:] = [d for d in subdirs if not d.startswith((".", "__"))]
            for fn in files:
                if fn.lower().endswith(".s2p") and "_extracted" not in fn.lower():
                    full = os.path.join(cur, fn)
                    if full not in found:
                        found.append(full)
    return found


def resolve_path(name):
    """Absolute path as given, else try each Pyto folder."""
    path = name.strip().strip('"').strip("'")
    if not os.path.isabs(path):
        for d in search_dirs():
            if os.path.exists(os.path.join(d, path)):
                return os.path.join(d, path)
    return path


def choose_file():
    """Pick an .s2p from the Pyto folders, type a path, make a demo, or synthesise a target
    first ([s] returns to this menu, so the design is not asked again). None = quit."""
    here = search_dirs()[0]
    while True:
        files = find_s2p()
        print("Looking in:")
        for d in search_dirs():
            print("  ", d)
        print()
        for i, fp in enumerate(files, 1):
            print("  [%d] %s" % (i, os.path.relpath(fp, here)))
        print("  [d] create demo file")
        print("  [s] synthesise a target design first (then pick the measurement file)")
        print("  [p] type a path / filename")
        print("  [q] quit")
        choice = input("\nSelect: ").strip().lower()

        if choice == "q":
            return None
        if choice == "s":
            if synthesize_interactive():
                print("\nNow pick the measurement file; the synthesised target will be the default "
                      "target.\n")
            continue
        if choice == "d" or (choice == "" and not files):
            path = os.path.join(here, "demo_filter_zeros.s2p" if CROSS_PAIRS else "demo_filter.s2p")
            make_demo(path)
            return path
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return files[int(choice) - 1]
        if choice in ("", "p"):
            choice = input("Path or filename: ").strip()
        return resolve_path(choice)


def choose_target():
    """Returns (mode, source). mode: 'skip' | 'file' | 'nominal'.
    Blank input = nominal design (the default); type 'skip' to omit the advisor."""
    if TARGET_FILE:
        return "file", resolve_path(TARGET_FILE)
    if ALLOW_NOMINAL_FALLBACK:
        return "nominal", None
    print("\nTuning advisor target design (matrix .txt/.csv, or %d-value vector)." % PARAM_LEN)
    if SYNTH_TARGET:
        ans = input("Target file [blank = the synthesised %s, 'nominal' = nominal design, "
                    "'skip' = no advisor, 'synth' = make another]: " % os.path.basename(SYNTH_TARGET)).strip()
        if not ans:
            return "file", SYNTH_TARGET
        if ans.lower() == "synth":
            path = synthesize_interactive()
            return ("file", path) if path else ("file", SYNTH_TARGET)
        if ans.lower() in ("skip", "s", "none"):
            return "skip", None
        if ans.lower() in ("nominal", "n"):
            return "nominal", None
        return "file", resolve_path(ans)
    if CROSS_PAIRS:
        print("NOTE: blank = nominal design, which has NO transmission zeros, so cross-coupling %s "
              "would be tuned to 0. Give a target file that includes it "
              "(the demo writes *_target.txt)." % cross_text())
    ans = input("Target file [blank = nominal design, 'skip' = no advisor, "
                "'synth' = synthesise one]: ").strip()
    if ans.lower() == "synth":
        path = synthesize_interactive()
        return ("file", path) if path else ("nominal", None)
    if ans.lower() in ("skip", "s", "none"):
        return "skip", None
    if not ans or ans.lower() in ("nominal", "n"):
        return "nominal", None
    return "file", resolve_path(ans)


def choose_fit_mode():
    """None = auto. 'complex' / 'magnitude' force the mode."""
    if not PROMPT_FOR_DESIGN:
        return None
    ans = input("Fit mode [blank = auto: complex if the data has phase; "
                "'c' complex, 'm' magnitude-only]: ").strip().lower()
    if ans.startswith("c"):
        return "complex"
    if ans.startswith("m"):
        return "magnitude"
    return None


def choose_previous():
    ans = input("Previous-iteration *_matrix.txt (blank = none): ").strip()
    if not ans:
        return None
    try:
        vec, _ = load_target_file(resolve_path(ans))
    except TargetError as exc:
        print("Previous iteration ignored: %s" % exc)
        return None
    if np.isnan(vec).any():
        print("Previous iteration ignored: file lacks R1/RN/delta values.")
        return None
    return vec


# =====================================================================
#  main
# =====================================================================
def main():
    if PROMPT_FOR_DESIGN:
        choose_design()
    else:
        print("Design: " + design_summary())
    path = choose_file()
    if path is None:                 # [s] synthesise-only was chosen
        return
    if not os.path.isfile(path):
        print("File not found:", path)
        return

    try:
        f, S, z0 = read_s2p(path)
    except S2PError as exc:
        print("Cannot use this file: %s" % exc)
        return
    print("Read %d points, %.1f - %.1f MHz, R = %g ohm"
          % (len(f), f[0] / 1e6, f[-1] / 1e6, z0))

    notches = detect_notches(f, S)
    if notches and not CROSS_PAIRS:
        pairs = suggest_pairs(N, len(notches))
        print("\nThis response has %d transmission zero(s) near %s." % (
            len(notches), ", ".join("%.1f MHz" % (x / 1e6) for x, _ in notches)))
        print("Without cross-couplings the model cannot reproduce a notch, so the fit would "
              "be poor.")
        if pairs:
            txt = ",".join("%d-%d" % (a + 1, b + 1) for a, b in pairs)
            if input("Enable cross-coupling %s and fit with it? [Y/n]: " % txt).strip().lower() != "n":
                set_design(N, F0, BW, RIPPLE_DB, QU, pairs)
                print("Design: " + design_summary())
    elif CROSS_PAIRS and not notches:
        print("\nNOTE: cross-couplings %s are enabled but no transmission-zero notch was found "
              "in this response; the cross-coupling may fit to ~0." % cross_text())

    mode, source = choose_target()
    previous = choose_previous() if mode != "skip" else None

    fit_mode = choose_fit_mode()
    print("Extracting (global search, then refinement)...")
    if CROSS_PAIRS and N >= 6:
        print("  (a higher order with cross-couplings is slower: this can take a few minutes on a tablet)")
    res = extract_robust(f, S, mode=fit_mode)
    p, cost, starts = res.p, res.cost, res.starts
    print("Fit cost: %.3e" % cost)
    print("\nFIT METHOD AND IDENTIFIABILITY")
    for line in fit_summary_lines(res):
        print("  " + line)

    base, _ = os.path.splitext(path)
    out_s2p, out_txt = base + "_extracted.s2p", base + "_matrix.txt"
    S_ext = model_with_planes(res, f)        # as measured: reference planes applied
    cmt = "Extracted from %s | N=%d f0=%g MHz BW=%g MHz | %s fit" % (
        os.path.basename(path), N, F0 / 1e6, BW / 1e6, res.mode)
    if res.mode == "complex":
        cmt += " | ref-plane delays %.1f ps / %.1f ps, S21 phase %.1f deg" % (
            res.nuisance[0] * 1e12, res.nuisance[1] * 1e12, np.degrees(res.nuisance[2]))
    write_s2p(out_s2p, f, S_ext, cmt, z0)

    if mode != "skip":
        run_tuning_advisor(f, S, S_ext, p, starts, cost,
                           source if mode == "file" else None,
                           allow_nominal=(mode == "nominal"),
                           previous=previous, out_base=base, fit=res)

    report(p, out_txt)
    print("\nSaved:", out_s2p)
    print("Saved:", out_txt)

    if input("\nOpen live editable-matrix editor? [Y/n]: ").strip().lower() != "n":
        planes = ([res.nuisance[0], res.nuisance[1], res.nuisance[2], res.refl_sign]
                  if res.mode == "complex" else None)
        open_live_editor(f, S, p, source if mode == "file" else None,
                         allow_nominal=(mode == "nominal"), base=base, planes=planes, z0=z0)

    plot_compare(f, S, S_ext, os.path.basename(path) + " vs extracted")


if __name__ == "__main__":
    main()
