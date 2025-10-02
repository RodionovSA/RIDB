import numpy as np

# Physical constants (SI)
c = 299_792_458.0               # m/s
h = 6.626_070_15e-34            # J·s
e = 1.602_176_634e-19           # C (J/eV)

# Unit scales
_WL = {'A':1e-10, 'nm':1e-9, 'um':1e-6, 'cm':1e-2, 'm':1.0}
_FR = {'Hz':1.0, 'THz':1e12, '1/um': c * 1e6}
_EN = {'eV':1.0}

def convert_spectral(x, from_u: str, to_u: str):
    """Convert scalar/array x between wavelength (A,nm,um,cm,m), frequency (Hz,THz), and energy (eV)."""
    x = np.asarray(x, dtype=float)

    # wavelength → wavelength
    if from_u in _WL and to_u in _WL:
        return x * (_WL[from_u] / _WL[to_u])

    # frequency → frequency
    if from_u in _FR and to_u in _FR:
        return x * (_FR[from_u] / _FR[to_u])

    # energy → energy (only eV supported here; extend if needed)
    if from_u in _EN and to_u in _EN:
        return x  # same

    # wavelength ↔ frequency
    if from_u in _WL and to_u in _FR:
        lam_m = x * _WL[from_u]
        nu_Hz = c / lam_m
        return nu_Hz / _FR[to_u]
    if from_u in _FR and to_u in _WL:
        nu_Hz = x * _FR[from_u]
        lam_m = c / nu_Hz
        return lam_m / _WL[to_u]

    # wavelength ↔ energy
    if from_u in _WL and to_u in _EN:
        lam_m = x * _WL[from_u]
        E_eV = (h * c) / (lam_m * e)
        return E_eV / _EN[to_u]
    if from_u in _EN and to_u in _WL:
        E_eV = x * _EN[from_u]
        lam_m = (h * c) / (E_eV * e)
        return lam_m / _WL[to_u]

    # frequency ↔ energy
    if from_u in _FR and to_u in _EN:
        nu_Hz = x * _FR[from_u]
        E_eV = (h * nu_Hz) / e
        return E_eV / _EN[to_u]
    if from_u in _EN and to_u in _FR:
        E_eV = x * _EN[from_u]
        nu_Hz = (E_eV * e) / h
        return nu_Hz / _FR[to_u]

    raise ValueError(f"Unsupported units: {from_u} → {to_u}")

def lorentzfunc(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Returns the complex ε profile given a set of Lorentzian parameters p
    (σ_0, ω_0, γ_0, σ_1, ω_1, γ_1, ...) for a set of frequencies x in 1/um units.
    (From MEEP)
    """
    N = len(p) // 3
    y = np.zeros(len(x))
    for n in range(N):
        A_n = p[3 * n + 0]
        x_n = p[3 * n + 1]
        g_n = p[3 * n + 2]
        y = y + A_n / (np.square(x_n) - np.square(x) - 1j * x * g_n)
    return y