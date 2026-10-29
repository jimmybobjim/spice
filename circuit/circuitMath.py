import math

_EXP_CLAMP = 40.0

def exp(x: float) -> tuple[float, float]:
    """exp(x) and its derivative, continued linearly past _EXP_CLAMP to avoid overflow."""
    if x <= _EXP_CLAMP:
        e = math.exp(x)
        return e, e
    e = math.exp(_EXP_CLAMP)
    return e * (1.0 + x - _EXP_CLAMP), e

def pnjlim(vnew: float, vold: float, vt: float, vcrit: float) -> float:
    """SPICE's junction limiter: stop a forward-biased junction voltage leaping up the exponential."""
    if vnew > vcrit and abs(vnew - vold) > 2.0 * vt:
        if vold > 0.0:
            arg = 1.0 + (vnew - vold) / vt
            return vold + vt * math.log(arg) if arg > 0.0 else vcrit
        return vt * math.log(vnew / vt)
    return vnew

def criticalVoltage(vt: float, isat: float) -> float:
    return vt * math.log(vt / (math.sqrt(2.0) * isat))

def squareLaw(vgs: float, vds: float, vth: float, k: float, lam: float) -> tuple[float, float, float]:
    """NMOS drain current, gm and gds for vds >= 0."""
    vov = vgs - vth
    if vov <= 0.0:
        return 0.0, 0.0, 0.0
    clm = 1.0 + lam * vds
    if vds < vov:  # triode
        base = vov * vds - 0.5 * vds * vds
        return k * base * clm, k * vds * clm, k * (vov - vds) * clm + k * base * lam
    half = 0.5 * k * vov * vov  # saturation
    return half * clm, k * vov * clm, half * lam