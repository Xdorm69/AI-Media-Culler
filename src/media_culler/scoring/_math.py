from __future__ import annotations


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def ramp(value: float, in_low: float, in_high: float, out_low: float, out_high: float) -> float:
    """Linear map of ``value`` from [in_low, in_high] to [out_low, out_high], clamped at ends."""
    if value <= in_low:
        return out_low
    if value >= in_high:
        return out_high
    return out_low + (value - in_low) / (in_high - in_low) * (out_high - out_low)
