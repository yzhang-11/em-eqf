"""Compatibility entry: legacy additive-phase defaults remain unchanged."""
from src.models.mamba.mamba3_event_time import (
    _Mamba3TimeBase, _Mamba3EventTimeFunction, _NativeAngleFunction, _event_time_scan,
)

class Mamba3Rotary(_Mamba3TimeBase):
    """Legacy name: base updates, SISO default, extra event-time phase enabled."""
    pass
