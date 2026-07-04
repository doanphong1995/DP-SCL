"""Supported DP-SCL experiment modes."""

DP_SCL_MODE = "dp_scl"
DP_SCL_BACKEND_MODE = "supcon_lstm_attn"
ALL_MODES = frozenset({DP_SCL_MODE})


def resolve_backend_mode(mode):
    if mode == DP_SCL_MODE:
        return DP_SCL_BACKEND_MODE
    raise ValueError(f"Unknown mode: {mode}. Valid modes: {sorted(ALL_MODES)}")

