# Lazy imports via __getattr__ so that traversing the candidates package
# (e.g. to reach .cross_momentum.parameters) does not trigger Lumibot
# initialization.  Named imports like `from .candidates import TripleScreenStrategy`
# still work — the real module is loaded on first attribute access.

__all__ = [
    "CrossMomentumStrategyV1",
    "CrossMomentumStrategyV2",
    "CrossMomentumStrategyV21",
    "CrossMomentumStrategyV22a",
    "CrossMomentumStrategyV22b",
    "CrossMomentumStrategyV22c",
    "CrossMomentumStrategyV23a",
    "CrossMomentumStrategyV23b",
    "CrossMomentumStrategyV23c",
    "CrossMomentumStrategyV24",
    "CrossMomentumStrategyV25",
    "CrossMomentumStrategyV26a",
    "CrossMomentumStrategyV3",
    "CrossMomentumStrategyV41",
    "CrossMomentumStrategyV42",
    "CrossMomentumStrategyV5",
]


def __getattr__(name: str):
    _LAZY = {
        "CrossMomentumStrategyV1": (".v1.agent_cross_momentum_v1", "CrossMomentumStrategyV1"),
        "CrossMomentumStrategyV2": (".v2.agent_cross_momentum_v2", "CrossMomentumStrategyV2"),
        "CrossMomentumStrategyV21": (".v2.agent_cross_momentum_v21", "CrossMomentumStrategyV21"),
        "CrossMomentumStrategyV22a": (".v2.agent_cross_momentum_v22a", "CrossMomentumStrategyV22a"),
        "CrossMomentumStrategyV22b": (".v2.agent_cross_momentum_v22b", "CrossMomentumStrategyV22b"),
        "CrossMomentumStrategyV22c": (".v2.agent_cross_momentum_v22c", "CrossMomentumStrategyV22c"),
        "CrossMomentumStrategyV23a": (".v2.agent_cross_momentum_v23a", "CrossMomentumStrategyV23a"),
        "CrossMomentumStrategyV23b": (".v2.agent_cross_momentum_v23b", "CrossMomentumStrategyV23b"),
        "CrossMomentumStrategyV23c": (".v2.agent_cross_momentum_v23c", "CrossMomentumStrategyV23c"),
        "CrossMomentumStrategyV24": (".v2.agent_cross_momentum_v24", "CrossMomentumStrategyV24"),
        "CrossMomentumStrategyV25": (".v2.agent_cross_momentum_v25", "CrossMomentumStrategyV25"),
        "CrossMomentumStrategyV26a": (".v2.agent_cross_momentum_v26a", "CrossMomentumStrategyV26a"),
        "CrossMomentumStrategyV3": (".v3.agent_cross_momentum_v3", "CrossMomentumStrategyV3"),
        "CrossMomentumStrategyV41": (".v4.agent_cross_momentum_v41", "CrossMomentumStrategyV41"),
        "CrossMomentumStrategyV42": (".v4.agent_cross_momentum_v42", "CrossMomentumStrategyV42"),
        "CrossMomentumStrategyV5": (".v5.agent_cross_momentum_v5", "CrossMomentumStrategyV5"),
    }

    if name in _LAZY:
        from importlib import import_module  # noqa: PLC0415

        module_path, attr = _LAZY[name]
        mod = import_module(module_path, package=__name__)
        value = getattr(mod, attr)
        # Cache in globals so __getattr__ is only called once per name
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + __all__
