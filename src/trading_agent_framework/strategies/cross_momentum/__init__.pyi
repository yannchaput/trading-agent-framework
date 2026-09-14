from .v1.agent_cross_momentum_v1 import CrossMomentumStrategyV1
from .v2.agent_cross_momentum_v2 import CrossMomentumStrategyV2
from .v2.agent_cross_momentum_v21 import CrossMomentumStrategyV21
from .v2.agent_cross_momentum_v22a import CrossMomentumStrategyV22a
from .v2.agent_cross_momentum_v22b import CrossMomentumStrategyV22b
from .v2.agent_cross_momentum_v22c import CrossMomentumStrategyV22c
from .v2.agent_cross_momentum_v23a import CrossMomentumStrategyV23a
from .v2.agent_cross_momentum_v23b import CrossMomentumStrategyV23b
from .v2.agent_cross_momentum_v23c import CrossMomentumStrategyV23c
from .v2.agent_cross_momentum_v24 import CrossMomentumStrategyV24
from .v2.agent_cross_momentum_v25 import CrossMomentumStrategyV25
from .v2.agent_cross_momentum_v26a import CrossMomentumStrategyV26a
from .v3.agent_cross_momentum_v3 import CrossMomentumStrategyV3
from .v4.agent_cross_momentum_v41 import CrossMomentumStrategyV41
from .v4.agent_cross_momentum_v42 import CrossMomentumStrategyV42
from .v5.agent_cross_momentum_v5 import CrossMomentumStrategyV5

_CM_CLASSES = [
    CrossMomentumStrategyV1,
    CrossMomentumStrategyV2,
    CrossMomentumStrategyV21,
    CrossMomentumStrategyV22a,
    CrossMomentumStrategyV22b,
    CrossMomentumStrategyV22c,
    CrossMomentumStrategyV23a,
    CrossMomentumStrategyV23b,
    CrossMomentumStrategyV23c,
    CrossMomentumStrategyV24,
    CrossMomentumStrategyV25,
    CrossMomentumStrategyV26a,
    CrossMomentumStrategyV3,
    CrossMomentumStrategyV41,
    CrossMomentumStrategyV42,
    CrossMomentumStrategyV5,
]
__all__ = [cls.__name__ for cls in _CM_CLASSES]
