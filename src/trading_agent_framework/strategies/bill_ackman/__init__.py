"""The Bill Ackman portfolio strategy: a researcher, a short seller and a trader over the fundamentals quality screen."""

from trading_agent_framework.strategies.bill_ackman.agent_bill_ackman import BillAckmanStrategy
from trading_agent_framework.utils.package_helper import get_version

__all__ = ["BillAckmanStrategy"]

def __dir__():
    return list(globals().keys()) + __all__


__version__ = get_version("trading_agent_framework")
