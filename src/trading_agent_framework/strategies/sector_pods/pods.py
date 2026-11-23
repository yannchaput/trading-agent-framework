"""The sleeve table: five sector pods and the ETFs each one covers. Prompts and checks are built from it."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pod:
    """One sector pod.

    Attributes:
        key: Stable identifier, e.g. `tech_pod`.
        label: Name shown to the agents.
        focus: One sentence on what drives the pod's sleeve.
        sleeve: The ETF symbols the pod rates.
    """

    key: str
    label: str
    focus: str
    sleeve: tuple[str, ...]


PODS: tuple[Pod, ...] = (
    Pod(
        "tech_pod",
        "Technology & Communications",
        "Interest rates, growth expectations and liquidity.",
        ("XLK", "XLC", "SMH", "IGV", "FDN"),
    ),
    Pod(
        "financials_pod",
        "Financials & Real Estate",
        "The yield curve, credit spreads and policy rates.",
        ("XLF", "KRE", "KIE", "IYR"),
    ),
    Pod(
        "healthcare_pod",
        "Healthcare",
        "Defensive growth, drug pricing and health policy.",
        ("XLV", "XBI", "IHI", "XPH", "IHF"),
    ),
    Pod(
        "energy_pod",
        "Energy, Utilities, Materials & Industrials",
        "Oil and gas prices, inflation, the dollar, the industrial cycle and utilities' rate sensitivity.",
        ("XLE", "XOP", "OIH", "XLU", "XLB", "XME", "XLI", "ITA", "IYT"),
    ),
    Pod(
        "consumer_pod",
        "Consumer & Housing",
        "Income, inflation, housing and retail demand.",
        ("XLY", "XLP", "XHB", "XRT"),
    ),
)

ALL_SYMBOLS: tuple[str, ...] = tuple(symbol for pod in PODS for symbol in pod.sleeve)

DEFENSIVE_ETFS: tuple[str, ...] = ("XLP", "XLV", "XLU")  # the sleeves that fall least in a stressed market

_POD_OF = {symbol: pod for pod in PODS for symbol in pod.sleeve}


def pod_of(symbol: str) -> Pod:
    """Return the pod whose sleeve holds `symbol`.

    Raises:
        KeyError: The symbol is in no sleeve.
    """
    return _POD_OF[symbol]
