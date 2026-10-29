from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ModelCard:
    """A ``.model`` card.

    Attributes:
        name: Lower-cased model name.
        kind: ``"d"``, ``"npn"``, ``"pnp"``, ``"nmos"`` or ``"pmos"``.
        params: Parameter values by lower-cased name (``is``, ``bf``, ``vto`` ...), already numeric.
        line: 1-based line number.
    """
    name: str
    kind: Literal["d", "npn", "pnp", "nmos", "pmos"]
    params: dict[str, float]
    line: int = 0
