from enum import Enum, auto

type Terminals = BranchTerminals | DiodeTerminals | BjtTerminals | MosfetTerminals

class BranchTerminals(Enum):
    POSITIVE = auto()
    NEGATIVE = auto()

class DiodeTerminals(Enum):
    ANODE = auto()
    CATHODE = auto()

class BjtTerminals(Enum):
    COLLECTOR = auto()
    BASE = auto()
    EMITTER = auto()

class MosfetTerminals(Enum):
    DRAIN = auto()
    GATE = auto()
    SOURCE = auto()