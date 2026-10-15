"""Engine stages. Each module is one architectural step; see README."""
from .naive import NaiveEngine

ENGINES = {
    NaiveEngine.name: NaiveEngine,
}

__all__ = ["ENGINES", "NaiveEngine"]
