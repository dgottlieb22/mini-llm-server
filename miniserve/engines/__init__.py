"""Engine stages. Each module is one architectural step; see README."""
from .async_serial import AsyncSerialEngine
from .naive import NaiveEngine

ENGINES = {
    NaiveEngine.name: NaiveEngine,
    AsyncSerialEngine.name: AsyncSerialEngine,
}

__all__ = ["ENGINES", "NaiveEngine", "AsyncSerialEngine"]
