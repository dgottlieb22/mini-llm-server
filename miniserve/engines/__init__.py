"""Engine stages. Each module is one architectural step; see README."""
from .async_serial import AsyncSerialEngine
from .naive import NaiveEngine
from .static_batch import StaticBatchEngine

ENGINES = {
    NaiveEngine.name: NaiveEngine,
    AsyncSerialEngine.name: AsyncSerialEngine,
    StaticBatchEngine.name: StaticBatchEngine,
}

__all__ = ["ENGINES", "NaiveEngine", "AsyncSerialEngine", "StaticBatchEngine"]
