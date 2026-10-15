"""Engine stages. Each module is one architectural step; see README."""
from .async_serial import AsyncSerialEngine
from .continuous import ContinuousBatchEngine
from .naive import NaiveEngine
from .paged import PagedEngine
from .static_batch import StaticBatchEngine

ENGINES = {
    NaiveEngine.name: NaiveEngine,
    AsyncSerialEngine.name: AsyncSerialEngine,
    StaticBatchEngine.name: StaticBatchEngine,
    ContinuousBatchEngine.name: ContinuousBatchEngine,
    PagedEngine.name: PagedEngine,
}

__all__ = ["ENGINES", "NaiveEngine", "AsyncSerialEngine", "StaticBatchEngine", "ContinuousBatchEngine",
           "PagedEngine"]
