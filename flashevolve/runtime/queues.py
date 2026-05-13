import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class QueueClosed(Exception):
    """Raised by ``Queue.get`` when the queue is closed and drained."""


class Queue(ABC, Generic[T]):
    """Async queue protocol connecting two stages.

    Single-producer, multi-consumer: one upstream stage puts; one or more
    downstream workers get. Closing is one-way and idempotent. After
    ``close``, pending and future ``get`` calls raise ``QueueClosed``
    once the buffer is drained.
    """

    @abstractmethod
    async def put(self, item: T) -> None: ...

    @abstractmethod
    async def get(self) -> T:
        """Return the next item; raise ``QueueClosed`` when closed and empty."""

    @abstractmethod
    async def close(self) -> None:
        """Signal no more items will be put. Idempotent."""


class FIFOQueue(Queue[T]):
    """First-in-first-out queue backed by ``asyncio.Queue``.

    Items are delivered in the order they were ``put``. Bounded when
    ``maxsize > 0``: producers block on ``put`` until space frees. EOF is
    propagated via an internal sentinel that is re-enqueued on
    observation so any number of consumers will each see ``QueueClosed``
    exactly once.
    """

    _EOF: object = object()

    def __init__(self, maxsize: int = 0) -> None:
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    async def put(self, item: T) -> None:
        if self._closed:
            raise RuntimeError("put on closed queue")
        await self._q.put(item)

    async def get(self) -> T:
        item = await self._q.get()
        if item is FIFOQueue._EOF:
            # Re-enqueue so sibling consumers also observe EOF.
            self._q.put_nowait(FIFOQueue._EOF)
            raise QueueClosed
        return item  # type: ignore[return-value]

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._q.put(FIFOQueue._EOF)
