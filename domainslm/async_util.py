from typing import Awaitable, Iterable
import asyncio


async def gather_with_semaphore[T](
    awaitables: Iterable[Awaitable[T]], max_concurrent: int
) -> list[T]:
    """Gathers awaitables with a maximum concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def sem_awaitable(aw: Awaitable[T]) -> T:
        async with semaphore:
            return await aw

    return await asyncio.gather(*(sem_awaitable(aw) for aw in awaitables))
