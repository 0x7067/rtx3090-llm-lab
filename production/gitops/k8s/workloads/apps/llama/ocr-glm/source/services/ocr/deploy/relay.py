"""Bounded TCP ingress to the private OCR API; no request-selected destination."""
from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass, field

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 9417
UPSTREAM_HOST = "ocr-api"
UPSTREAM_PORT = 9417
MAX_CONNECTIONS = 32
CHUNK_BYTES = 64 * 1024
MAX_DIRECTION_BYTES = 16 * 1024 * 1024
CONNECT_SECONDS = 5
IDLE_SECONDS = 30
OVERALL_SECONDS = 120


class RelayLimitError(Exception):
    pass


@dataclass
class Activity:
    updated: float = field(default_factory=time.monotonic)
    finished: int = 0
    complete: asyncio.Event = field(default_factory=asyncio.Event)


async def pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
               activity: Activity) -> None:
    transferred = 0
    while True:
        chunk = await reader.read(CHUNK_BYTES)
        if not chunk:
            # Propagate only the write half-close. The opposite pump must keep
            # forwarding a valid response after a client finishes its request.
            if writer.can_write_eof():
                writer.write_eof()
                await writer.drain()
            activity.finished += 1
            if activity.finished == 2:
                activity.complete.set()
            return
        transferred += len(chunk)
        if transferred > MAX_DIRECTION_BYTES:
            raise RelayLimitError()
        writer.write(chunk)
        await writer.drain()
        activity.updated = time.monotonic()


async def watch_idle(activity: Activity) -> None:
    while not activity.complete.is_set():
        remaining = IDLE_SECONDS - (time.monotonic() - activity.updated)
        if remaining <= 0:
            raise RelayLimitError()
        try:
            await asyncio.wait_for(activity.complete.wait(), remaining)
        except TimeoutError:
            # Either direction may have transferred bytes while waiting.
            continue


async def close_stream(writer: asyncio.StreamWriter | None) -> None:
    if writer is None:
        return
    writer.close()
    try:
        await asyncio.wait_for(writer.wait_closed(), 1)
    except (TimeoutError, OSError):
        writer.transport.abort()


class Relay:
    def __init__(self, connector=asyncio.open_connection):
        # Injection is for local functional tests; production main has no target
        # arguments, environment override, HTTP parsing or CONNECT support.
        self.connector = connector
        self.active: set[asyncio.Task] = set()

    async def handle(self, reader: asyncio.StreamReader,
                     writer: asyncio.StreamWriter) -> None:
        if len(self.active) >= MAX_CONNECTIONS:
            writer.close()
            return
        task = asyncio.current_task()
        self.active.add(task)
        upstream_writer = None
        try:
            writer.transport.set_write_buffer_limits(high=CHUNK_BYTES, low=CHUNK_BYTES // 2)
            async with asyncio.timeout(OVERALL_SECONDS):
                upstream_reader, upstream_writer = await asyncio.wait_for(
                    self.connector(UPSTREAM_HOST, UPSTREAM_PORT, limit=CHUNK_BYTES),
                    CONNECT_SECONDS,
                )
                upstream_writer.transport.set_write_buffer_limits(high=CHUNK_BYTES, low=CHUNK_BYTES // 2)
                activity = Activity()
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(pump(reader, upstream_writer, activity))
                    tasks.create_task(pump(upstream_reader, writer, activity))
                    tasks.create_task(watch_idle(activity))
        except (Exception, asyncio.CancelledError):
            # Transport/timeout errors can include addresses or payload details.
            # The API owns authentication and structured errors; emit no request logs.
            pass
        finally:
            await asyncio.gather(close_stream(writer), close_stream(upstream_writer), return_exceptions=True)
            self.active.discard(task)

    async def close(self) -> None:
        pending = list(self.active)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)


async def main() -> None:
    relay = Relay()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    server = await asyncio.start_server(relay.handle, LISTEN_HOST, LISTEN_PORT,
                                       limit=CHUNK_BYTES, backlog=MAX_CONNECTIONS)
    print("Private OCR ingress relay ready.", flush=True)
    try:
        async with server:
            await stop.wait()
    finally:
        server.close()
        await server.wait_closed()
        await relay.close()


if __name__ == "__main__":
    asyncio.run(main())
