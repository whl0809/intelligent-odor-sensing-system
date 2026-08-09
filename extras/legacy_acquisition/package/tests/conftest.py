from __future__ import annotations

from collections.abc import Sequence


class FakeBus:
    def __init__(self, reads: list[bytes] | None = None) -> None:
        self.reads = list(reads or [])
        self.writes: list[tuple[int, bytes]] = []
        self.registers: dict[tuple[int, int], bytes] = {}
        self.present: set[int] = set()

    def write(self, address: int, data: Sequence[int]) -> None:
        self.writes.append((address, bytes(data)))

    def read(self, address: int, length: int) -> bytes:
        if not self.reads:
            raise AssertionError("unexpected fake I2C read")
        result = self.reads.pop(0)
        assert len(result) == length
        return result

    def read_register(self, address: int, register: int, length: int = 1) -> bytes:
        result = self.registers[(address, register)]
        assert len(result) == length
        return result

    def probe(self, address: int) -> bool:
        return address in self.present


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        assert duration >= 0
        self.sleeps.append(duration)
        self.value += duration

