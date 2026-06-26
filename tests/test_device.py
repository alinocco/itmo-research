"""Tests for device resolution."""
from textvec.utils import resolve_device


def test_resolve_device_cuda_fallback_when_unavailable(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeBackends:
        mps = None

    class FakeTorch:
        cuda = FakeCuda()
        backends = FakeBackends()
        version = type("V", (), {"cuda": None})()

    monkeypatch.setitem(__import__("sys").modules, "torch", FakeTorch)
    assert resolve_device("cuda") == "cpu"
