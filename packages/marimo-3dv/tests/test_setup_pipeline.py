"""Tests for SetupPipeline typed composition."""

import pytest

from marimo_3dv.pipeline.setup import SetupPipeline


def test_empty_pipeline_returns_input():
    pipeline: SetupPipeline[int, int] = SetupPipeline()
    assert pipeline.run(42) == 42


def test_single_op():
    pipeline = SetupPipeline().pipe(lambda x: x * 2)
    assert pipeline.run(5) == 10


def test_multiple_ops_chain():
    pipeline = (
        SetupPipeline().pipe(lambda x: x + 1).pipe(lambda x: x * 3).pipe(str)
    )
    assert pipeline.run(4) == "15"


def test_pipe_returns_new_pipeline():
    p1 = SetupPipeline()
    p2 = p1.pipe(lambda x: x)
    assert p1 is not p2


def test_beartype_catches_wrong_input_type():
    def expect_int(x: int) -> int:
        return x + 1

    pipeline = SetupPipeline().pipe(expect_int)

    with pytest.raises(Exception):
        pipeline.run("not an int")


def test_ops_run_in_order():
    log = []

    def record(label: str):
        def op(x: list) -> list:
            log.append(label)
            return x + [label]

        return op

    pipeline = (
        SetupPipeline().pipe(record("a")).pipe(record("b")).pipe(record("c"))
    )
    result = pipeline.run([])
    assert result == ["a", "b", "c"]
    assert log == ["a", "b", "c"]
