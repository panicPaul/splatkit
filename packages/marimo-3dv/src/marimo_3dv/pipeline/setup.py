"""Typed setup pipeline for transforming source scene data into render data."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

import beartype
import beartype.door

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
NextT = TypeVar("NextT")


class SetupPipeline(Generic[InputT, OutputT]):
    """Typed sequential pipeline that transforms source data into render data.

    Each op is a plain Python callable that consumes the previous output type
    and returns the next output type. Beartype validates wiring compatibility
    at ``.pipe()`` time so mismatches fail early.

    Example::

        pipeline = (
            SetupPipeline()
            .pipe(load_scene)
            .pipe(normalize_scene)
        )
        render_data = pipeline.run(raw_input)
    """

    def __init__(self) -> None:
        self._ops: list[Callable[[Any], Any]] = []

    def pipe(
        self, op: Callable[[OutputT], NextT]
    ) -> SetupPipeline[InputT, NextT]:
        """Append a typed setup op and return a new pipeline with updated output type.

        Args:
            op: A callable that consumes the current output type and returns
                the next output type. Beartype validates the annotation at
                call time when the pipeline is run.

        Returns:
            A new SetupPipeline with the updated output type.
        """
        next_pipeline: SetupPipeline[InputT, NextT] = SetupPipeline()
        next_pipeline._ops = [*self._ops, beartype.beartype(op)]
        return next_pipeline

    def run(self, input_data: InputT) -> OutputT:
        """Run all ops sequentially and return the final transformed output.

        Args:
            input_data: The initial input, consumed by the first op.

        Returns:
            The result after all ops have been applied in order.
        """
        result: Any = input_data
        for op in self._ops:
            result = op(result)
        return result  # type: ignore[return-value]
