from __future__ import annotations

from ember_splatting_training.tensorboard_analysis import (
    checkpoint_logs_dir,
    find_event_files,
    read_scalars,
    scalar_tags,
)
from torch.utils.tensorboard import SummaryWriter


def test_tensorboard_analysis_reads_checkpoint_logs_as_polars(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    log_dir = checkpoint_logs_dir(checkpoint_dir)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_scalar("train/loss", 1.0, global_step=1)
    writer.add_scalar("train/iterations_per_second", 2.0, global_step=1)
    writer.close()

    event_files = find_event_files(checkpoint_dir)
    frame = read_scalars(checkpoint_dir)

    assert event_files
    assert frame.height == 2
    assert set(scalar_tags(frame)) == {
        "train/iterations_per_second",
        "train/loss",
    }
    assert frame.filter(frame["tag"] == "train/loss")["value"].item() == 1.0
