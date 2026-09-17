"""Always-local scalar logs, with optional TensorBoard mirroring."""

import json
from pathlib import Path


class ScalarWriter:
    def __init__(self, directory, tensorboard=False):
        self.path = Path(directory) / "metrics.jsonl"
        self.tensorboard = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=directory)

    def add_scalar(self, tag, scalar_value, global_step):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"step": int(global_step), "metric": tag, "value": float(scalar_value)})
                + "\n"
            )
        if self.tensorboard:
            self.tensorboard.add_scalar(tag, scalar_value, global_step)

    def close(self):
        if self.tensorboard:
            self.tensorboard.close()
