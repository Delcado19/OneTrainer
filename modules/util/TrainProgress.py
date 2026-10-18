class TrainProgress:
    def __init__(
            self,
            epoch: int = 0,
            epoch_step: int = 0,
            epoch_sample: int = 0,
            global_step: int = 0,
            optimizer_step: int | None = 0,
    ):
        self.epoch = epoch
        self.epoch_step = epoch_step
        self.epoch_sample = epoch_sample
        self.global_step = global_step
        # None marks legacy backups whose true update count cannot be reconstructed after accumulation changes.
        self.optimizer_step = optimizer_step

    def next_optimizer_step(self):
        if self.optimizer_step is not None:
            self.optimizer_step += 1

    def next_step(self, batch_size: int):
        self.epoch_step += 1
        self.epoch_sample += batch_size
        self.global_step += 1

    def next_epoch(self):
        self.epoch_step = 0
        self.epoch_sample = 0
        self.epoch += 1

    def filename_string(self):
        return f"{self.global_step}-{self.epoch}-{self.epoch_step}"
