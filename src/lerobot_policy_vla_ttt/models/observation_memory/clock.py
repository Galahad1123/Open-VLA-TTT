"""Physical-observation clock. Denoising iterations cannot advance it."""


class ObservationClock:
    def __init__(self, execution):
        if execution < 2 or execution % 2:
            raise ValueError("Execution horizon must be even")
        self.execution, self.stride = execution, execution // 2
        self.last = -1

    def check(self, step):
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("env_step must be a nonnegative integer")
        if step < self.last:
            raise ValueError("Observation time moved backwards; reset episode")
        if step == self.last:
            return False
        next_write = ((self.last // self.stride) + 1) * self.stride
        if next_write < step:
            raise ValueError("Cannot skip a scheduled midpoint/decision observation")
        return True

    def commit(self, step):
        self.last = step
