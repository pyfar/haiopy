from . import devices  # noqa: F401 TODO: remove this after implementation


class _AudioIO(object):
    def __init__(
            self,
            device
            ):
        super().__init__()

    def start():
        pass

    def stop():
        pass

    def wait():
        pass


class Playback(_AudioIO):
    def __init__(
            self,
            device,
            input_channels,
            repetitions=1,
            loop=False):
        super().__init__(device=device)
        self._output_signal = None

    @property
    def output_signal(self):
        return self._output_signal

    @output_signal.setter
    def output_signal(self, sig):
        self._output_signal = sig


class Record(_AudioIO):
    def __init__(
            self,
            device,
            output_channels,
            duration=None,
            fft_norm='amplitude',
            ):
        super().__init__(device=device)
        self._input_signal = None

    @property
    def input_signal(self):
        return self._input_signal


class PlaybackRecord(Playback, Record):
    def __init__(
            self,
            device,
            input_channels,
            output_channels,
            ):
        super().__init__(
            device=device,
            input_channels=input_channels,
            output_channels=output_channels)


def playback():
    pass


def record():
    pass


def playback_record():
    pass
