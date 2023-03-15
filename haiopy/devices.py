import sounddevice as sd  # noqa: F401 TODO: remove this after implementation


def list_devices():
    pass


class _Device(object):
    def __init__(
            self,
            id,
            sampling_rate,
            block_size,
            dtype):
        super().__init__()

    @property
    def name(self):
        raise NotImplementedError('Abstract method')

    def playback():
        pass

    def record():
        pass

    def playback_record():
        pass

    def initialize_playback():
        pass

    def initialize_record():
        pass

    def initialize_playback_record():
        pass

    def abort():
        pass


class AudioDevice(_Device):
    def __init__(
            self,
            id,
            sampling_rate,
            block_size,
            dtype,
            latency=None,
            extra_settings=None,
            # finished_callback=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None
            ):
        super().__init__(id, sampling_rate, block_size, dtype)

    @property
    def stream():
        pass

    @staticmethod
    def callback():
        pass

    def playback(data):
        # fill queue, stream.start()
        pass

    def record(n_samples):
        # stream start, read into the queue
        pass

    def playback_record(data):
        # see combination above
        pass

    def initialize_playback(channels):
        # init queue, define callback, init stream
        pass

    def initialize_record(channels):
        pass

    def initialize_playback_record(input_channels, output_channels):
        pass

    def abort():
        # abort
        pass

    def close():
        # remove stream
        pass
