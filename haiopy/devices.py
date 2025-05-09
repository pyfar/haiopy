from multiprocessing import Event
import numpy as np
import sys
import sounddevice as sd
from abc import abstractmethod, ABCMeta
import platform

from haiopy.buffers import EmptyBuffer
from haiopy.buffers import _Buffer


def list_devices():
    pass


class _Device(metaclass=ABCMeta):
    def __init__(
            self,
            name,
            sampling_rate,
            block_size,
            dtype):
        super().__init__()
        self._name = name
        self._sampling_rate = sampling_rate
        self._block_size = block_size
        self._dtype = dtype

    @property
    def name(self):
        return self._name

    @property
    def sampling_rate(self):
        """Sampling rate of the device."""
        return self._sampling_rate

    @sampling_rate.setter
    @abstractmethod
    def sampling_rate(self, sampling_rate):
        """Set the sampling rate of the device."""
        raise NotImplementedError('Needs to be implemented in child class.')

    @property
    def block_size(self):
        """Block size used by the device."""
        return self._block_size

    @block_size.setter
    @abstractmethod
    def block_size(self, block_size):
        """Set the block size of the device."""
        raise NotImplementedError('Needs to be implemented in child class.')

    @property
    def dtype(self):
        """Data type of the devices audio buffer."""
        return self._dtype

    @dtype.setter
    @abstractmethod
    def dtype(self, dtype):
        """Set the data type of the device."""
        raise NotImplementedError('Needs to be implemented in child class.')


class AudioDevice(_Device):
    """Abstract class implementing audio devices based on python-sounddevice.
    """

    def __init__(
            self,
            identifier=0,
            sampling_rate=44100,
            block_size=512,
            dtype='float32',
            ):

        identifier = sd.query_devices(identifier)['name']

        super().__init__(
            name=sd.query_devices(identifier)['name'],
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype,
        )
        self._identifier = identifier

        self._callback = None
        self._stream = None
        self._input_buffer = None
        self._output_buffer = None

        self._stream_finished = Event()

    @property
    def identifier(self):
        """The identifier of the device."""
        return self._identifier

    @abstractmethod
    def check_settings():
        """Check if settings are compatible with the physical device.
        """
        raise NotImplementedError('Needs to be implemented in child class.')

    @property
    def name(self):
        """The name of the device."""
        return self._name

    @property
    def stream(self):
        """The sounddevice audio stream.
        """
        return self._stream

    def _stream_active(self):
        """Check if the stream is active."""
        return self.stream.active if self.stream is not None else False

    def finished_callback(self) -> None:
        """Custom callback after a audio stream has finished.
        Can be overwritten by users.
        """
        pass

    def _finished_callback(self) -> None:
        """Private portaudio callback after a audio stream has finished.

        Ensures that the buffer is stopped.
        """
        self._stream_finished.set()
        self.finished_callback()
        self.stream.stop()

    def start(self):
        """Start the audio stream and consume the buffer."""
        if self.stream.closed:
            print("Stream is closed. Try re-initializing.", file=sys.stderr)
            return

        elif not self.stream.active:
            self._stream_finished.clear()
            self.stream.start()
        else:
            print("Stream is already active.", file=sys.stderr)

    def wait(self):
        """Wait for the audio stream to finish the buffer."""
        self._stream_finished.wait(timeout=None)

    def abort(self):
        """Stop the audio steam without finishing remaining callbacks."""
        if self.stream.active is True:
            self.stream.abort()
            self._stop_buffer()

    def close(self):
        """Close the audio device and release the sound card lock."""
        if self.stream is not None:
            self.stream.close()
            self._stop_buffer()

    def stop(self):
        """Stop the audio stream after finishing all remaining callbacks."""
        if self.stream.active is True:
            self.stream.stop()
            self._stop_buffer()

    @abstractmethod
    def _stop_buffer(self):
        raise NotImplementedError()

    @abstractmethod
    def _close_stream(self):
        raise NotImplementedError()


class _ChannelMapping(metaclass=ABCMeta):
    """Class to handle the channel mapping of the device.

    Parameters
    ----------
    channels : list
        The channels to be used by the device.
    """

    _valid_apis_windows = [
        'asio',
        'windows directsound', 'directsound',
        'windows wdm-ks', 'wdm',
        'windows wasapi', 'wsapi']
    _valid_apis_linux = [
        'alsa',
        'oss',
        'pulse',
        'jack']
    _valid_apis_darwin = [
        'coreaudio']

    _valid_apis = {
        'Windows': _valid_apis_windows,
        'Linux': _valid_apis_linux,
        'Darwin': _valid_apis_darwin,
    }

    _default_apis = {
        'Windows': 'asio',
        'Linux': 'alsa',
        'Darwin': 'coreaudio',
    }

    def __init__(
            self,
            channels: list[int],
            n_channels_device: int,
            api: str):

        if api.lower() not in self._valid_apis[platform.system()]:
            raise ValueError(
                f"Invalid driver {api}. For your platform supported drivers"
                f" are: f{self._valid_apis[platform.system()]}")

        self._api = api.lower()
        self._n_channels_device = n_channels_device
        self.channels = channels

    @property
    def channels(self) -> list[int]:
        """The channels to be used by the device."""
        return self._channels

    @channels.setter
    @abstractmethod
    def channels(self, channels: list[int]):
        """Set the channels to be used by the device."""
        raise NotImplementedError()

    @property
    def n_channels_used(self) -> int:
        """The number of channels containing data."""
        return len(self._channels)

    @property
    def n_channels_device(self) -> int:
        """The number of channels supported by the device."""
        return self._n_channels_device

    @n_channels_device.setter
    def n_channels_device(self, n_channels_device: int):
        """Set the number of channels supported by the device."""
        self._n_channels_device = n_channels_device

    @property
    def n_channels_mapping(self) -> int:
        """The number of output channels required for the stream.

        This includes a number of unused pre-pended channels which need to be
        filled with zeros before writing the portaudio buffer. In case of
        using only the first channel, portaudio plays back a mono signal,
        which will be broadcast to the first two channels. To avoid this,
        the minimum number of channels opened is always two, the unused second
        channel is filled with zeros.
        """
        if self.extra_settings is not None:
            return self.n_channels_used
        else:
            return np.max((2, np.max(self._channels) + 1))

    @property
    def extra_settings(self) -> sd.AsioSettings | sd.CoreAudioSettings | None:
        """The sounddevice extra settings for the device.

        These are specific to python-sounddevice and are used when opening the
        portaudio stream.
        """
        return self._extra_settings

    def __call__(
            self,
            data_buffer: np.ndarray[float]) -> np.ndarray[float]:
        """Apply the mapping defined by the object to a data buffer.

        Parameters
        ----------
        data_buffer : np.ndarray[float]
            The input data buffer with shape (n_channels, block_size).

        Returns
        -------
        np.ndarray[float]
            The output data buffer with shape (n_channels_mapping, block_size).
        """

        if self._api in ['asio', 'coreaudio']:
            # ASIO and CoreAudio handle the routing
            return np.atleast_2d(data_buffer).T

        # Write a block to an array with all required output channels
        # including zeros for unused channels. Required if the routing
        # is not handled by ASIO or CoreAudio. Sounddevice alone does
        # not support routing matrices
        data = data_buffer
        block_size = data_buffer.shape[-1]

        size_matches = block_size == self.n_channels_mapping

        if not self._stream_block_out or not size_matches:
            self._stream_block_out = np.zeros(
                (self.n_channels_mapping, block_size),
                dtype=data.dtype)
        self._stream_block_out[self.channels] = data

        return self._stream_block_out.T


class InputChannelMapping(_ChannelMapping):
    """Class to handle the input channel mapping of an audio device.

    Examples
    --------
    Create a mapping for a device with 8 channels and use only the second
    channel. The input data is a 2D array with shape (1, 512).
    After calling the mapping, the output data is a 2D array with shape
    (512, 2), where the first channel is filled with zeros and the second
    channel is filled with the input data.

    >>> import numpy as np
    >>> from haiopy.devices import InputChannelMapping
    >>> input_data = np.random.randn((1, 512), dtype='float32')
    >>> device = InputChannelMapping(
    ...     channels=[2],
    ...     n_channels_device=8,
    ...     api='wasapi')
    >>> device(input_data)

    """

    def __init__(
            self,
            channels: list[int],
            n_channels_device: int,
            api: str):
        super().__init__(channels, n_channels_device, api)

    @_ChannelMapping.channels.setter
    def channels(self, channels: list[int]):
        """Set the channels to be used by the device."""
        if np.any(np.asarray(channels) > self.n_channels_device):
            raise ValueError(
                f"Invalid channels {channels}. The device only supports "
                f"{self.n_channels_device} channels.")

        if 'asio' in self._api:
            extra_settings = sd.AsioSettings(
                channel_selectors=channels)

        elif 'coreaudio' in self._api:
            extra_settings = sd.CoreAudioSettings(
                channel_map=channels)
        else:
            extra_settings = None
            self._stream_block_out = None

        self._channels = channels
        self._extra_settings = extra_settings


class OutputChannelMapping(_ChannelMapping):
    """Class to handle the output channel mapping of an audio device.

    Examples
    --------
    Create a mapping for a device with 8 channels and use only the second
    channel. The input data is a 2D array with shape (1, 512).
    After calling the mapping, the output data is a 2D array with shape
    (512, 2), where the first channel is filled with zeros and the second
    channel is filled with the input data.

    >>> import numpy as np
    >>> from haiopy.devices import InputChannelMapping
    >>> input_data = np.random.randn((1, 512), dtype='float32')
    >>> device = InputChannelMapping(
    ...     channels=[2],
    ...     n_channels_device=8,
    ...     api='wasapi')
    >>> device(input_data)

    """

    def __init__(
            self,
            channels: list[int],
            n_channels_device: int,
            api: str):
        super().__init__(channels, n_channels_device, api)

    @_ChannelMapping.channels.setter
    def channels(self, channels: list[int]):
        """Set the channels to be used by the device."""
        if np.any(np.asarray(channels) > self.n_channels_device):
            raise ValueError(
                f"Invalid channels {channels}. The device only supports "
                f"{self.n_channels_device} channels.")

        if 'asio' in self._api:
            extra_settings = sd.AsioSettings(
                channel_selectors=channels)

        elif 'coreaudio' in self._api:
            channel_map = np.ones(
                self.n_channels_device, dtype=int) * -1
            channel_map[channels] = channels

            extra_settings = sd.CoreAudioSettings(
                channel_map=channel_map)
        else:
            extra_settings = None
            self._stream_block_out = None

        self._channels = channels
        self._extra_settings = extra_settings


class OutputAudioDevice(AudioDevice):

    def __init__(
            self,
            identifier=sd.default.device['output'],
            sampling_rate=44100,
            block_size=512,
            channels=[1],
            dtype='float32',
            output_buffer=None,
            latency=None,
            extra_settings=None,
            clip_off=None,
            dither_off=None,
            never_drop_input=None,
            prime_output_buffers_using_stream_callback=None):

        # First check the settings before continuing
        max_channel = np.max(channels)
        n_channels = len(channels)
        sd.check_output_settings(
            device=identifier,
            channels=np.max([n_channels, max_channel+1]),
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)
        self._extra_settings = extra_settings

        self._identifier = identifier
        super().__init__(
            identifier=identifier,
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype)

        self._output_channels = channels

        if output_buffer is None:
            output_buffer = SignalBuffer(
                self.block_size,
                pf.Signal(np.zeros(
                        (self.n_channels_output, self.block_size),
                        dtype=self.dtype),
                    self.sampling_rate, fft_norm='rms'))
        self.output_buffer = output_buffer
        self.initialize()

    def check_settings(
            self,
            n_channels=None,
            sampling_rate=None,
            dtype=None,
            extra_settings=None):
        """Check if settings are compatible with the physical devices.

        Parameters
        ----------
        n_channels : int
            The number of channels to be used
        sampling_rate : int
            The audio sampling rate
        dtype : np.float32, np.int8, np.int16, np.int32
            The audio buffer data type
        extra_settings : extra settings
            Audio API specific settings.

        Raises
        ------
        PortAudioError
            If the settings are incompatible with the device an exception is
            raised.
        """
        sd.check_output_settings(
            device=self.id,
            channels=n_channels,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

    @property
    def output_channels(self):
        return self._output_channels

    @property
    def n_channels_output(self):
        """The total number of output channels.

        Returns
        -------
        int
            The number of output channels
        """
        return len(self._output_channels)

    @property
    def _n_channels_stream(self):
        """The number of output channels required for the stream.

        This includes a number of unused pre-pended channels which need to be
        filled with zeros before writing the portaudio buffer. In case of
        using only the first channel, portaudio plays back a mono signal,
        which will be broadcasted to the first two channels. To avoid this,
        the minimum number of channels opened is always two, the unused second
        channel is filled with zeros.
        """
        return np.max((2, np.max(self._output_channels) + 1))

    @property
    def max_channels_output(self):
        """The number of output channels supported by the device"""
        return sd.query_devices(self.id, 'output')['max_output_channels']

    def output_callback(self, outdata, frames, time, status) -> None:
        """Portudio callback for output streams

        Parameters
        ----------
        outdata : array
            Output buffer view
        frames : int
            Length of the buffer
        time : PaTimestamp
            Timestamp of the callback event
        status : sounddevice.CallbackFlags
            Portaudio status flags

        Raises
        ------
        sd.CallbackAbort
            Abort the playback if a buffer underflow occurs.
        sd.CallbackStop
            Stop the playback if the output queue is empty.
        """
        assert frames == self.block_size
        if status.output_underflow:
            print('Output underflow: increase blocksize?', file=sys.stderr)
            raise sd.CallbackAbort('Buffer underflow')
        assert not status

        try:
            # Write a block to an array with all required output channels
            # including zeros for unused channels. Required as sounddevice does
            # not support routing matrices
            self._stream_block_out[self.output_channels] = next(
                self.output_buffer)
            outdata[:] = self._stream_block_out.T
        except StopIteration as e:
            raise sd.CallbackStop("Buffer empty") from e

    def initialize(self):
        """Initialize the playback stream for a given number of channels."""
        # Init array buffering a block of all required output channels
        # including zeros for unused channels. Required as sounddevice does
        # not support routing matrices
        self._stream_block_out = np.zeros(
            (self._n_channels_stream, self.block_size), dtype=self.dtype)

        ostream = sd.OutputStream(
            self.sampling_rate,
            self.block_size,
            self.id,
            self._n_channels_stream,
            self._dtype,
            callback=self.output_callback,
            finished_callback=self._finished_callback)
        self._stream = ostream

    def initialize_buffer(self):
        self.output_buffer._start()
        self.output_buffer._is_active.wait()

    @property
    def output_buffer(self):
        return self._output_buffer

    @output_buffer.setter
    def output_buffer(self, buffer):
        """Sets the output buffer"""
        if buffer.block_size != self.block_size:
            raise ValueError(
                "The buffer's block size does not match. ",
                f"Needs to be {self.block_size}")

        if buffer.n_channels != self.n_channels_output:
            raise ValueError(
                "The buffer's channel number does not match the channel "
                f"mapping. Currently used channels are {self.output_channels}")

        self._output_buffer = buffer

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, identifier):
        if self.stream.active is True or self.output_buffer.is_active is True:
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self._close_stream()
        max_channel = np.max(self._output_channels)
        n_channels = len(self._output_channels)
        sd.check_output_settings(
            device=sd.query_devices(identifier)['name'],
            channels=np.max([n_channels, max_channel+1]),
            dtype=self._dtype,
            extra_settings=self._extra_settings,
            samplerate=self._sampling_rate)
        self._id = sd.query_devices(identifier)['name']
        self.initialize()

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, block_size):
        """Sets the blocksize of the OutputDevice and the output buffer.
        Therefore, the current stream is closed and a new stream with setted
        blocksize and output buffer is initialized."""
        if self.stream.active is True or self.output_buffer.is_active is True:
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self._close_stream()
        self._block_size = block_size
        self.output_buffer.block_size = block_size
        self.initialize()

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        """Sets the sampling rate of the OutputDevice and the output buffer.
        Therefore, the current stream is closed and a new stream with setted
        samplingrate and output buffer is initialized."""
        if self.stream.active is True or self.output_buffer.is_active is True:
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self.check_settings(sampling_rate=sampling_rate)
        self._close_stream()
        self._sampling_rate = sampling_rate
        self.output_buffer.sampling_rate = sampling_rate
        self.initialize()

    @property
    def channels(self):
        return self._output_channels

    @channels.setter
    def channels(self, channels):
        """Sets the channels of the Output device. Therefore, the current
        stream is closed and a new stream with setted channels is
        initialized."""
        if self.stream.active is True or self.output_buffer.is_active is True:
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self._close_stream()
        max_channel = np.max(channels)
        n_channels = len(channels)
        self.check_settings(n_channels=np.max([n_channels, max_channel+1]))
        self._output_channels = channels
        self.initialize()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        """Sets the dtype of the output buffer. Therefore, the current stream
        is closed and a new stream with setted dtype is initialized."""
        if self.stream.active is True or self.output_buffer.is_active is True:
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self._close_stream()
        self.check_settings(dtype=dtype)
        self._dtype = dtype
        self.initialize()

    def _stop_buffer(self):
        self._output_buffer._stop()

    def _close_stream(self):
        if self.stream is not None:
            self.stream.close()
            self._output_buffer._stop(msg=None)

    def start(self):
        self.output_buffer._start()
        self.output_buffer._is_active.wait()
        super().start()

    def wait(self):
        super().wait()
        self.output_buffer._is_finished.wait()
