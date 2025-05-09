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
    """Class implementing an output audio device.

    The implementation is based on python-sounddevice and portaudio.

    """

    def __init__(
            self,
            identifier=sd.default.device['output'],
            sampling_rate=44100,
            block_size=512,
            channels=[0],
            dtype='float32',
            output_buffer=None,
        ):

        # First check the settings before continuing
        n_channels = len(channels)
        sd.check_output_settings(
            device=identifier,
            channels=np.max([n_channels, np.max(channels)+1]),
            dtype=dtype,
            samplerate=sampling_rate)

        # Init base class
        super().__init__(
            identifier=identifier,
            sampling_rate=sampling_rate,
            block_size=block_size,
            dtype=dtype)

        # Set the output channel mapping which is specific for each host api
        self._output_channel_mapping = OutputChannelMapping(
            channels, self.max_channels_output, self.host_api)

        # Set the output buffer which will be consumed in the callback
        # function. If no buffer is given, an empty buffer is created.
        self._output_buffer = None
        if output_buffer is None:
            output_buffer = EmptyBuffer(
                block_size, n_channels, sampling_rate)
        self.output_buffer = output_buffer

        # Initialize the device
        self.initialize()

    @property
    def name(self) -> str:
        """The name of the device."""
        return sd.query_devices(self.identifier)['name']

    @property
    def host_api(self) -> str:
        """The host API used by the device."""
        return sd.query_hostapis(
            sd.query_devices(self.identifier)['hostapi'])['name']

    def check_settings(
            self,
            n_channels: list[int] = None,
            sampling_rate: int = None,
            dtype: np.int8 | np.int16 | np.int32 | np.float32 = None,
            extra_settings: sd.CoreAudioSettings | sd.AsioSettings | None = None):
        """Check if settings are compatible with the physical device.

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
            If the settings are incompatible with the device an exception is raised.
        """
        sd.check_output_settings(
            device=self.identifier,
            channels=n_channels,
            dtype=dtype,
            extra_settings=extra_settings,
            samplerate=sampling_rate)

    @property
    def output_channel_mapping(self):
        """The channel mapping of the device."""
        return self._output_channel_mapping

    @property
    def output_channels(self) -> list[int]:
        """The output channels of the device.
        """
        return self.output_channel_mapping.channels

    @output_channels.setter
    def output_channels(self, channels):
        """Set the output channels of the device.

        Parameters
        ----------
        channels : list
            The output channels to be used by the device.
        """

        if self._stream_active() or self._buffer_active():
            raise ValueError("The device is currently in use and needs to be closed first")
        self._close_stream()
        self.output_channel_mapping.channels = channels

    @property
    def n_channels_output(self):
        """The total number of output channels.

        Returns
        -------
        int
            The number of output channels
        """
        return self.output_channel_mapping.n_channels_used

    @property
    def max_channels_output(self):
        """The number of output channels supported by the device."""
        return sd.query_devices(
            self.identifier, 'output')['max_output_channels']

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

    def initialize(self) -> None:
        """Initialize and open the playback stream.
        This will set the device to active, potentially blocking the soundcard for other
        applications. Playback is not yet started.
        """

        ostream = sd.OutputStream(
            samplerate=self.sampling_rate,
            blocksize=self.block_size,
            device=self.identifier,
            channels=self.output_channel_mapping.n_channels_mapping,
            dtype=self._dtype,
            callback=self.output_callback,
            finished_callback=self._finished_callback,
            extra_settings=self.output_channel_mapping.extra_settings,
        )
        self._stream = ostream

    def initialize_buffer(self) -> None:
        """Initialize the output buffer.
        Starts the buffer and waits for it to be active.
        """
        self.output_buffer._start()
        self.output_buffer._is_active.wait()

    @property
    def output_buffer(self) -> type[_Buffer] | None:
        """The output buffer which is consumed by the device.
        """
        return self._output_buffer

    @output_buffer.setter
    def output_buffer(self, buffer: type[_Buffer]):
        """Set the output buffer which is consumed by the device.
        The number of channels and the block size need to match the device settings.
        """
        if self._stream_active() or self._buffer_active():
            raise ValueError(
                "The device is currently in use and needs to be closed first")

        if buffer.block_size != self.block_size:
            raise ValueError(
                "The buffer's block size does not match. ",
                f"Needs to be {self.block_size}")

        if buffer.n_channels != self.n_channels_output:
            raise ValueError(
                "The buffer's channel number does not match the channel "
                f"mapping. A number of {self.n_channels_output} are currently "
                f"used. These are {self.output_channels}")

        self._output_buffer = buffer

    def _buffer_active(self) -> bool:
        """Check if the output buffer is active."""
        return False if self._output_buffer is None else self.output_buffer.is_active

    @_Device.block_size.setter
    def block_size(self, block_size: int):
        """Sets the block size of the device and the output buffer.
        Any open stream needs to be closed an re-opened. The output buffer is reset.
        """
        if self._stream_active() or self._buffer_active():
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self._close_stream()
        self._block_size = block_size
        self.output_buffer.block_size = block_size
        self.initialize()

    @_Device.sampling_rate.setter
    def sampling_rate(self, sampling_rate: int):
        """Sets the sampling rate of the device.
        Any open stream needs to be closed an re-opened.
        """
        if self._stream_active() or self._buffer_active():
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self.check_settings(sampling_rate=sampling_rate)
        self._close_stream()
        self._sampling_rate = sampling_rate
        self.output_buffer.sampling_rate = sampling_rate
        self.initialize()

    @_Device.dtype.setter
    def dtype(self, dtype):
        """Sets the dtype of the device buffer (portaudio specific).
        Any open stream needs to be closed an re-opened.
        """
        if self._stream_active() or self._buffer_active():
            raise ValueError(
                "The device is currently in use and needs to be closed first")
        self._close_stream()
        self.check_settings(dtype=dtype)
        self._dtype = dtype
        self.initialize()

    def _stop_buffer(self):
        """Stop the output buffer iteration.
        This will raise a StopIteration exception in the buffer.
        """
        self._output_buffer._stop()

    def _close_stream(self):
        """Close the steam and stop the output buffer.
        This will release the soundcard lock.
        """
        if self.stream is not None:
            self.stream.close()
            self._output_buffer._stop(msg=None)

    def start(self):
        """Start playback."""
        if self._stream is None:
            self.initialize()
        self.output_buffer._start()
        self.output_buffer._is_active.wait()
        super().start()

    def wait(self):
        """Wait for the device to finish playback."""
        super().wait()
        self.output_buffer._is_finished.wait()
