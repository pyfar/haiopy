"""
Playback and recording functionality including classes and convenience
functions.
"""

import numpy as np
import pyfar as pf

from . import devices
from abc import ABC, abstractmethod


class _AudioIO(ABC):
    """Abstract class for playback and recording.

    This class holds all the methods and properties that are common to its
    three sub-classes :py:class:`Playback`, :py:class:`Record`, and
    :py:class:`PlaybackRecord`.
    """
    def __init__(self, device, blocking):
        if isinstance(device, devices._Device):
            self._device = device
            self.blocking = blocking
        else:
            raise ValueError("Incorrect device, needs to be a"
                             ":py:class:`~haiopy.AudioDevice` object.")

    @property
    def blocking(self):
        """Boolean parameter blocking."""
        return self._blocking

    @blocking.setter
    def blocking(self, value):
        """Set blocking parameter to True or False."""
        if isinstance(value, bool):
            self._blocking = value
        else:
            raise ValueError("Blocking needs to be True or False.")

    @abstractmethod
    def start():
        """ This function depends on the use case (playback, recording or
        playback and record) and therefore is implemented in the subclasses.
        """
        pass

    def stop(self):
        """Immediately terminate the playback/recording."""
        self._device.abort()
        print("Playback / Recording terminated.")

    def wait(self):
        """Wait until playback/recording is finished."""
        self._device.wait()


class Playback(_AudioIO):
    """Class for playback of signals.
    """
    def __init__(
            self, device, output_channels, repetitions=1,
            output_signal=None, digital_level=0., blocking=False):
        """Create a Playback object.

        Parameters
        ----------
        device : haiopy.AudioDevice
            The device to play the signal.
        output_channels : array-like
            The output channels. The parameter can be a single number, list,
            tuple or a 1D array with unique values.
        repetitions : int, optional
            Number of repitions, the default ``1``.
        output_signal : pyfar.Signal, optional
            The signal to be played. The default ``None``, requires the signal
            to be set before :py:func:`~play` is called.
        digital_level : float, optional
            Digital output level (the digital output amplification) in dB
            referenced to an amplitude of 1, so only levels <= 0 dB can be set.
            The default is ``0``, which results in an unmodified playback.
        blocking : bool, optional
            If ``True`` :py:func:`~play` function doesn’t return until the
            playback is finished. The default is ``False``.
        """
        super().__init__(device=device, blocking=blocking)
        # Set properties, check implicitly
        self.output_channels = output_channels
        self.repetitions = repetitions
        self.output_signal = output_signal
        self.digital_level = digital_level
        # Initialize device
        self.device.initialize_playback(self.output_channels)

    @property
    def device(self):
        """Output device."""
        return self._device

    @property
    def output_signal(self):
        """``pyfar.Signal`` to be played back."""
        return self._output_signal

    @output_signal.setter
    def output_signal(self, signal):
        """Set ``pyfar.Signal`` to be played back."""
        if signal is None:
            self._output_signal = signal
        elif not isinstance(signal, pf.Signal):
            raise ValueError("Output signal needs to be a pyfar.Signal.")
        elif signal.sampling_rate != self.device.sampling_rate:
            raise ValueError(
                f"Sampling rates of the signal ({signal.sampling_rate}) "
                f"and the device ({self.device.sampling_rate}) "
                f"do not match.")
        elif signal.dtype != self.device.dtype:
            raise ValueError(
                f"Datatypes of the signal ({signal.dtype}) "
                f"and the device ({self.device.dtype}) "
                f"do not match.")
        elif signal.cshape != self.output_channels.shape:
            raise ValueError(
                f"The shapes of the signal ({signal.cshape}) "
                f"and the channels ({self.output_channels.shape}) "
                f"do not match.")
        else:
            self._output_signal = signal

    @property
    def output_channels(self):
        """Output channels."""
        return self._output_channels

    @output_channels.setter
    def output_channels(self, channels):
        """Set output_channels parameter. It can be a single number, list,
        tuple or a 1D array with unique values.
        """
        channels_int = np.unique(channels).astype(int)
        if np.atleast_1d(channels).shape != channels_int.shape:
            raise ValueError("Output_channels must be a single number, list, "
                             "tuple or a 1D array with unique values.")
        elif not np.all(channels == channels_int):
            raise ValueError("Parameter output_channels must contain only"
                             "integers.")
        else:
            self._output_channels = channels_int

        # Initialize device
        self.device.initialize_playback(self._output_channels)

    @property
    def repetitions(self):
        """Number of repetitions of the playback."""
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value):
        """Set the number of repetitions of the playback. ``repetitions`` can
        be set to a decimal number. The default is ``1``,
        which is a single playback."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError("Repetitions must be a scalar number.")
        if value > 0:
            self._repetitions = value
        else:
            raise ValueError("Repetitions must be positive.")

    @property
    def digital_level(self):
        """Digital output level in dB."""
        return self._digital_level

    @digital_level.setter
    def digital_level(self, value):
        """Set the digital output level in dB. The level is referenced to an
         amplitude of 1, so only levels <= 0 dB can be set."""
        try:
            level = float(value)
        except (ValueError, TypeError):
            raise ValueError("The digital level must be single number.")
        if level <= 0:
            self._digital_level = level
        else:
            raise ValueError("The digital level must be <= 0.")

    def start(self):
        """Start the playback."""
        if self.output_signal is None:
            raise ValueError("To start the playback, first set an output "
                             "signal.")
        # Extract time data
        data = self.output_signal.time
        # Amplification / Attenuations
        data = data * 10**(self._digital_level/20)
        # Repeat and append
        append_idx = int(self.repetitions % 1 * self.output_signal.n_samples)
        data_out = np.tile(data, int(self.repetitions))
        data_out = np.append(data_out, data[..., :append_idx], axis=-1)
        self.device.playback(data_out)
        # Block
        if self._blocking:
            self.wait()


class Record(_AudioIO):
    def __init__(
            self,
            device,
            input_channels,
            duration=None,
            fft_norm='amplitude',
            blocking=False
            ):
        super().__init__(device=device, blocking=blocking)
        self._input_signal = None
        self.input_channels = input_channels

    @property
    def input_channels(self):
        return self._input_channels

    @input_channels.setter
    def input_channels(self, channels):
        self._input_channels = channels

    @property
    def input_signal(self):
        return self._input_signal

    def start():
        """ This function depends on the use case (playback, recording or
        playback and record) and therefore is implemented in the subclasses.
        """
        pass


class PlaybackRecord(Playback, Record):
    def __init__(
            self,
            device,
            input_channels,
            output_channels,
            blocking=False
            ):
        Record.__init__(
            self,
            device=device,
            input_channels=input_channels,
            blocking=blocking)
        Playback.__init__(
            self,
            device=device,
            output_channels=output_channels,
            blocking=blocking)

    def start():
        """ This function depends on the use case (playback, recording or
        playback and record) and therefore is implemented in the subclasses.
        """
        pass


def playback():
    pass


def record():
    pass


def playback_record():
    pass
