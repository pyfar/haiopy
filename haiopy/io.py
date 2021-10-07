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
    def __init__(self, device):
        if isinstance(device, devices._Device):
            self._device = device
        else:
            raise ValueError("Incorrect device, needs to be a"
                             ":py:class:`~haiopy.AudioDevice` object.")

    @abstractmethod
    def start():
        """ This function depends on the use case (playback, recording or
        playback and record) and therefore is implemented in the subclasses.
        """
        pass

    def stop(device):
        """ Immediately terminate the playback/recording."""
        device.abort()
        print("Playback / Recording terminated.")

    def wait():
        pass


class Playback(_AudioIO):
    """Class for playback of signals.
    """
    def __init__(
            self, device, output_channels, repetitions=1,
            output_signal=None):
        """Create a Playback object.

        Parameters
        ----------
        device : [type]
            [description]
        output_channels : [type]
            [description]
        repetitions : int, optional
            [description], by default 1
        """
        super().__init__(device=device)
        # Set properties, check implicitly
        self.output_channels = output_channels
        self.repetitions = repetitions
        self.output_signal = output_signal
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
                f"The cshape of the signal ({signal.cshape}) "
                f"and the number of channels ({self.output_channels.shape}) "
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

    def start(self):
        """Start the playback."""
        if self.output_signal is None:
            raise ValueError("To start the playback, first set an output "
                             "signal.")
        # Extract time data
        data = self.output_signal.time
        # Repeat and append
        append_idx = int(self.repetitions % 1 * self.output_signal.n_samples)
        data_out = np.tile(data, int(self.repetitions))
        data_out = np.append(data_out, data[..., :append_idx], axis=-1)
        self.device.playback(data_out)


class Record(_AudioIO):
    def __init__(
            self,
            device,
            input_channels,
            duration=None,
            fft_norm='amplitude',
            ):
        super().__init__(device=device)
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
            ):
        Record.__init__(
            self,
            device=device,
            input_channels=input_channels)
        Playback.__init__(
            self,
            device=device,
            output_channels=output_channels)

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
