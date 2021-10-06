import haiopy as hp
import pytest
import numpy as np
import pyfar as pf

from unittest.mock import patch

from haiopy import mock_utils
from haiopy import io
from haiopy import Playback
from haiopy import Record
from haiopy import PlaybackRecord


@patch('haiopy.devices._Device', new=mock_utils._Device)
@patch('haiopy.AudioDevice', new=mock_utils.AudioDevice)
class TestPlayback:
    def test__AudioIO_init_error(self):
        """ Test error for instatiating abstract class"""
        with pytest.raises(TypeError):
            io._AudioIO(0)

    def test_init(self):
        device = hp.AudioDevice()
        output_channels = 0
        Playback(device, output_channels)

    def test_init_default_parameters(self):
        device = hp.AudioDevice()
        output_channels = 0
        playback = Playback(device, output_channels)
        assert playback.device == device
        assert playback.output_channels == output_channels
        assert playback.repetitions == 1
        assert playback.output_signal is None

    def test_init_set_parameters(self):
        device = hp.AudioDevice()
        output_channels = 2
        repetitions = 3
        playback = Playback(
            device, output_channels, repetitions)
        assert playback.output_channels == output_channels
        assert playback.repetitions == repetitions

    def test_init_device_error(self):
        device = 0
        channels = 0
        with pytest.raises(ValueError, match="Incorrect device"):
            Playback(device, channels)

    def test_set_output_channels(self):
        device = hp.AudioDevice()
        pb = Playback(device, 0)
        # Check allowed input formats
        for ch in [1, [1, 2], np.array([1, 2]), (1, 2)]:
            pb.output_channels = ch
            assert np.all(pb.output_channels == ch)
            assert len(pb.output_channels.shape) == 1

    def test_set_output_channels_errors(self):
        device = hp.AudioDevice()
        # Error for non integer input
        for ch in [1.1, [1.1, 2], np.array([1.1, 2]), (1.1, 2)]:
            with pytest.raises(ValueError, match="integers"):
                Playback(device, ch)
        # Error for array which is not 2D
        ch = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D array"):
            Playback(device, ch)
        # Error for non unique values
        for ch in [[1, 1], np.array([1, 1]), (1, 1)]:
            with pytest.raises(ValueError, match="unique"):
                Playback(device, ch)

    def test_set_repetitions(self):
        pb = Playback(hp.AudioDevice(), 0)
        pb.repetitions = 2
        assert pb.repetitions == 2
        pb.repetitions = np.inf
        assert pb.repetitions == np.inf

    def test_set_repetitions_errors(self):
        pb = Playback(hp.AudioDevice(), 0)
        for value in ['a', [1], np.array([1, 2])]:
            with pytest.raises(ValueError, match="scalar number"):
                pb.repetitions = value
        for value in [0, -1]:
            with pytest.raises(ValueError, match="positive"):
                pb.repetitions = value

    def test_output_signal_setter(self):
        signal = pf.signals.sine(100, 100)
        pb = Playback(hp.AudioDevice(), 0)
        pb.output_signal = signal

        signal = pf.signals.sine([100, 200], 100)
        pb = Playback(hp.AudioDevice(), [0, 1])
        pb.output_signal = signal

    def test_output_signal_setter_errors(self):
        pass

    def test_start(self):
        pass

    def test_stop(self):
        pass

    def test_wait(self):
        pass

    def test_repetitions(self):
        pass


class TestRecord:
    def test_init_device_error(self):
        device = 0
        channels = 0
        with pytest.raises(ValueError, match="Incorrect device"):
            Record(device, channels)


class TestPlaybackRecord:
    def test_init_device_error(self):
        device = 0
        channels = 0
        with pytest.raises(ValueError, match="Incorrect device"):
            PlaybackRecord(device, channels, channels)
