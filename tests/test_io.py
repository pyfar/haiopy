import unittest
import haiopy as hp
import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt
import unittest

from unittest import mock

from haiopy import mock_utils
from haiopy import io
from haiopy import Playback
from haiopy import Record
from haiopy import PlaybackRecord


@mock.patch('haiopy.devices._Device', new=mock_utils._Device)
@mock.patch('haiopy.AudioDevice', new=mock_utils.AudioDevice)
class TestPlayback():
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
        signal = pf.signals.sine(100, 100)
        playback = Playback(
            device, output_channels, repetitions, signal)
        assert playback.output_channels == output_channels
        assert playback.repetitions == repetitions
        assert playback.output_signal == signal

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
        device = hp.AudioDevice(sampling_rate=44100, dtype=np.float64)
        pb = Playback(device, 0)
        with pytest.raises(ValueError, match="pyfar.Signal"):
            pb.output_signal = 1
        with pytest.raises(ValueError, match="Sampling rate"):
            signal = pf.Signal(np.random.rand(100), 1000, dtype=np.float64)
            pb.output_signal = signal
        with pytest.raises(ValueError, match="Datatypes"):
            signal = pf.Signal(np.random.rand(100), 44100, dtype=np.int32)
            pb.output_signal = signal

    @pytest.mark.parametrize(
        "data, repetitions, expected",
        [(np.array([[0., 1., 2.]]), 1, np.array([[0., 1., 2.]])),
         (np.array([[0., 1., 2.]]), 1.5, np.array([[0., 1., 2., 0.]])),
         (np.array([[0., 1., 2.]]), 1.7, np.array([[0., 1., 2., 0., 1.]])),
         (np.array([[0., 1., 2., 3.]]), 1.5,
         np.array([[0., 1., 2., 3., 0., 1.]])),
         (np.array([[0, 1, 2], [3, 4, 5]]), 1.5,
         np.array([[0., 1., 2., 0.], [3., 4., 5., 3.]]))])
    def test_start(self, data, repetitions, expected):
        device = hp.AudioDevice()
        signal = pf.Signal(data, 44100)
        channels = np.arange(data.shape[0])
        pb = Playback(device, channels, repetitions, signal)
        with mock.patch.object(
                device, 'playback',
                new=lambda x: npt.assert_array_equal(x, expected)):
            pb.start()

    def test_start_missing_signal(self):
        device = hp.AudioDevice()
        pb = Playback(device, 0)
        with pytest.raises(ValueError, match="set an output signal"):
            pb.start()

    def test_stop(self):
        pass

    def test_wait(self):
        pass

    def test_repetitions(self):
        pass


@mock.patch('haiopy.devices._Device', autospec=True)
@mock.patch('haiopy.AudioDevice', autospec=True)
class TestPatching:
    @mock.patch('__main__.isinstance', return_value=True)
    def test_init(self, II, AD, DE):
        device = hp.AudioDevice(0, 44100, 0, 0)
        output_channels = 0
        Playback(device, output_channels)

@mock.patch('__main__.isinstance', return_value=True)
def test_instance(II):
    assert isinstance('a', int)

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
