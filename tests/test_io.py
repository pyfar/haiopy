import pytest
import numpy as np
import pyfar as pf
import numpy.testing as npt

from unittest import mock

from haiopy import io
from haiopy import Playback
from haiopy import Record
from haiopy import PlaybackRecord


class TestPlayback():
    @mock.patch('haiopy.devices._Device', autospec=True)
    def test__AudioIO_init_error(self, de):
        """ Test error for instatiating abstract class"""
        with pytest.raises(TypeError):
            io._AudioIO(0)

    @mock.patch('haiopy.io.isinstance', return_value=True)
    def test_init(self, isinstance_mock, device_stub):
        output_channels = 0
        Playback(device_stub, output_channels)

    def test_init_default_parameters(self, device_stub):
        output_channels = 0
        pb = Playback(device_stub, output_channels)
        assert pb._device == device_stub
        assert pb._output_channels == output_channels
        assert pb._repetitions == 1
        assert pb._output_signal is None

    def test_init_set_parameters(self, device_stub):
        output_channels = 2
        repetitions = 3
        signal = pf.signals.sine(100, 100)
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(device_stub, output_channels, repetitions, signal)
        assert pb._device == device_stub
        assert pb._output_channels == output_channels
        assert pb._repetitions == repetitions
        assert pb._output_signal == signal

    def test_init_device_error(self):
        device = 0
        channels = 0
        with pytest.raises(ValueError, match="Incorrect device"):
            Playback(device, channels)

    def test_device_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._device = 1
        assert pb.device == 1

    def test_output_channels_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._output_channels = 1
        assert pb.output_channels == 1

    def test_output_channels_setter(self, device_stub):
        pb = Playback(device_stub, 0)
        # Check allowed input formats
        for ch in [1, [1, 2], np.array([1, 2]), (1, 2)]:
            pb.output_channels = ch
            assert np.all(pb._output_channels == ch)
            assert len(pb._output_channels.shape) == 1

    def test_output_channels_errors(self, device_stub):
        # non integer input
        for ch in [1.1, [1.1, 2], np.array([1.1, 2]), (1.1, 2)]:
            with pytest.raises(ValueError, match="integers"):
                Playback(device_stub, ch)
        # array which is not 2D
        ch = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D array"):
            Playback(device_stub, ch)
        # non unique values
        for ch in [[1, 1], np.array([1, 1]), (1, 1)]:
            with pytest.raises(ValueError, match="unique"):
                Playback(device_stub, ch)

    def test_repetitions_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._repetitions = 2
        assert pb.repetitions == 2

    def test_repetitions_setter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb.repetitions = 2
        assert pb._repetitions == 2

    def test_repetitions_errors(self, device_stub):
        pb = Playback(device_stub, 0)
        for value in ['a', [1], np.array([1, 2])]:
            with pytest.raises(ValueError, match="scalar number"):
                pb.repetitions = value
        for value in [0, -1]:
            with pytest.raises(ValueError, match="positive"):
                pb.repetitions = value

    def test_output_signal_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._output_signal = 1
        assert pb.output_signal == 1

    def test_output_signal_setter(self, device_stub):
        # One channel
        signal = pf.signals.sine(100, 100)
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(device_stub, 0)
        pb.output_signal = signal
        assert pb._output_signal == signal
        # Two channels
        signal = pf.signals.sine([100, 200], 100)
        pb = Playback(device_stub, [0, 1])
        pb.output_signal = signal
        assert pb._output_signal == signal

    def test_output_signal_errors(self, device_stub):
        device_stub.sampling_rate = 44100
        device_stub.dtype = np.float64
        pb = Playback(device_stub, 0)
        # Signal
        with pytest.raises(ValueError, match="pyfar.Signal"):
            pb.output_signal = 1
        # sampling_rate
        with pytest.raises(ValueError, match="Sampling rate"):
            signal = pf.Signal(
                np.random.rand(100), 1000, dtype=device_stub.dtype)
            pb.output_signal = signal
        # dtype
        with pytest.raises(ValueError, match="Datatypes"):
            signal = pf.Signal(np.random.rand(100), 44100, dtype=np.int32)
            pb.output_signal = signal
        # shape
        with pytest.raises(ValueError, match="shapes"):
            signal = pf.Signal(
                np.array([[1, 2, 3], [4, 5, 6]]), 44100)
            pb.output_signal = signal
        with pytest.raises(ValueError, match="shapes"):
            pb = Playback(device_stub, [0, 1])
            signal = pf.Signal(np.array([[1, 2, 3]]), 44100)
            pb.output_signal = signal

    def test_blocking_getter(self):
        pass

    def test_blocking_setter(self):
        pass

    def test_blocking_errors(self):
        pass

    def test_digital_level_getter(self):
        pass

    def test_digital_levelsetter(self):
        pass

    def test_digital_level_errors(self):
        pass

    @pytest.mark.parametrize(
        "data, repetitions, expected",
        [(np.array([[0., 1., 2.]]), 1, np.array([[0., 1., 2.]])),
         (np.array([[0., 1., 2.]]), 1.5, np.array([[0., 1., 2., 0.]])),
         (np.array([[0., 1., 2.]]), 1.7, np.array([[0., 1., 2., 0., 1.]])),
         (np.array([[0., 1., 2., 3.]]), 1.5,
         np.array([[0., 1., 2., 3., 0., 1.]])),
         (np.array([[0, 1, 2], [3, 4, 5]]), 1.5,
         np.array([[0., 1., 2., 0.], [3., 4., 5., 3.]]))])
    def test_start(self, device_stub, data, repetitions, expected):
        signal = pf.Signal(data, 44100)
        channels = np.arange(data.shape[0])
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(device_stub, channels, repetitions, signal)
        with mock.patch.object(
                device_stub, 'playback',
                new=lambda x: npt.assert_array_equal(x, expected)):
            pb.start()

    def test_start_missing_signal(self, device_stub):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="set an output signal"):
            pb.start()

    def test_stop(self, device_stub):
        pb = Playback(device_stub, 0)
        with mock.patch.object(device_stub, 'abort') as abort_mock:
            pb.stop()
            abort_mock.assert_called_with()

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
