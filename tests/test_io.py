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
    def test__AudioIO_init_error(self, device_stub):
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
        assert pb._blocking is False

    def test_init_set_parameters(self, device_stub):
        output_channels = 2
        repetitions = 3
        signal = pf.signals.sine(100, 100)
        digital_level = -10
        blocking = True
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(
            device_stub, output_channels, repetitions, signal,
            digital_level, blocking)
        assert pb._device == device_stub
        assert pb._output_channels == output_channels
        assert pb._repetitions == repetitions
        assert pb._output_signal == signal
        assert pb._digital_level == digital_level
        assert pb._blocking is blocking

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

    @pytest.mark.parametrize("channels", [1, [1, 2], np.array([1, 2]), (1, 2)])
    def test_output_channels_setter(self, device_stub, channels):
        pb = Playback(device_stub, 0)
        pb.output_channels = channels
        assert np.all(pb._output_channels == channels)
        assert len(pb._output_channels.shape) == 1

    @pytest.mark.parametrize(
        "channels", [1.1, [1.1, 2], np.array([1.1, 2]), (1.1, 2)])
    def test_output_channels_setter_errors(self, device_stub, channels):
        # non integer input
        with pytest.raises(ValueError, match="integers"):
            Playback(device_stub, channels)

    def test_output_channels_setter_errors_2D(self, device_stub):
        # array which is not 2D
        ch = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D array"):
            Playback(device_stub, ch)

    @pytest.mark.parametrize("channels", [[1, 1], np.array([1, 1]), (1, 1)])
    def test_output_channels_setter_errors_unique(self, device_stub, channels):
        with pytest.raises(ValueError, match="unique"):
            Playback(device_stub, channels)

    def test_repetitions_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._repetitions = 2
        assert pb.repetitions == 2

    def test_repetitions_setter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb.repetitions = 2
        assert pb._repetitions == 2

    @pytest.mark.parametrize("reps", ['a', [1], np.array([1, 2])])
    def test_repetitions_setter_errors_scalar_number(self, device_stub, reps):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="scalar number"):
            pb.repetitions = reps

    @pytest.mark.parametrize("reps", [0, -1])
    def test_repetitions_setter_errors_positive(self, device_stub, reps):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="positive"):
            pb.repetitions = reps

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

    def test_output_signal_setter_errors(self, device_stub):
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

    def test_blocking_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._blocking = 1
        assert pb.blocking == 1

    def test_blocking_setter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb.blocking = True
        assert pb._blocking is True

    def test_blocking_setter_errors(self, device_stub):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="True or False"):
            pb.blocking = 1

    def test_digital_level_getter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb._digital_level = 'a'
        assert pb.digital_level == 'a'

    def test_digital_level_setter(self, device_stub):
        pb = Playback(device_stub, 0)
        pb.digital_level = -10
        assert pb._digital_level == -10

    @pytest.mark.parametrize("level", ['a', (1, 2), [1, 2], np.array([1, 2])])
    def test_digital_level_setter_errors_number(self, device_stub, level):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="single number"):
            pb.digital_level = level

    def test_digital_level_setter_errors_positive(self, device_stub):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="<= 0"):
            pb.digital_level = 10

    @pytest.mark.parametrize(
        "data, repetitions, expected",
        [(np.array([[0., 1., 2.]]), 1, np.array([[0., 1., 2.]])),
         (np.array([[0., 1., 2.]]), 1.5, np.array([[0., 1., 2., 0.]])),
         (np.array([[0., 1., 2.]]), 1.7, np.array([[0., 1., 2., 0., 1.]])),
         (np.array([[0., 1., 2., 3.]]), 1.5,
         np.array([[0., 1., 2., 3., 0., 1.]])),
         (np.array([[0, 1, 2], [3, 4, 5]]), 1.5,
         np.array([[0., 1., 2., 0.], [3., 4., 5., 3.]]))])
    def test_start_repetitions(self, device_stub, data, repetitions, expected):
        signal = pf.Signal(data, 44100)
        channels = np.arange(data.shape[0])
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(device_stub, channels, repetitions, signal)
        with mock.patch.object(
                device_stub, 'playback',
                new=lambda x: npt.assert_array_equal(x, expected)):
            pb.start()

    def test_start_digital_level(self, device_stub):
        signal = pf.Signal([1, 2, 3], 44100)
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(
            device=device_stub, output_channels=0, repetitions=2,
            output_signal=signal, digital_level=-20)
        expected = np.array([[.1, .2, .3, .1, .2, .3]])
        with mock.patch.object(
                device_stub, 'playback',
                new=lambda x: npt.assert_allclose(x, expected, atol=1e-15)):
            pb.start()

    def test_start_missing_signal(self, device_stub):
        pb = Playback(device_stub, 0)
        with pytest.raises(ValueError, match="set an output signal"):
            pb.start()

    def test_start_blocking(self, device_stub):
        signal = pf.Signal([1, 2, 3], 44100)
        device_stub.sampling_rate = signal.sampling_rate
        device_stub.dtype = signal.dtype
        pb = Playback(device_stub, 0, output_signal=signal, blocking=True)
        with mock.patch.object(device_stub, 'wait') as wait_mock:
            pb.start()
            wait_mock.assert_called_once()

    def test_stop(self, device_stub):
        pb = Playback(device_stub, 0)
        with mock.patch.object(device_stub, 'abort') as abort_mock:
            pb.stop()
            abort_mock.assert_called_once()


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
