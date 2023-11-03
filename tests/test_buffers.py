import numpy as np
import numpy.testing as npt
from haiopy.buffers import _Buffer, SignalBuffer
from haiopy.buffers import SineGenerator, NoiseGenerator
import pytest
import pyfar as pf
from scipy import signal


def test_buffer_block_size():

    block_size = 512
    buffer = _Buffer(block_size)

    assert buffer._block_size == block_size

    assert buffer.block_size == block_size

    new_block_size = 128
    buffer.block_size = int(new_block_size)
    assert buffer._block_size == new_block_size

    with pytest.raises(
            ValueError, match='The block size needs to be an integer'):
        buffer.block_size = float(10)

    with pytest.raises(
            ValueError, match='The block size needs to be an integer'):
        _Buffer(float(10))


def test_buffer_state():
    block_size = 512

    # create new buffer
    buffer = _Buffer(block_size)
    buffer._start()
    assert buffer.is_active is True

    # check if the correct error is raised on stopping
    with pytest.raises(StopIteration, match="iteration stopped"):
        buffer._stop()

    assert buffer.is_active is False

    # The buffer will be automatically set to be active after the first call
    # to the __next__ method
    # The pytest raises is required here, as the sub-class specific next
    # method is an abstract class method
    with pytest.raises(NotImplementedError):
        next(buffer)
        assert buffer._is_active is True

        # check_if_active() raises an exception if buffer is active
        with pytest.raises(BufferError, match="needs to be inactive"):
            buffer.check_if_active()

    # check iterator dunder
    assert iter(buffer) == buffer


def test_signal_buffer():
    sampling_rate = 44100
    n_blocks = 10
    block_size = 512
    n_samples = block_size*n_blocks
    sine = pf.signals.sine(
        440, n_samples, amplitude=[1, 1], sampling_rate=sampling_rate)

    with pytest.raises(ValueError, match='one-dimensional'):
        SignalBuffer(
            block_size,
            pf.Signal(np.zeros((2, 3, block_size), 'float32'), sampling_rate))

    with pytest.raises(ValueError, match='must be a pyfar.Signal'):
        SignalBuffer(block_size, [1, 2, 3])

    buffer = SignalBuffer(block_size, sine)

    assert buffer._n_blocks == n_blocks
    assert buffer.n_blocks == n_blocks

    # test sampling rate getter
    assert buffer.sampling_rate == sampling_rate

    # test number of channels
    assert buffer.n_channels == 2

    # check if the initial index s correct
    assert buffer._index == 0
    assert buffer.index == 0

    # check if the data arrays are correct
    npt.assert_array_equal(buffer._data.time, sine.time)
    npt.assert_array_equal(buffer.data.time, sine.time)

    # check if the data strides are correct
    strided_buffer_data = np.lib.stride_tricks.as_strided(
        sine.time, (*sine.cshape, n_blocks, block_size))
    npt.assert_array_equal(
        buffer._strided_data, strided_buffer_data)

    # check first step
    block_data = next(buffer)
    npt.assert_array_equal(block_data, strided_buffer_data[..., 0, :])

    # check second step
    block_data = next(buffer)
    npt.assert_array_equal(block_data, strided_buffer_data[..., 1, :])

    # check if a error is raised if the end of the buffer is reached
    with pytest.raises(StopIteration, match="buffer is empty"):
        while True:
            next(buffer)

    # test the looping blocks
    buffer = SignalBuffer(block_size, sine)
    for idx, block in enumerate(buffer):
        assert buffer.is_active is True
        npt.assert_array_equal(
            block, strided_buffer_data[..., idx, :])

    # check if state is set to inactive after loop finished
    assert buffer.is_active is False
    assert buffer.is_finished is True


def test_signal_buffer_padding():
    sampling_rate = 44100
    n_samples = 800

    n_blocks = 2
    block_size = 512
    sine = pf.signals.sine(
        440, n_samples, amplitude=[1], sampling_rate=sampling_rate)

    buffer = SignalBuffer(block_size, sine)

    assert buffer.data.n_samples == n_blocks*block_size

    expected_data = np.concatenate((
        np.squeeze(sine.time),
        np.zeros(n_blocks*block_size-n_samples, dtype=float)))

    npt.assert_equal(np.squeeze(buffer.data.time), expected_data)


def test_writing_signal_buffer():
    sampling_rate = 44100
    block_size = 512

    block_data = np.atleast_2d(np.arange(block_size))

    sig = pf.Signal(np.zeros(block_size, dtype='float32'), sampling_rate)
    buffer = SignalBuffer(block_size, sig)

    next(buffer)[:] = block_data

    # we need to stop the buffer which raises a StopIteration error
    with pytest.raises(StopIteration):
        buffer._stop()
    np.testing.assert_array_equal(buffer.data.time, block_data)


def test_signal_buffer_updates():
    sampling_rate = 44100
    n_blocks = 10
    block_size = 512
    n_samples = block_size*n_blocks
    noise = pf.signals.noise(
        n_samples, rms=[1, 1], sampling_rate=sampling_rate)
    sine = pf.signals.sine(
        440, n_samples, amplitude=[1, 1], sampling_rate=sampling_rate)

    # Create a new buffer
    buffer = SignalBuffer(block_size, noise)

    # Set a new signal as data for the buffer
    buffer.data = sine
    npt.assert_array_equal(buffer._data.time, sine.time)
    npt.assert_array_equal(buffer.data.time, sine.time)

    # The new block size is 4 times smaller than the old one
    new_block_size = 128
    buffer.block_size = new_block_size

    # The data itself is not touched in this case
    npt.assert_array_equal(buffer._data.time, sine.time)
    npt.assert_array_equal(buffer.data.time, sine.time)

    # Stride the array with the new block size
    # The new number of blocks is an integer multiple of the old block size
    new_n_blocks = n_samples // new_block_size
    strided_buffer_data = np.lib.stride_tricks.as_strided(
        sine.time, (*sine.time.shape[:-1], new_n_blocks, new_block_size))
    npt.assert_array_equal(
        buffer._strided_data, strided_buffer_data)

    # Check if Errors are raised when buffer is in use
    next(buffer)
    assert buffer.is_active is True

    # Setting the block size is not allowed if the buffer is active
    with pytest.raises(BufferError, match="needs to be inactive"):
        buffer.block_size = 512

    # Setting and getting the data is not allowed if the buffer is active
    with pytest.raises(BufferError, match="needs to be inactive"):
        buffer.data = sine

    with pytest.raises(BufferError, match="needs to be inactive"):
        buffer.data


def test_SineGenerator():
    frequency = 440
    block_size = 512
    amplitude = 1
    sampling_rate = 44100

    sine = SineGenerator(frequency,
                         block_size,
                         amplitude,
                         sampling_rate)

    # test getters
    assert sine.frequency == frequency
    assert sine.amplitude == amplitude
    assert sine.sampling_rate == sampling_rate
    assert sine.phase == 0

    # check if sine generator is not active yet
    assert sine.is_active is False

    # check first step
    omega = 2 * np.pi * frequency
    phase = 0
    sine_data = amplitude * np.sin(
            omega*np.arange(block_size)/sampling_rate+phase)
    block_data = next(sine)
    npt.assert_array_equal(block_data, sine_data)

    # check second step
    phase = omega * (block_size/sampling_rate)
    assert sine.phase == phase
    sine_data = amplitude * np.sin(
            omega*np.arange(block_size)/sampling_rate+phase)
    block_data = next(sine)
    npt.assert_array_equal(block_data, sine_data)

    # check if sine generator is active now
    assert sine.is_active is True


def test_SineGenerator_updates():
    frequency = 440
    block_size = 512
    amplitude = 1
    sampling_rate = 44100

    sine = SineGenerator(frequency,
                         block_size,
                         amplitude,
                         sampling_rate)

    # Update Parameters
    new_frequency = 500
    sine.frequency = new_frequency
    assert sine.frequency == new_frequency

    new_block_size = 128
    sine.block_size = new_block_size
    assert sine.block_size == new_block_size

    new_amplitude = 2
    sine.amplitude = new_amplitude
    assert sine.amplitude == new_amplitude

    new_sampling_rate = 48000
    sine.sampling_rate = new_sampling_rate
    assert sine.sampling_rate == new_sampling_rate

    # check updated date
    updated_block_data = next(sine)
    omega = 2 * np.pi * new_frequency
    phase = 0
    sine_data = new_amplitude * np.sin(
            omega*np.arange(new_block_size)/new_sampling_rate+phase)
    npt.assert_array_equal(updated_block_data, sine_data)

    # Test setting phase to 0 when updating parameters
    assert sine.phase == omega*(new_block_size / new_sampling_rate)
    # we need to stop the SineGenerator which raises a StopIteration error
    with pytest.raises(StopIteration):
        sine._stop()
    sine.frequency = 1000
    assert sine.phase == 0

    # Check if Errors are raised when SineGenerator is in use
    next(sine)
    assert sine.is_active is True

    # Setting the Parameters is not allowed if the SineGenerator is active
    with pytest.raises(BufferError, match="needs to be inactive"):
        sine.frequency = 400
    with pytest.raises(BufferError, match="needs to be inactive"):
        sine.block_size = 256
    with pytest.raises(BufferError, match="needs to be inactive"):
        sine.amplitude = 3
    with pytest.raises(BufferError, match="needs to be inactive"):
        sine.sampling_rate = 24000


def test_NoiseGenerator():
    block_size = 512
    seed = 10

    with pytest.raises(ValueError, match="spectrum is 'invalid'"):
        NoiseGenerator(block_size, spectrum="invalid")

    noise = NoiseGenerator(block_size, seed=seed)

    # test getters with default
    assert noise.block_size == block_size
    assert noise.spectrum == "white"
    assert noise.rms == 1
    assert noise.sampling_rate == 44100
    assert noise.seed == 10

    # check if noise generator is not active yet
    assert noise.is_active is False

    # Get Random Generator Object with seed = 10
    rng = np.random.default_rng(seed)
    # check first block
    noise_data = rng.standard_normal(block_size)
    # level the noise
    noise_data /= np.sqrt(np.mean(noise_data**2))
    block_data = next(noise)
    npt.assert_array_equal(block_data, noise_data)

    # check if noise generator is active now
    assert noise.is_active is True

    # check second block

    noise_data = rng.standard_normal(block_size)
    # level the noise
    noise_data /= np.sqrt(np.mean(noise_data**2))
    block_data = next(noise)
    npt.assert_array_equal(block_data, noise_data)


def test_NoiseGenerator_updates():
    block_size = 512
    noise = NoiseGenerator(block_size)

    # check if noise generator is not active yet
    assert noise.is_active is False

    # Update Parameters
    new_block_size = 1028
    noise.block_size = new_block_size
    assert noise.block_size == new_block_size

    # set invalid spectrum
    with pytest.raises(ValueError, match="spectrum is 'invalid'"):
        noise.spectrum = "invalid"
    new_spectrum = "pink"
    noise.spectrum = new_spectrum
    assert noise.spectrum == new_spectrum

    new_rms = 0.5
    noise.rms = new_rms
    assert noise.rms == new_rms

    new_sampling_rate = 48000
    noise.sampling_rate = new_sampling_rate
    assert noise.sampling_rate == new_sampling_rate

    new_seed = 5
    noise.seed = new_seed
    assert noise.seed == new_seed

    # Check updated data and pink spectrum
    rng = np.random.default_rng(5)
    noise_data = rng.standard_normal(new_block_size)
    # Apply Filter for Pink Noise
    B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    A = [1, -2.494956002, 2.017265875, -0.522189400]
    zi = signal.lfilter_zi(B, A)
    noise_data, zi2 = signal.lfilter(B, A, noise_data, zi=zi)
    # level the noise
    rms_current = np.sqrt(np.mean(noise_data**2))
    noise_data = noise_data / rms_current * new_rms

    block_data = next(noise)
    npt.assert_array_equal(block_data, noise_data)

    """Apply filter to second block of data with initial conditions from
    first block."""
    noise_data2 = rng.standard_normal(new_block_size)
    noise_data2, _ = signal.lfilter(B, A, noise_data2, zi=zi2)
    # level the noise
    rms_current = np.sqrt(np.mean(noise_data2**2))
    noise_data2 = noise_data2 / rms_current * new_rms

    block_data2 = next(noise)
    npt.assert_array_equal(block_data2, noise_data2)

    # Check if Errors are raised when NoiseGenerator is in use
    assert noise.is_active is True

    # Setting the Parameters is not allowed if the NoiseGenerator is active
    with pytest.raises(BufferError, match="needs to be inactive"):
        noise.spectrum = "white"
    with pytest.raises(BufferError, match="needs to be inactive"):
        noise.block_size = 256
    with pytest.raises(BufferError, match="needs to be inactive"):
        noise.rms = 3
    with pytest.raises(BufferError, match="needs to be inactive"):
        noise.sampling_rate = 24000
    with pytest.raises(BufferError, match="needs to be inactive"):
        noise.seed = 123


def test_sampling_rate_setter():
    # Test setting the sampling rate, resampling the Signal and updating data
    block_size = 512
    sampling_rate = 44100
    sine = pf.signals.sine(440, 4*block_size, sampling_rate=sampling_rate)
    buffer = SignalBuffer(block_size, sine)
    assert buffer.sampling_rate == 44100

    new_sampling_rate = sampling_rate*2
    with pytest.warns(UserWarning, match="Resampling the input Signal"):
        buffer.sampling_rate = new_sampling_rate
    resampled_sig = pf.dsp.resample(sine, new_sampling_rate)
    assert buffer.sampling_rate == 88200
    assert buffer.n_blocks == 8
    npt.assert_allclose(buffer.data.time, resampled_sig.time)


def test_reset_index():
    # Test reset_index method
    block_size = 512
    sampling_rate = 44100
    sine = pf.signals.sine(440, 4*block_size, sampling_rate=sampling_rate)
    buffer = SignalBuffer(block_size, sine)
    [next(buffer), next(buffer), next(buffer)]
    assert buffer.index == 3
    # reset_index() is not supposed to raise StopIteration
    buffer.reset_index()
    assert buffer.index == 0
