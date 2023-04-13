import numpy as np
import numpy.testing as npt
from haiopy.buffers import _Buffer, SignalBuffer, NoiseGenerator
import pytest
import pyfar as pf


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


def test_NoiseGenerator():
    block_size = 512

    noise = NoiseGenerator(block_size, seed=True)

    # test getters with default
    assert noise.block_size == block_size
    assert noise.spectrum == "white"
    assert noise.rms == 1
    assert noise.sampling_rate == 44100
    assert noise.seed is True

    # check if noise generator is not active yet
    assert noise.is_active is False

    # check first block
    rng = np.random.default_rng(0)
    noise_data = rng.standard_normal(block_size)
    # level the noise
    noise_data /= np.sqrt(np.mean(noise_data**2))
    block_data = next(noise)
    npt.assert_array_equal(block_data, noise_data)

    # check if noise generator is active now
    assert noise.is_active is True

    # check second block
    rng = np.random.default_rng(1)
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

    new_spectrum = "pink"
    noise.spectrum = new_spectrum
    assert noise.spectrum == new_spectrum

    new_rms = 0.5
    noise.rms = new_rms
    assert noise.rms == new_rms

    new_sampling_rate = 48000
    noise.sampling_rate = new_sampling_rate
    assert noise.sampling_rate == new_sampling_rate

    new_seed = True
    noise.seed = new_seed
    assert noise.seed == new_seed

    # Check updated data and pink spectrum
    rng = np.random.default_rng(0)
    noise_data = rng.standard_normal(new_block_size)
    # Apply 1/f filter for pink noise spectrum
    noise_data = pf.dsp.fft.rfft(noise_data, new_block_size,
                                 new_sampling_rate, 'none')
    noise_data /= np.sqrt(np.arange(1, noise_data.shape[-1]+1))
    noise_data = pf.dsp.fft.irfft(noise_data, new_block_size,
                                  new_sampling_rate, 'none')
    # level the noise
    rms_current = np.sqrt(np.mean(noise_data**2))
    noise_data = noise_data / rms_current * new_rms

    block_data = next(noise)
    npt.assert_array_equal(block_data, noise_data)

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
        noise.seed = False
