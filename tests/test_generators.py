import numpy as np
import numpy.testing as npt
from haiopy.generators import Buffer, ArrayBuffer, SignalBuffer
import pytest
import pyfar as pf


def test_buffer():

    block_size = 512
    buffer = Buffer(block_size)

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
        Buffer(float(10))


def test_buffer_state():
    block_size = 512

    # create new buffer
    buffer = Buffer(block_size)
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


def test_array_buffer():

    block_size = 512
    n_blocks = 10
    n_samples = block_size*n_blocks
    sampling_rate = 44100

    freq = 440

    data_pf = pf.signals.sine(
        freq, n_samples, sampling_rate=sampling_rate)
    data = data_pf.time

    with pytest.raises(ValueError, match='Only two-dimensional'):
        ArrayBuffer(block_size, np.zeros((1, 1, block_size)), sampling_rate)

    buffer = ArrayBuffer(block_size, data, sampling_rate)

    assert buffer._n_blocks == n_blocks
    assert buffer.n_blocks == n_blocks

    # check if the initial index s correct
    assert buffer._index == 0
    assert buffer.index == 0

    # check if the data arrays are correct
    npt.assert_array_equal(buffer._data, data)
    npt.assert_array_equal(buffer.data, data)

    # check if the data strides are correct
    strided_buffer_data = np.lib.stride_tricks.as_strided(
        data, (*data.shape[:-1], n_blocks, block_size))
    npt.assert_array_equal(
        buffer._strided_data, strided_buffer_data)

    # check first step
    block_data = buffer.__next__()
    npt.assert_array_equal(block_data, strided_buffer_data[..., 0, :])

    # check second step
    block_data = buffer.__next__()
    npt.assert_array_equal(block_data, strided_buffer_data[..., 1, :])

    # check if a error is raised if the end of the buffer is reached
    with pytest.raises(StopIteration, match="buffer is empty"):
        while True:
            buffer.__next__()


def test_signal_buffer():
    sampling_rate = 44100
    n_blocks = 10
    block_size = 512
    n_samples = block_size*n_blocks
    sine = pf.signals.sine(
        440, n_samples, amplitude=[1, 1], sampling_rate=sampling_rate)

    with pytest.raises(ValueError, match='two-dimensional'):
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
    block_data = buffer.__next__()
    npt.assert_array_equal(block_data, strided_buffer_data[..., 0, :])

    # check second step
    block_data = buffer.__next__()
    npt.assert_array_equal(block_data, strided_buffer_data[..., 1, :])

    # check if a error is raised if the end of the buffer is reached
    with pytest.raises(StopIteration, match="buffer is empty"):
        while True:
            buffer.__next__()


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
    assert buffer._is_active is True

    # Setting the block size is not allowed if the buffer is active
    with pytest.raises(BufferError, match="needs to be inactive"):
        buffer.block_size = 512

    # Setting and getting the data is not allowed if the buffer is active
    with pytest.raises(BufferError, match="needs to be inactive"):
        buffer.data = sine

    with pytest.raises(BufferError, match="needs to be inactive"):
        buffer.data
