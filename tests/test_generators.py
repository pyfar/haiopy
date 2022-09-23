import numpy as np
import numpy.testing as npt
from haiopy.generators import Buffer, ArrayBuffer
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

    # with pytest.raises(NotImplementedError):
    #     buffer.data

    with pytest.raises(NotImplementedError):
        buffer.__next__()

    with pytest.raises(NotImplementedError):
        buffer.__iter__()


def test_array_buffer():

    block_size = 512
    n_blocks = 10
    n_samples = block_size*n_blocks
    sampling_rate = 44100

    freq = 440

    data_pf = pf.signals.sine(freq, n_samples, sampling_rate=sampling_rate)
    data = data_pf.time

    buffer = ArrayBuffer(block_size, data)

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


def test_buffer_updates():
    block_size = 512
    n_blocks = 10
    n_samples = block_size*n_blocks
    sampling_rate = 44100

    freq = 440

    data_pf = pf.signals.sine(freq, n_samples, sampling_rate=sampling_rate)
    data = data_pf.time

    data_empty = np.zeros_like(data)

    buffer = ArrayBuffer(block_size, data_empty)

    buffer.data = data
    npt.assert_array_equal(buffer._data, data)
    npt.assert_array_equal(buffer.data, data)

    # The new block size is 4 times smaller than the old one
    new_block_size = 128
    buffer.block_size = new_block_size

    # The data itself is not touched in this case
    npt.assert_array_equal(buffer._data, data)
    npt.assert_array_equal(buffer.data, data)

    # Stride the array with the new block size
    # The new number of blocks is an integer multiple of the old block size
    new_n_blocks = n_samples // new_block_size
    strided_buffer_data = np.lib.stride_tricks.as_strided(
        data, (*data.shape[:-1], new_n_blocks, new_block_size))
    npt.assert_array_equal(
        buffer._strided_data, strided_buffer_data)
