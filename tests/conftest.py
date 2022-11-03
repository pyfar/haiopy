import pytest
from .utils import signal_buffer_stub
from haiopy.buffers import SignalBuffer
import pyfar as pf

import numpy as np


@pytest.fixture
def empty_buffer_stub():
    """Create a stub representing an empty ArrayBuffer.

    Returns
    -------
    ArrayBuffer
        Stub of ArrayBuffer
    """

    block_size = 512
    n_blocks = 10
    data = np.zeros((1, n_blocks*block_size), dtype='float32')

    return signal_buffer_stub(block_size, data)


@pytest.fixture
def sine_buffer_stub():
    """Create a stub representing an empty ArrayBuffer.

    Returns
    -------
    ArrayBuffer
        Stub of ArrayBuffer
    """
    sampling_rate = 44100
    block_size = 512
    n_blocks = 10
    data = np.zeros((1, n_blocks*block_size), dtype='float32')
    t = np.arange(0, 512)
    data = np.sin(2*np.pi*t*(block_size + 1)/sampling_rate)

    data = np.tile(data, 100)

    data = np.atleast_2d(data).astype('float32')

    return SignalBuffer(block_size, pf.Signal(data, sampling_rate))
