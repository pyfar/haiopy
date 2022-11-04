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

    buffer = signal_buffer_stub(block_size, data)
    duration = block_size*n_blocks/buffer.sampling_rate

    return buffer, duration


@pytest.fixture
def sine_buffer_stub():
    """Create a stub representing an empty ArrayBuffer.

    Returns
    -------
    buffer: SignalBuffer
        Stub of SignalBuffer
    duration: float
        Duration of the buffer in seconds. Required if waiting for the buffer
        to finish is required.

    """
    sampling_rate = 44100
    block_size = 512
    n_blocks = 86
    data = np.zeros((1, n_blocks*block_size), dtype='float32')
    t = np.arange(0, block_size*n_blocks)
    data = np.sin(2*np.pi*t*(block_size + 1)/sampling_rate)*10**(-6/20)

    data = np.atleast_2d(data).astype('float32')
    buffer = SignalBuffer(block_size, pf.Signal(data, sampling_rate))
    duration = block_size*n_blocks/sampling_rate

    return buffer, duration
