import pytest
from unittest import mock
import numpy as np
from haiopy.buffers import SignalBuffer


def default_devices():
    return [0, 0]


def query_devices(id=None, kind=None):
    if kind == 'input':
        return {
            'name': "MockDevice",
            'index': 0,
            'hostapi': 'CoreAudio',
            'max_input_channels': 8,
            'default_low_input_latency': 0.1,
            'default_high_input_latency': 0.15,
            'default_samplerate': 44100
        }
    elif kind == 'output':
        return {
            'name': "MockInputDevice",
            'index': 0,
            'hostapi': 'CoreAudio',
            'max_output_channels': 8,
            'default_low_output_latency': 0.1,
            'default_high_output_latency': 0.15,
            'default_samplerate': 44100
        }
    else:
        return {
            'name': "MockOutput",
            'index': 0,
            'hostapi': 'CoreAudio',
            'max_input_channels': 8,
            'max_output_channels': 8,
            'default_low_input_latency': 0.1,
            'default_low_output_latency': 0.1,
            'default_high_input_latency': 0.15,
            'default_high_output_latency': 0.15,
            'default_samplerate': 44100
        }


def supported_mock_device_parameters():
    return {
        'samplerate': [44.1e3, 48e3, 2*44.1e3, 96e3, 192e3],
        'dtype': ['float32'],
        'channels': [8]}


def check_output_settings(
        device=None,
        channels=None,
        dtype=None,
        extra_settings=None,
        samplerate=None):
    """So far this only passes for all settings"""
    pass


def check_input_settings(
        device=None,
        channels=None,
        dtype=None,
        extra_settings=None,
        samplerate=None):
    """So far this only passes for all settings"""
    pass


def array_buffer_stub(block_size=512, data=np.zeros((1, 512))):
    """Generate a ArrayBuffer Stub with given block size and data

    Parameters
    ----------
    block_size : int
        Block size for the sound card callback
    data : array_like, float32, int24, int16, int8
        The data of the buffer
    """
    if np.mod(data.shape[-1], block_size) != 0:
        raise ValueError(
            'The data needs to be an integer multiple of the block size')

    n_blocks = data.shape[-1] // block_size

    def next_block():
        strided = np.lib.stride_tricks.as_strided(
            data, (*data.shape[:-1], n_blocks, block_size))

        for idx in range(n_blocks):
            yield strided[..., idx, :]

    # buffer = mock.MagicMock(spec_set=ArrayBuffer(block_size, data))
    buffer = SignalBuffer(block_size, data)

    # buffer.data = data
    # buffer._strided_data = np.atleast_3d(data)
    # buffer.next = np.atleast_3d(next_block)
    # buffer.n_blocks = n_blocks
    # buffer.block_size = block_size

    return buffer
