import pytest
from unittest import mock


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
