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
