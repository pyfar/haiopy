from haiopy import devices
import sounddevice as sd
from . import utils
from unittest.mock import patch, MagicMock
import time
import pytest


def default_device_multiface_fireface():
    device_list = sd.query_devices()
    found = False
    for identifier, device in enumerate(device_list):
        if 'Fireface' in device['name'] or 'Multiface' in device['name']:
            found = True
            break
    if not found:
        raise ValueError(
            "Please connect Fireface or Multiface, or specify test device.")

    return identifier, device
    # default = MagicMock(spec_sec=sd.default)
    # default.device = [idx, idx]
    # default._default_device = (idx, idx)

    # return default


def test_default_device_helper():
    identifier, device = default_device_multiface_fireface()
    fireface = 'Fireface' in sd.query_devices(identifier)['name']
    multiface = 'Multiface' in sd.query_devices(identifier)['name']
    assert fireface or multiface

    if fireface:
        assert device['max_input_channels'] == 18
        assert device['max_output_channels'] == 18


def test_check_input_settings():
    identifier, config = default_device_multiface_fireface()

    default_sampling_rate = config['default_samplerate']

    # Create device
    in_device = devices.InputAudioDevice(identifier)

    # Check sampling rate
    in_device.check_settings(sampling_rate=default_sampling_rate)
    with pytest.raises(sd.PortAudioError, match="Invalid"):
        in_device.check_settings(sampling_rate=10)

    # Check the dtype, apparently this raises a ValueError if invalid
    in_device.check_settings(dtype='float32')
    with pytest.raises(ValueError, match="Invalid"):
        in_device.check_settings(dtype=float)

    # Check number of channels
    in_device.check_settings(n_channels=config['max_input_channels'])
    with pytest.raises(sd.PortAudioError, match="Invalid"):
        in_device.check_settings(config['max_input_channels']+10)


def test_check_output_settings(empty_buffer_stub):
    identifier, config = default_device_multiface_fireface()

    channels = [3]
    block_size = 512

    out_device = devices.OutputAudioDevice(
        identifier, 44100, block_size, channels=channels, dtype='float32',
        output_buffer=empty_buffer_stub)

    # Check sampling rate
    out_device.check_settings(sampling_rate=config['default_samplerate'])
    with pytest.raises(sd.PortAudioError, match="Invalid"):
        out_device.check_settings(sampling_rate=10)

    # Check the dtype, apparently this raises a ValueError if invalid
    out_device.check_settings(dtype='float32')
    with pytest.raises(ValueError, match="Invalid"):
        out_device.check_settings(dtype=float)

    # Check number of channels
    out_device.check_settings(n_channels=config['max_output_channels'])
    with pytest.raises(sd.PortAudioError, match="Invalid"):
        out_device.check_settings(config['max_output_channels']+10)


def test_sine_playback(sine_buffer_stub):

    buffer = sine_buffer_stub[0]
    duration = sine_buffer_stub[1]

    out_device = devices.OutputAudioDevice(
        identifier=default_device_multiface_fireface(),
        output_buffer=buffer,
        channels=[3])
    out_device.check_settings()

    out_device.start()
    time.sleep(duration)
