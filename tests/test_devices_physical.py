from haiopy import devices
import sounddevice as sd
from . import utils
from unittest.mock import patch, MagicMock
import time


def default_device_multiface_fireface():
    device_list = sd.query_devices()
    found = False
    for idx, dev in enumerate(device_list):
        if 'Fireface' in dev['name'] or 'Multiface' in dev['name']:
            found = True
            break
    if not found:
        raise ValueError(
            "Please connect Fireface or Multiface, or specify test device.")

    return idx
    # default = MagicMock(spec_sec=sd.default)
    # default.device = [idx, idx]
    # default._default_device = (idx, idx)

    # return default


def test_default_device_helper():
    identifier = default_device_multiface_fireface()
    fireface = 'Fireface' in sd.query_devices(identifier)['name']
    multiface = 'Multiface' in sd.query_devices(identifier)['name']
    assert fireface or multiface


def test_check_input_settings():
    identifier = default_device_multiface_fireface()

    in_device = devices.InputAudioDevice(identifier)
    in_device.check_settings()


def test_check_output_settings(empty_buffer_stub):
    identifier = default_device_multiface_fireface()

    channels = [1]
    block_size = 512

    out_device = devices.OutputAudioDevice(
        identifier, 44100, block_size, channels=channels, dtype='float32',
        output_buffer=empty_buffer_stub)

    out_device.check_settings(sampling_rate=23e3)

    sd.check_output_settings(identifier, samplerate=23e3)


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
