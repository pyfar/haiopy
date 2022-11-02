from haiopy import devices
import sounddevice as sd
from . import utils
from unittest.mock import patch, MagicMock
import asyncio


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
    id = default_device_multiface_fireface()
    fireface = 'Fireface' in sd.query_devices(id)['name']
    multiface = 'Multiface' in sd.query_devices(id)['name']
    assert fireface or multiface


def test_check_input_settings():
    id = default_device_multiface_fireface()

    in_device = devices.InputAudioDevice(id)
    in_device.check_settings()


def test_check_output_settings(empty_buffer_stub):
    id = default_device_multiface_fireface()

    channels = [1]
    block_size = 512

    out_device = devices.OutputAudioDevice(
        id, 44100, block_size, channels=channels, dtype='float32',
        output_buffer=empty_buffer_stub)

    out_device.check_settings(sampling_rate=23e3)

    sd.check_output_settings(id, samplerate=23e3)

    pass


def test_sine_playback(sine_buffer_stub):

    sine_buffer_stub

    out_device = devices.OutputAudioDevice(
        identifier=default_device_multiface_fireface(),
        output_buffer=sine_buffer_stub,
        channels=[3])
    out_device.check_settings()

    out_device.start()

    asyncio.sleep(1)


    # out_device.close()
