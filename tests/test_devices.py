from haiopy import devices
from . import utils
from . import sounddevice_mocks as sdm
from unittest.mock import patch


@patch('sounddevice.query_devices', new=utils.query_devices)
def test_audio_device():
    devices.AudioDevice(0)


@patch('sounddevice.query_devices', new=utils.query_devices)
@patch('sounddevice.check_output_settings', new=utils.check_output_settings)
@patch('sounddevice.OutputStream', new=sdm.output_stream_mock())
def test_check_output_settings(empty_buffer_stub):
    out_device = devices.OutputAudioDevice(
        output_buffer=empty_buffer_stub[0])
    out_device.check_settings()
