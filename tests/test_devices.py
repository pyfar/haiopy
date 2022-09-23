from haiopy import devices
from . import utils
from unittest.mock import patch


@patch('sounddevice.query_devices', new=utils.query_devices)
def test_audio_device():
    devices.AudioDevice(0)
