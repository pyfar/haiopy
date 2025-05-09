from unittest.mock import patch
import pytest

from haiopy.devices import (
    OutputChannelMapping, InputChannelMapping)


@pytest.mark.parametrize(
    'valid_apis', [
        'windows wasapi',
        'windows directsound',
        'asio',
        'windows wdm-ks'])
@patch('platform.system', new=lambda: 'Windows')
@patch('sounddevice.AsioSettings', new=lambda channel_selectors: None)
def test_init_checks_windows(valid_apis):
    OutputChannelMapping([0], 1, valid_apis)
    InputChannelMapping([0], 1, valid_apis)


@pytest.mark.parametrize(
    'valid_apis', [
        'alsa',
        'oss',
        'pulse',
        'jack'])
@patch('platform.system', new=lambda: 'Linux')
def test_init_checks_linux(valid_apis):
    OutputChannelMapping([0], 1, valid_apis)
    InputChannelMapping([0], 1, valid_apis)


@pytest.mark.parametrize(
    'valid_apis', ['coreaudio'])
@patch('platform.system', new=lambda: 'Darwin')
@patch('sounddevice.CoreAudioSettings', new=lambda channel_map: None)
def test_init_checks_macos(valid_apis):
    OutputChannelMapping([0], 1, valid_apis)
    InputChannelMapping([0], 1, valid_apis)


@patch('platform.system', new=lambda: 'Linux')
def test_manual_mapping():
    output_mapping = OutputChannelMapping([2, 3], 4, 'alsa')

    data_block = np.ones((2, 512), dtype=float)
    truth = np.vstack((
        np.zeros_like(data_block),
        data_block,
    ))
    np.testing.assert_array_equal(
        output_mapping(data_block), truth)
