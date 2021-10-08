import pytest
import numpy as np
from unittest import mock

import haiopy as hp


@pytest.fixture
@mock.patch('haiopy.AudioDevice', autospec=True)
def device_stub(AudioDevice):
    return hp.AudioDevice(0, 44100, 512, np.float)
