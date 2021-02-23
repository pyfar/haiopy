"""Main module."""

import warnings
import numpy as np

import scipy.io.wavfile as wavfile

import pyfar
from pyfar import Signal                     # managing audio signals
from pyfar.coordinates import Coordinates    # managing satial sampling points
from pyfar.orientations import Orientations  # managing orientation vectors
from pyfar.spatial import samplings          # creating predefined spatial sampling grids
import pyfar.plot as plot                    # beatiful plots
import pyfar.dsp.filter as filt

import sounddevice





