import numpy as np
import pyfar as pf
from abc import abstractmethod
from threading import Event
from scipy import signal
import warnings


class _Buffer(object):
    """Abstract base class for audio buffers for block-wise iteration.

    The base class primarily implements buffer state related functionality.
    """

    def __init__(self, block_size) -> None:
        """Create a Buffer object with a given block size.

        Parameters
        ----------
        block_size : _type_
            _description_
        """
        self._check_block_size(block_size)
        self._block_size = block_size
        self._buffer = None
        self._is_active = Event()
        self._is_finished = Event()

    def _check_block_size(self, block_size):
        """Check if the block size is an integer."""
        if not isinstance(block_size, int):
            raise ValueError("The block size needs to be an integer")

    def _set_block_size(self, block_size):
        """Private block size setter implementing validity checks."""
        self.check_if_active()
        self._check_block_size(block_size)
        self._block_size = block_size

    @property
    def block_size(self):
        """Returns the block size of the buffer in samples"""
        return self._block_size

    @block_size.setter
    def block_size(self, block_size):
        """Set the block size in samples. Only integer values are supported"""
        self._set_block_size(block_size)

    @property
    @abstractmethod
    def sampling_rate(self):
        """Return sampling rate."""
        pass

    def __iter__(self):
        return self

    def __next__(self):
        """Next dunder method for iteration"""
        self._start()
        return self.next()

    @abstractmethod
    def next(self):
        """Next method which for sub-class specific handling of data."""
        raise NotImplementedError()

    @property
    def is_active(self):
        """Return the state of the buffer.
        `True` if the buffer is active, `False` if inactive."""
        return self._is_active.is_set()

    @property
    def is_finished(self):
        """Return if the buffer has finished iteration.
        `True` if the buffer is finished, `False` if not.
        """
        return self._is_finished.is_set()

    def check_if_active(self):
        """Check if the buffer is active and raise an exception if so.
        If the buffer is active a BufferError exception is raised. In case the
        buffer is currently inactive, the method simply passes without any
        return value. This method should always be called before attempting to
        modify properties of the buffer to prevent undefined behavior during
        iteration of the buffer.

        Raises
        ------
        BufferError
            Exception is raised if the buffer is currently active.
        """
        if self.is_active:
            raise BufferError(
                "The buffer needs to be inactive to be modified.")

    def _stop(self, msg="Buffer iteration stopped."):
        """Stop buffer iteration and set the state to inactive."""
        self._is_active.clear()
        self._is_finished.set()
        raise StopIteration(msg)

    def _start(self):
        """Set the state to active.
        Additional operations required before iterating the sub-class can be
        implemented in the respective sub-class."""
        self._is_active.set()
        self._is_finished.clear()

    def _reset(self):
        """Stop and reset the buffer.
        Resetting the buffer is implemented in the respective sub-class"""
        self._is_active.clear()
        self._is_finished.clear()
        raise StopIteration("Resetting the buffer.")


class SignalBuffer(_Buffer):
    """Buffer to block wise iterate a `pyfar.Signal`

    Examples
    --------

    >>> import pyfar as pf
    >>> from haiopy.buffers import SignalBuffer
    >>> block_size = 512
    >>> sine = pf.signals.sine(440, 4*block_size)
    >>> buffer = SignalBuffer(block_size, sine)
    >>> for block in buffer:
    >>>     print(block)


    """

    def __init__(self, block_size, signal) -> None:
        """Initialize a `SignalBuffer` with a given block size from a
        `pyfar.Signal`.
        If the number of audio samples is not an integer multiple of the
        block size, the last block will be filled with zeros.

        Parameters
        ----------
        block_size : int
            The block size in samples
        signal : pyfar.Signal
            The audio data to be block wise iterated.

        """
        super().__init__(block_size)
        if not isinstance(signal, pf.Signal):
            raise ValueError("signal must be a pyfar.Signal object.")
        if signal.time.ndim > 2:
            raise ValueError("Only one-dimensional arrays are allowed")
        self._n_samples = signal.n_samples
        self._data = self._pad_data(signal)
        self._update_data()
        self._index = 0

    def _pad_data(self, data):
        """Pad the signal with zeros to avoid partially filled blocks

        Parameters
        ----------
        data : pyfar.Signal
            The input audio signal.

        Returns
        -------
        pyfar.Signal
            Zero-padded signal.
        """
        if np.mod(self._n_samples, self._block_size) <= 0:
            return data
        pad_samples = self.block_size - np.mod(
            self._n_samples, self.block_size)
        return pf.dsp.pad_zeros(data, pad_samples, mode='after')

    @property
    def n_channels(self):
        """The number of audio channels as integer."""
        return self.data.cshape[0]

    @property
    def sampling_rate(self):
        """The sampling rate of the underlying data."""
        return self.data.sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        """Set new sampling_rate and resample the input Signal without
        the padded zeros."""
        signal = pf.Signal(data=self._data.time[..., :self._n_samples],
                           sampling_rate=self.sampling_rate,
                           n_samples=self._n_samples,
                           domain=self._data.domain,
                           fft_norm=self._data._fft_norm,
                           comment=self._data.comment)
        self.data = pf.dsp.resample(signal, sampling_rate)
        warnings.warn("Resampling the input Signal to sampling_rate="
                      f"{sampling_rate} might generate artifacts.")

    @property
    def n_blocks(self):
        """The number of blocks contained in the buffer."""
        return self._n_blocks

    @property
    def index(self):
        """The current block index as integer."""
        return self._index

    @property
    def data(self):
        """Return the underlying signal if the buffer is not active."""
        self.check_if_active()
        return self._data

    @data.setter
    def data(self, data):
        """Set the underlying signal if the buffer is not active."""
        self.check_if_active()
        self._n_samples = data.n_samples
        self._data = self._pad_data(data)
        self._update_data()

    def _set_block_size(self, block_size):
        """Set block_size and data without the padded zeros,
        data setter will pad the data for the new blocksize."""
        super()._set_block_size(block_size)
        self._data.time = self._data.time[..., :self._n_samples]
        # Use data setter to pad the data and update the strides
        self.data = self._data

    def _update_data(self):
        """Update the data block strided of the underlying data.
        The function creates a block-wise view of the numpy data array storing
        the time domain data.
        """
        self.check_if_active()
        self._n_blocks = int(np.ceil(self.data.n_samples / self.block_size))
        self._strided_data = np.lib.stride_tricks.as_strided(
            self.data.time,
            (*self.data.cshape, self.n_blocks, self.block_size))
        self._index = 0

    def next(self):
        """Return the next audio block as numpy array and increment the block
        index.
        """
        if self._index < self._n_blocks:
            current = self._index
            self._index += 1
            return self._strided_data[..., current, :]
        self._stop("The buffer is empty.")

    def reset_index(self):
        """Resets the index and the block-wise view of the underlying data"""
        self._is_active.clear()
        self._is_finished.set()
        self._index = 0

    def _reset(self):
        self._index = 0
        super()._reset()


class SineGenerator(_Buffer):
    """Generator to block wise calculate a sinewave`

    Examples
    --------

    >>> from haiopy.buffers import SineGenerator
    >>> import matplotlib.pyplot as plt
    >>> sine = SineGenerator(440, 128)
    >>> blocks = [next(sine), next(sine), next(sine)]
    >>> for block in blocks:
    >>>     plt.plot((block))
    >>> plt.show()


    """

    def __init__(self,
                 frequency,
                 block_size,
                 amplitude=1,
                 sampling_rate=44100) -> None:
        """Initialize a `SineGenerator`with a given frequency, block_size,
        amplitude and samplingrate.

        Parameters
        ----------
        frequency : double
            Frequency of the sine in Hz(0 <= `frequency` <= `sampling_rate`/2).
        block_size : int
            The block size in samples.
        amplitude: double, optional
            The amplitude of the sine. The default is ``1``.
        sampling_rate : int, optional
            The sampling rate in Hz. The default is ``44100``.

        """
        super().__init__(block_size)
        self._frequency = frequency
        self._amplitude = amplitude
        self._sampling_rate = sampling_rate
        self._phase = 0

    @property
    def frequency(self):
        """Return the frequency of the sinewave"""
        return self._frequency

    @frequency.setter
    def frequency(self, frequency):
        self.check_if_active()
        self._frequency = frequency
        self._phase = 0

    @property
    def amplitude(self):
        """Return the amplitude of the sinewave"""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, amplitude):
        self.check_if_active()
        self._amplitude = amplitude
        self._phase = 0

    @property
    def sampling_rate(self):
        """Return the sampling rate of the generated sinewave."""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self.check_if_active()
        self._sampling_rate = sampling_rate
        self._phase = 0

    @property
    def phase(self):
        """Return the current phase of the sinewave"""
        return self._phase

    def _set_block_size(self, block_size):
        self.check_if_active()
        super()._set_block_size(block_size)
        self._phase = 0

    def next(self):
        """Return the next audio block as numpy array and increases the phase.
        """
        omega = 2 * np.pi * self._frequency
        data = self._amplitude * np.sin(
            omega*np.arange(self._block_size)/self._sampling_rate+self._phase)
        self._phase += omega*(self._block_size / self._sampling_rate)
        return data

    def _reset(self):
        self._phase = 0


class NoiseGenerator(_Buffer):
    """Generator for block wise noise generation.
    Supports white and pink noise spectra.
    >>> import pyfar as pf
    >>> from haiopy.buffers import NoiseGenerator
    >>> block_size = 22050
    >>> noise = NoiseGenerator(block_size, seed=10)
    >>> data = next(noise)
    >>> pf.plot.time_freq(pf.Signal(data, 44100))


    """
    def __init__(self,
                 block_size,
                 spectrum="white",
                 rms=1,
                 sampling_rate=44100,
                 seed=None) -> None:
        """Initialize a `NoiseGenerator` with given block size, spectrum,
        rms, sampling_rate and seed.

        The filter coefficients for generating the pink noise are presented in
        <https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html>.

        Parameters
        ----------
        block_size : int
            The block size in samples.
        spectrum: str, optional
            ``white`` to generate noise with constant energy across frequency.
            ``pink`` to generate noise with constant energy across filters with
            constant relative bandwidth. The default is ``"white"``.
        rms : double, optional
            The route mean square (RMS) value of the noise signal.
            The default is ``1``.
        sampling_rate :
            The sampling rate in Hz. The default is ``44100``.
        seed : int, None, optional
            The seed for the random generator. Pass a seed to obtain identical
            results for multiple calls. Consecutive blocks will still be
            different from each other. The default is ``None``, which will
            yield to radom results for every block.

        """
        super().__init__(block_size)
        self._rms = rms
        self._sampling_rate = sampling_rate
        # Generate a Random Generator Object and set self._seed = seed
        self._generate_rng(seed)
        self._spectrum = self._validate_spectrum(spectrum)
        # filter coefficients for pink noise
        self._pink_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        self._pink_A = [1, -2.494956002, 2.017265875, -0.522189400]
        self._zi = signal.lfilter_zi(self._pink_B, self._pink_A)

    @property
    def sampling_rate(self):
        """Return the sampling rate of the generated Noise."""
        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, sampling_rate):
        self.check_if_active()
        self._sampling_rate = sampling_rate

    @property
    def spectrum(self):
        """Return if NoiseGenerator generates white or pink noise."""
        return self._spectrum

    @spectrum.setter
    def spectrum(self, spectrum):
        self.check_if_active()
        self._spectrum = self._validate_spectrum(spectrum)

    def _validate_spectrum(self, spectrum):
        if spectrum not in ("white", "pink"):
            raise ValueError(
                f"spectrum is '{spectrum}' but must be 'white' or 'pink'")
        return spectrum

    @property
    def rms(self):
        """Return the route mean square (RMS) value of the noise."""
        return self._rms

    @rms.setter
    def rms(self, rms):
        self.check_if_active()
        self._rms = rms

    @property
    def seed(self):
        """Return the seed of the random generator."""
        return self._seed

    @seed.setter
    def seed(self, seed):
        self.check_if_active()
        """Seed will be set in _generate_rng, also a new Random Generator
        Object is created"""
        self._generate_rng(seed)

    def _generate_rng(self, seed):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def next(self):
        """Return the next audio block as numpy array, and calculates data
        with seed_idx if seed is True.
        """
        data = self._rng.standard_normal(self._block_size)
        if self._spectrum == "pink":
            # Apply Pink Noise Filter
            data, self._zi = signal.lfilter(self._pink_B, self._pink_A, data,
                                            zi=self._zi)
        # level the noise
        rms_current = np.sqrt(np.mean(data**2))
        data = data / rms_current * self._rms
        return data

    def _reset(self):
        super()._reset()
