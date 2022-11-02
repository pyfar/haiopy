import numpy as np
import pyfar as pf
from abc import abstractproperty, abstractmethod


class Buffer(object):

    def __init__(self, block_size) -> None:
        self._check_block_size(block_size)
        self._block_size = block_size
        self._buffer = None
        self._is_active = False

    def _check_block_size(self, block_size):
        if type(block_size) != int:
            raise ValueError("The block size needs to be an integer")

    def _set_block_size(self, block_size):
        self._check_block_size(block_size)
        self._block_size = block_size

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, block_size):
        self._set_block_size(block_size)

    @abstractproperty
    def sampling_rate(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def __next__(self):
        self._start()
        return self.next()

    @abstractmethod
    def next(self):
        raise NotImplementedError()

    @property
    def is_active(self):
        return self._is_active

    def check_if_active(self):
        """Check if the buffer is active.
        If the buffer is active a BufferError exception is raised. In case the
        buffer is currently inactive, the method simply passes without any
        return value.

        Raises
        ------
        BufferError
            Exception is raised if the buffer is currently active.
        """
        if self.is_active:
            raise BufferError(
                "The buffer needs to be inactive to be modified.")

    def _stop(self, msg="Buffer iteration stopped."):
        self._is_active = False
        raise StopIteration(msg)

    def _start(self):
        self._is_active = True

    def _reset(self):
        self._stop()


class SignalBuffer(Buffer):

    def __init__(self, block_size, signal) -> None:
        super().__init__(block_size)
        if not isinstance(signal, pf.Signal):
            raise ValueError("signal must be a pyfar.Signal object.")
        if signal.time.ndim > 2:
            raise ValueError("Only two-dimensional arrays are allowed")
        self._data = self._pad_data(signal)
        self._update_data()
        self._index = 0

    def _pad_data(self, data):
        n_samples = data.n_samples
        if np.mod(n_samples, self._block_size) > 0:
            pad_samples = self.block_size - np.mod(n_samples, self.block_size)
            return pf.dsp.pad_zeros(data, pad_samples, mode='after')
        else:
            return data

    @property
    def n_channels(self):
        return self.data.cshape[0]

    @property
    def sampling_rate(self):
        return self.data.sampling_rate

    @property
    def n_blocks(self):
        return self._n_blocks

    @property
    def index(self):
        return self._index

    @property
    def data(self):
        self.check_if_active()
        return self._data

    @data.setter
    def data(self, data):
        self.check_if_active()
        self._data = self._pad_data(data)
        self._update_data()

    def _set_block_size(self, block_size):
        self.check_if_active()
        super()._set_block_size(block_size)
        self._update_data()

    def _update_data(self):
        self.check_if_active()
        self._n_blocks = int(np.ceil(self.data.n_samples / self.block_size))
        self._strided_data = np.lib.stride_tricks.as_strided(
            self.data.time,
            (*self.data.cshape, self.n_blocks, self.block_size))

    def next(self):
        if self._index < self._n_blocks:
            current = self._index
            self._index += 1
            return self._strided_data[..., current, :]
        self._stop("The buffer is empty.")
