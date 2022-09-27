import numpy as np
from abc import abstractproperty, abstractmethod


class Buffer(object):

    def __init__(self, block_size) -> None:
        self._check_block_size(block_size)
        self._block_size = block_size
        self._buffer = None

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
        return self.next()

    @abstractmethod
    def next(self):
        raise NotImplementedError()


class ArrayBuffer(Buffer):

    def __init__(self, block_size, data, sampling_rate) -> None:
        super().__init__(block_size)
        if data.ndim > 2:
            raise ValueError("Only two-dimensional arrays are allowed")
        self.data = np.atleast_2d(data)
        self._index = 0
        self._sampling_rate = sampling_rate

    def _pad_data(self, data):
        n_samples = data.shape[-1]
        if np.mod(n_samples, self._block_size) > 0:
            pad_samples = self.block_size - np.mod(n_samples, self.block_size)
            pad_array = np.zeros((data.shape[0], 2), dtype=int)
            pad_array[-1][-1] = pad_samples
            padded = np.pad(data, pad_array)
        else:
            padded = data

        return padded

    @property
    def n_channels(self):
        return self.data.shape[0]

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def n_blocks(self):
        return self._n_blocks

    @property
    def index(self):
        return self._index

    @property
    def data(self):
        return self._data

    def _set_block_size(self, block_size):
        super()._set_block_size(block_size)
        self._update_data()

    def _update_data(self):
        self._n_blocks = int(np.ceil(self.data.shape[-1] / self.block_size))
        self._strided_data = np.lib.stride_tricks.as_strided(
            self.data,
            (*self.data.shape[:-1], self.n_blocks, self.block_size))

    @data.setter
    def data(self, data):
        self._data = self._pad_data(np.atleast_2d(data))
        self._update_data()

    def next(self):
        if self._index < self._n_blocks:
            current = self._index
            self._index += 1
            return self._strided_data[..., current, :]
        raise StopIteration("The buffer is empty.")


class OutputArrayBuffer(ArrayBuffer):

    def __init__(self, block_size, data, sampling_rate) -> None:
        super().__init__(block_size, data, sampling_rate)

    def _update_data(self):
        self._n_blocks = int(np.ceil(self.data.shape[-1] / self.block_size))
        self._strided_data = np.lib.stride_tricks.as_strided(
            self.data,
            (*self.data.shape[:-1], self.n_blocks, self.block_size),
            writeable=False)


class InputArrayBuffer(ArrayBuffer):

    def __init__(self, block_size, data, sampling_rate) -> None:
        super().__init__(block_size, data, sampling_rate)
