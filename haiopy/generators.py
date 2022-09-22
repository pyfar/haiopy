import numpy as np


class Buffer(object):

    def __init__(self, block_size) -> None:
        if type(block_size) != int:
            raise ValueError("The block size needs to be an integer")
        self._buffer = None
        self._block_size = block_size

    def __iter__(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()


class ArrayBuffer(Buffer):

    def __init__(self, data, block_size) -> None:
        super().__init__(block_size)

        n_samples = data.shape[-1]
        if np.mod(n_samples, block_size) > 0:
            pad_samples = block_size - np.mod(n_samples, block_size)
            pad_array = np.zeros((data.shape[0], 2), dtype=int)
            pad_array[-1][-1] = pad_samples
            data = np.pad(data, pad_array)

        self._block_size = block_size
        self._n_blocks = int(np.ceil(data.shape[-1] / block_size))

        self._buffer = np.lib.stride_tricks.as_strided(
            data.copy(),
            (*data.shape[:-1], self._n_blocks, block_size))

        self._index = 0


class OutputArrayBuffer(ArrayBuffer):

    def __init__(self, data, block_size) -> None:
        super().__init__(self, data, block_size)

    def __next__(self):
        return self.next()

    def next(self):
        if self._index < self._n_blocks:
            current = self._index
            self._index += 1
            return self._buffer[..., current, :]
        raise StopIteration("Buffer is empty")
