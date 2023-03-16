import numpy as np

class Gen:
    def __init__(self, n):
        self._n = 0
        self._cursor = 0
    def __next__(self):
        pass
    def __iter__(self):
        return self
    def reset(self):
        self._cursor = 0
    def __len__(self):
        return self._n

class GenUsingFile(Gen):
    def __init__(self, filelist, n):
        self._filelist = filelist
        self._n = len(filelist) if n <= 0 or n > len(filelist) else n
        self.reset()
    def __next__(self):
        if self._cursor == self._n:
            raise StopIteration
        self._cursor += 1
        return np.load(self._filelist[self._cursor - 1])

class GenUsingFunc (Gen):
    def __init__(self, func, n):
        self._func = func
        self._n = n
        self.reset()
    def __next__(self):
        if self._cursor == self._n:
            raise StopIteration
        self._cursor += 1
        return self._func()

class GenUsingList (Gen):
    def __init__(self, data_list, n):
        self._data_list = data_list
        self._n = len(data_list) if n <= 0 or n > len(data_list) else n
        self.reset()
    def __next__(self):
        if self._cursor == self._n:
            raise StopIteration
        self._cursor += 1
        return self._data_list[self._cursor - 1]

if __name__ == '__main__':
    L, p = 6, 0.6
    def perc_func(L, p):
        def F():
            return (np.random.random(size=(L,L)) < p).astype(int)
        return F
    f = perc_func(L, p)


