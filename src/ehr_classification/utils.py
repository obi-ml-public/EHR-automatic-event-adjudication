import lz4.frame
from pickle import dump, load, HIGHEST_PROTOCOL


def lz4_dump(data, filename, protocol=HIGHEST_PROTOCOL):
    with lz4.frame.open(filename, mode='wb') as f:
        dump(data, f, protocol=protocol)


def lz4_load(filename):
    with lz4.frame.open(filename, mode='rb') as f:
        return load(f)
