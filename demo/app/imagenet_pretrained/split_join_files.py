#!/usr/bin/env python
"""
Split the imagenet model file.

Script based on: https://gist.github.com/mattiasostmar/7883550
"""
# Author: Mainak Jas <mainak.jas@telecom-paristech.fr>
#
# License: BSD (3-clause)

import os


def split_file(input_fname, size):
    """Split the file.

    Parameters
    ----------
    input_fname : str
        The name of file to be split.
    size : int
        The size of each file chunk.

    Returns
    -------
    n_chunks : int
        Number of chunks the is split data into.
    """
    # read the contents of the file
    with open(input_fname, 'rb') as f:
        data = f.read()
    # get the length of data, i.e. size of the input file in bytes
    bytes = len(data)

    # calculate the number of chunks to be created
    n_chunks = bytes // size
    if bytes % size:
        n_chunks += 1

    for idx, pos in enumerate(range(0, bytes + 1, size)):
        print('Writing chunk %d' % (idx + 1))
        chunk_fname = "%s_%d" % (input_fname, pos)
        with open(chunk_fname, 'wb') as f:
            f.write(data[pos:pos + size])
    return n_chunks


def join_files(fname, n_chunks, size, cleanup=True):
    """Join the files.

    Parameters
    ----------
    fname : str
        The prefix of the split files.
    n_chunks : int
        The number of chunks.
    size : int
        The size of each chunk.
    cleanup : bool
        Remove the small files if True.
    """
    with open(fname, 'wb') as f2:
        for idx in range(0, n_chunks):
            print('Reading chunk %d' % (idx + 1))
            fname_chunk = '%s_%d' % (fname, idx * size)
            with open(fname_chunk, 'rb') as f:
                data = f.read()
                f2.write(data)
            os.remove(fname_chunk)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("-j", "--join",
                      action="store_false", dest="join", default=True,
                      help="join the files")
    parser.add_option("-s", "--split",
                      action="store_false", dest="split", default=True,
                      help="split the files")
    (options, args) = parser.parse_args()
    join, split = options.join, options.split

    size = 90 * (1024 ** 2)  # ~90 MB
    n_chunks = 3
    if join:
        n_chunks = split_file('imagenet.decafnet.epoch90', size)
    if split:
        join_files('imagenet.decafnet.epoch90', n_chunks=n_chunks, size=size)
