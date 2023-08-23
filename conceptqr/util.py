import itertools

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def concatentate(lists):
    return list(itertools.chain.from_iterable(lists))