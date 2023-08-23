import itertools
import re 

def batch_iter(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def concatenate(lists):
    return list(itertools.chain.from_iterable(lists))

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)