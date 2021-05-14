import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time()
        result = func(*args, **kargs)
        end = time.time()
        print('run time for function {} = {} seconds'.format(func.__name__, end - start))
        return result
    return wrapper
