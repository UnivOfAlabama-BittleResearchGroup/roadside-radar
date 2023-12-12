import time
import polars as pl


def timeit(func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(
            "function: {} took: {} seconds".format(func.__name__, end_time - start_time)
        )

        return result

    return timed


def lazify(func):
    # if the input is a lazy frame, then return a lazy frame
    # otherwise return a regular frame
    def lazy_func(*args, **kwargs):
        if isinstance(args[0], pl.LazyFrame):
            res = func(*args, **kwargs)
            if isinstance(res, pl.DataFrame):
                return res.lazy()
            return res
        else:
            return func(*args, **kwargs)

    return lazy_func
