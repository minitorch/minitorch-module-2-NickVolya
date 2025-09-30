"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional
import typing as tp

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> float:
    return x < y


def eq(x: float, y: float) -> float:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    return max(0, x)


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def inv(x: float) -> float:
    return 1 / x


def log_back(x: float, y: float) -> float:
    return 1 / x * y


def inv_back(x: float, y: float) -> float:
    return -1 / x**2 * y


def relu_back(x: float, y: float) -> float:
    return y if x >= 0 else 0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[float], float], iterable: Iterable[float]) -> Iterable[float]:
    return (func(elem) for elem in iterable)


def zipWith(
    iterable_1: Iterable[float],
    iterable_2: Iterable[float],
    func: Callable[[float, float], float],
) -> Iterable[float]:
    return (func(elem_1, elem_2) for elem_1, elem_2 in zip(iterable_1, iterable_2))


def reduce(func: Callable[[float, float], float], iterable: Iterable[float]) -> float:
    result = None
    for elem in iterable:
        if result is None:
            result = elem
            continue
        result = func(result, elem)
    if result is None:
        return 0
    return result


def negList(lst: tp.List[float]) -> tp.List[float]:
    return list(map(neg, lst))


def addLists(lst1: tp.List[float], lst2: tp.List[float]) -> tp.List[float]:
    return list(zipWith(lst1, lst2, add))


def sum(lst: tp.List[float]) -> float:
    result = reduce(add, lst)
    return result if result is not None else 0


def prod(lst: tp.List[float]) -> float:
    return reduce(mul, lst)
