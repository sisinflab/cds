from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import BoxCellularDirectSearch, ObjectiveTracker

ArrayObjective = Callable[[np.ndarray], np.ndarray]

FEATURE_NAMES = (
    "differentiable",
    "separable",
    "scalable",
    "multi_modal",
    "non_convex",
    "non_plateau",
    "non_zero_solution",
    "symmetric",
)


@dataclass(frozen=True)
class DirectGOLibProblem:
    name: str
    family: str
    source_name: str
    dimension: int
    instance: int
    objective: ArrayObjective
    bounds: np.ndarray
    fmin: float
    xmin: np.ndarray
    shift: np.ndarray
    rotation: np.ndarray
    features: Tuple[int, ...]


@dataclass(frozen=True)
class DirectGOLibFunction:
    source_name: str
    family: str
    objective: ArrayObjective
    lower: float
    upper: float
    fmin: Callable[[int], float]
    xmin: Callable[[int], np.ndarray]
    min_dimension: int
    features: Tuple[int, ...]


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    runner: Callable[..., Tuple[np.ndarray, float]]
    params: Dict[str, float | int | str]


@dataclass(frozen=True)
class DirectGOLibBenchmarkSettings:
    budget_evaluations: int = 5000
    seeds: Tuple[int, ...] = tuple(range(20))
    dimensions: Tuple[int, ...] = (10,)
    instances: Tuple[int, ...] = (1, 2, 3, 4, 5)
    families: Tuple[str, ...] = ("ABS", "Layeb")
    cds_h_list: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    cds_n_list: Tuple[int, ...] = (8, 32)
    cma_sigma_list: Tuple[float, ...] = (0.1, 0.3, 0.5)
    pso_swarm_list: Tuple[int, ...] = (30, 50, 100)
    lshade_pop_factors: Tuple[int, ...] = (10, 20)
    include_cds: bool = True
    include_cmaes: bool = True
    include_pso: bool = True
    include_de: bool = True
    include_random: bool = True
    include_neldermead: bool = True
    include_pdfo: bool = True
    include_grid_search: bool = True
    include_bads: bool = True
    include_nomad: bool = True
    include_lshade: bool = True


def _as_batch(theta: np.ndarray) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.ndim == 1:
        return theta_array[np.newaxis, :]
    if theta_array.ndim != 2:
        raise ValueError(f"theta must be one- or two-dimensional, got shape {theta_array.shape}")
    return theta_array


def _sum(values: np.ndarray) -> np.ndarray:
    return np.sum(values, axis=1)


def _sin_rad(values: np.ndarray) -> np.ndarray:
    degrees = values / math.pi * 180.0
    result = np.sin(np.deg2rad(degrees))
    reduced = np.mod(degrees, 360.0)
    result = np.where(np.isclose(reduced, 0.0, atol=1e-10) | np.isclose(reduced, 180.0, atol=1e-10) | np.isclose(reduced, 360.0, atol=1e-10), 0.0, result)
    result = np.where(np.isclose(reduced, 90.0, atol=1e-10), 1.0, result)
    result = np.where(np.isclose(reduced, 270.0, atol=1e-10), -1.0, result)
    return result


def _cos_rad(values: np.ndarray) -> np.ndarray:
    degrees = values / math.pi * 180.0
    result = np.cos(np.deg2rad(degrees))
    reduced = np.mod(degrees, 360.0)
    result = np.where(np.isclose(reduced, 0.0, atol=1e-10) | np.isclose(reduced, 360.0, atol=1e-10), 1.0, result)
    result = np.where(np.isclose(reduced, 180.0, atol=1e-10), -1.0, result)
    result = np.where(np.isclose(reduced, 90.0, atol=1e-10) | np.isclose(reduced, 270.0, atol=1e-10), 0.0, result)
    return result


def _tan_rad(values: np.ndarray) -> np.ndarray:
    degrees = values / math.pi * 180.0
    result = _sin_rad(values) / _cos_rad(values)
    reduced = np.mod(degrees, 180.0)
    result = np.where(np.isclose(reduced, 45.0, atol=1e-10), 1.0, result)
    result = np.where(np.isclose(reduced, 135.0, atol=1e-10), -1.0, result)
    return result


def _zigzag(values: np.ndarray, k: float, m: float, lmbd: float) -> np.ndarray:
    reduced = np.abs(values)
    reduced = reduced / k - np.floor(reduced / k)
    low_branch = reduced <= lmbd
    return (
        1.0
        - m
        + low_branch * m * (reduced / lmbd)
        + (~low_branch) * m * (1.0 - (reduced - lmbd) / (1.0 - lmbd))
    )


def _abs_poly(values: np.ndarray, k: float, m: float, lmbd: float) -> np.ndarray:
    return 10.0 * np.abs(np.sin(0.1 * values)) + 3e-9 * _zigzag(values, k, m, lmbd) * np.abs(
        (values - 40.0) * (values - 185.0) * values * (values + 50.0) * (values + 180.0)
    )


def _abs_logcos(values: np.ndarray, k: float, m: float, lmbd: float) -> np.ndarray:
    return (
        _zigzag(values, k, m, lmbd) * 3.0 * np.abs(np.log(np.abs(values) * 1000.0 + 1.0))
        + 30.0
        - 30.0 * np.abs(np.cos(values / (math.pi * 10.0)))
    )


def _abs_1(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_poly(x, k=16.0, m=1.0, lmbd=0.01))


def _abs_2(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_poly(x, k=8.0, m=0.5, lmbd=0.01))


def _abs_3(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_poly(_abs_poly(x, k=2.0, m=0.5, lmbd=0.99), k=2.0, m=0.5, lmbd=0.99))


def _abs_4(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_poly(_abs_poly(x, k=1.0, m=1.0, lmbd=0.1), k=1.0, m=1.0, lmbd=0.1))


def _abs_5(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_logcos(x, k=16.0, m=0.9, lmbd=0.01))


def _abs_6(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_logcos(x, k=8.0, m=0.9, lmbd=0.9))


def _abs_7(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_logcos(_abs_logcos(x, k=16.0, m=0.1, lmbd=0.1), k=16.0, m=0.1, lmbd=0.1))


def _abs_8(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    return _sum(_abs_logcos(_abs_logcos(x, k=4.0, m=0.9, lmbd=0.01), k=4.0, m=0.9, lmbd=0.01))


def _pairs(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = _as_batch(theta)
    return x[:, :-1], x[:, 1:]


def _layeb_1(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    with np.errstate(over="ignore", invalid="ignore"):
        return _sum(100.0 * np.abs(np.exp((x - 1.0) ** 2) - 1.0) ** 0.5)


def _layeb_2(theta: np.ndarray) -> np.ndarray:
    x = _as_batch(theta)
    with np.errstate(over="ignore", invalid="ignore"):
        return _sum(np.abs(np.exp(100.0 * ((x - 1.0) ** 2) / (np.exp(x) + 1.0)) - 1.0))


def _layeb_3(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    radial = np.sqrt(xi**2 + xj**2) / math.pi
    terms = -1.0 / (np.abs(np.exp(np.abs(100.0 - radial)) * _sin_rad(xi) + _sin_rad(xj)) + 1.0) ** 0.1
    return _sum(terms)


def _layeb_4(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    return _sum(np.log(np.abs(xj * xi) + 0.001) + _cos_rad(xj + xi))


def _layeb_5(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    numerator = np.log(np.abs(_sin_rad(xi - math.pi / 2.0) + _cos_rad(xj - math.pi)) + 0.001)
    denominator = np.abs(_cos_rad(2.0 * xi - xj + math.pi / 2.0)) + 1.0
    return _sum(numerator / denominator)


def _layeb_6(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    terms = np.abs(_cos_rad(np.sqrt(xi**2 + xj**2)) * _sin_rad(xj) + _cos_rad(xi) + 1.0) ** 0.1
    return _sum(terms)


def _layeb_7(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    terms = 100.0 * np.abs(_cos_rad(xi + xj - math.pi / 2.0)) ** 0.1
    terms = terms - np.exp(_cos_rad(16.0 * xj * xi / math.pi)) + math.exp(1.0)
    return _sum(terms)


def _layeb_8(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    return _sum(np.abs(100.0 * _cos_rad(xi - xj)) + np.log(np.abs(xi + xj) + 0.001))


def _layeb_9(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    numerator = np.exp(np.abs(xj * _sin_rad(xi)) - np.abs(xj)) + _cos_rad(xi + xj)
    denominator = np.exp(_cos_rad(xi + xj) - 1.0)
    return _sum(np.abs(numerator / denominator) ** 0.5)


def _layeb_10(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    terms = np.abs(100.0 * _sin_rad(xi - xj)) + np.log(xi**2 + xj**2 + 0.5) ** 2
    return _sum(terms)


def _layeb_11(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    terms = np.cos(xi * xj + math.pi) / ((100.0 * (xi**2 - xj - 1.0)) ** 2 + 1.0)
    return _sum(terms)


def _layeb_12(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    terms = -(_cos_rad(xi * math.pi / 2.0 - xj * math.pi / 4.0 - math.pi / 2.0) * np.exp(_cos_rad(2.0 * math.pi * xj * xi)) + 1.0)
    return _sum(terms)


def _layeb_13(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    terms = np.abs(_cos_rad(xi - xj)) + 100.0 * np.abs(np.log(np.abs(xi + xj) + 1.0)) ** 0.1
    return _sum(terms)


def _layeb_14(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.abs(np.log((xi + xj + 2.0) ** 2)) + 100.0 * np.abs(xi**2 - xj - 1.0) ** 0.1
    return _sum(terms)


def _layeb_15(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = 10.0 * np.abs(np.tanh(2.0 * np.abs(xi) - xj**2 - 1.0)) ** 0.5 + np.abs(np.exp(xj * xi + 1.0) - 1.0)
    return _sum(terms)


def _layeb_16(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    trig_difference = _cos_rad(xi) ** 2 - _sin_rad(xj) ** 2
    trig_difference = np.where(np.isclose(trig_difference, 0.0, atol=1e-12, rtol=0.0), 0.0, trig_difference)
    terms = np.abs(_tan_rad(xj) * xi + 100.0 * np.abs(trig_difference) - math.pi / 4.0) ** 0.2
    return _sum(terms)


def _layeb_17(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = 1.0 + 10.0 * np.abs(np.log((xi + xj + 2.0) ** 2))
        terms = terms - 1.0 / (np.abs(1e3 * (xi**2 - xj - 1.0)) ** 2 + 1.0)
    return _sum(terms)


def _layeb_18(theta: np.ndarray) -> np.ndarray:
    xi, xj = _pairs(theta)
    numerator = np.log(np.abs(_cos_rad(2.0 * xj * xi / math.pi)) + 0.001)
    denominator = np.abs(_sin_rad(xi + xj) * _cos_rad(xi)) + 1.0
    return _sum(numerator / denominator)


def _zeros(dim: int) -> np.ndarray:
    return np.zeros(dim, dtype=float)


def _ones(dim: int) -> np.ndarray:
    return np.ones(dim, dtype=float)


def _constant(value: float) -> Callable[[int], np.ndarray]:
    return lambda dim: np.full(dim, value, dtype=float)


def _alternating(first: float, second: float) -> Callable[[int], np.ndarray]:
    def values(dim: int) -> np.ndarray:
        out = np.full(dim, first, dtype=float)
        out[1::2] = second
        return out

    return values


def _abs_functions() -> List[DirectGOLibFunction]:
    features = (0, 1, 1, 1, 0, 0, 1, 1)
    return [
        DirectGOLibFunction("ABS_1", "ABS", _abs_1, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_2", "ABS", _abs_2, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_3", "ABS", _abs_3, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_4", "ABS", _abs_4, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_5", "ABS", _abs_5, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_6", "ABS", _abs_6, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_7", "ABS", _abs_7, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
        DirectGOLibFunction("ABS_8", "ABS", _abs_8, -100.0, 100.0, lambda dim: 0.0, _zeros, 1, features),
    ]


def _layeb_functions() -> List[DirectGOLibFunction]:
    return [
        DirectGOLibFunction("Layeb01", "Layeb", _layeb_1, -100.0, 100.0, lambda dim: 0.0, _ones, 1, (1, 1, 1, 0, 0, 1, 0, 1)),
        DirectGOLibFunction("Layeb02", "Layeb", _layeb_2, -10.0, 10.0, lambda dim: 0.0, _ones, 1, (0, 1, 1, 0, 0, 1, 0, 1)),
        DirectGOLibFunction("Layeb03", "Layeb", _layeb_3, -10.0, 10.0, lambda dim: -dim + 1.0, _constant(math.pi), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb04", "Layeb", _layeb_4, -10.0, 10.0, lambda dim: (math.log(0.001) - 1.0) * (dim - 1), _alternating(0.0, math.pi), 2, (1, 0, 1, 1, 0, 0, 1, 1)),
        DirectGOLibFunction("Layeb05", "Layeb", _layeb_5, -10.0, 10.0, lambda dim: math.log(0.001) * (dim - 1), _alternating(2.0 * math.pi, math.pi), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb06", "Layeb", _layeb_6, -10.0, 10.0, lambda dim: 0.0, _constant(math.pi), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb07", "Layeb", _layeb_7, -10.0, 10.0, lambda dim: 0.0, _constant(math.pi), 2, (1, 0, 1, 1, 0, 0, 0, 1)),
        DirectGOLibFunction("Layeb08", "Layeb", _layeb_8, -10.0, 10.0, lambda dim: math.log(0.001) * (dim - 1), _alternating(math.pi / 4.0, -math.pi / 4.0), 2, (1, 0, 1, 1, 0, 0, 0, 1)),
        DirectGOLibFunction("Layeb09", "Layeb", _layeb_9, -10.0, 10.0, lambda dim: 0.0, _constant(math.pi / 2.0), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb10", "Layeb", _layeb_10, -100.0, 100.0, lambda dim: 0.0, _constant(0.5), 2, (0, 0, 1, 1, 0, 0, 0, 1)),
        DirectGOLibFunction("Layeb11", "Layeb", _layeb_11, -10.0, 10.0, lambda dim: -(dim - 1.0), _alternating(-1.0, 0.0), 2, (1, 0, 1, 1, 0, 0, 1, 0)),
        DirectGOLibFunction("Layeb12", "Layeb", _layeb_12, -5.0, 5.0, lambda dim: -(math.exp(1.0) + 1.0) * (dim - 1), _constant(2.0), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb13", "Layeb", _layeb_13, -5.0, 5.0, lambda dim: 0.0, _alternating(math.pi / 4.0, -math.pi / 4.0), 2, (1, 0, 1, 1, 0, 0, 0, 1)),
        DirectGOLibFunction("Layeb14", "Layeb", _layeb_14, -100.0, 100.0, lambda dim: 0.0, _alternating(0.0, -1.0), 2, (1, 0, 1, 1, 0, 0, 1, 0)),
        DirectGOLibFunction("Layeb15", "Layeb", _layeb_15, -100.0, 100.0, lambda dim: 0.0, _alternating(1.0, -1.0), 2, (0, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb16", "Layeb", _layeb_16, -10.0, 10.0, lambda dim: 0.0, _constant(-math.pi / 4.0), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
        DirectGOLibFunction("Layeb17", "Layeb", _layeb_17, -10.0, 10.0, lambda dim: 0.0, _alternating(-1.0, 0.0), 2, (1, 0, 1, 1, 0, 0, 1, 0)),
        DirectGOLibFunction("Layeb18", "Layeb", _layeb_18, -10.0, 10.0, lambda dim: math.log(0.001) * (dim - 1), _constant(math.pi / 2.0), 2, (1, 0, 1, 1, 0, 0, 0, 0)),
    ]


def directgolib_abs_layeb_functions(families: Sequence[str] = ("ABS", "Layeb")) -> List[DirectGOLibFunction]:
    selected = {family.lower() for family in families}
    functions: List[DirectGOLibFunction] = []
    if "abs" in selected:
        functions.extend(_abs_functions())
    if "layeb" in selected:
        functions.extend(_layeb_functions())
    return functions


def _unif(count: int, seed: int) -> np.ndarray:
    inseed = abs(int(seed))
    if inseed < 1:
        inseed = 1
    rgrand = np.zeros(32, dtype=float)
    aktseed = inseed
    for index in range(39, -1, -1):
        tmp = math.floor(aktseed / 127773)
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp
        if aktseed < 0:
            aktseed += 2147483647
        if index < 32:
            rgrand[index] = aktseed
    aktrand = rgrand[0]
    values = np.zeros(count, dtype=float)
    for index in range(count):
        tmp = math.floor(aktseed / 127773)
        aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp
        if aktseed < 0:
            aktseed += 2147483647
        tmp = math.floor(aktrand / 67108865) + 1
        aktrand = rgrand[tmp - 1]
        rgrand[tmp - 1] = aktseed
        values[index] = aktrand / 2147483647.0
    values[values == 0.0] = 1e-15
    return values


def _gauss(count: int, seed: int) -> np.ndarray:
    random_values = _unif(2 * count, seed)
    values = np.sqrt(-2.0 * np.log(random_values[:count])) * np.cos(2.0 * math.pi * random_values[count:])
    values[values == 0.0] = 1e-99
    return values


def _compute_rotation(seed: int, dim: int) -> np.ndarray:
    rotation = np.reshape(_gauss(dim * dim, seed), (dim, dim), order="F").T
    for row in range(dim):
        for previous_row in range(row):
            rotation[row, :] = rotation[row, :] - np.sum(rotation[row, :] * rotation[previous_row, :]) * rotation[previous_row, :]
        rotation[row, :] = rotation[row, :] / math.sqrt(float(np.sum(rotation[row, :] ** 2)))
    return rotation


def _rng_uniform(seed: int, count: int) -> np.ndarray:
    return np.random.RandomState(seed).rand(count)


def _rng_normal(seed: int, count: int) -> np.ndarray:
    return np.random.RandomState(seed).standard_normal(count)


def _shift_parameter_origin(rotation: np.ndarray, xmin: np.ndarray, lower: np.ndarray, upper: np.ndarray, center: np.ndarray) -> np.ndarray:
    return rotation.T @ (xmin + rotation @ center - center)


def _box_tmax(origin: np.ndarray, direction: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    upper_t = math.inf
    for origin_value, direction_value, lower_value, upper_value in zip(origin, direction, lower, upper):
        if abs(direction_value) <= 1e-15:
            if origin_value < lower_value or origin_value > upper_value:
                return 0.0
            continue
        if direction_value > 0.0:
            candidate = (upper_value - origin_value) / direction_value
        else:
            candidate = (lower_value - origin_value) / direction_value
        upper_t = min(upper_t, float(candidate))
    if not np.isfinite(upper_t):
        return 0.0
    return max(0.0, upper_t)


def _compute_shift_bounds(rotation: np.ndarray, xmin: np.ndarray, lower: np.ndarray, upper: np.ndarray, center: np.ndarray, instance: int) -> Tuple[np.ndarray, np.ndarray]:
    origin = _shift_parameter_origin(rotation, xmin, lower, upper, center)
    if np.linalg.norm(center - xmin) < 1e-3:
        direction = _rng_normal(instance, len(xmin))
        shift_min = np.zeros_like(direction)
    else:
        direction = np.clip(origin, lower, upper) - origin
        if np.linalg.norm(direction) < 1e-3:
            direction = _rng_normal(instance, len(xmin))
            shift_min = np.zeros_like(direction)
        else:
            shift_min = direction.copy()
    t_max = _box_tmax(origin, direction, lower, upper)
    shift_max = t_max * direction
    return shift_min, shift_max


def _compute_instance_parameters(function: DirectGOLibFunction, dim: int, instance: int) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.full(dim, function.lower, dtype=float)
    upper = np.full(dim, function.upper, dtype=float)
    center = (lower + upper) / 2.0
    xmin = function.xmin(dim)
    rotation = np.eye(dim) if instance in (1, 2) else _compute_rotation(instance, dim)
    shift_min, shift_max = _compute_shift_bounds(rotation, xmin, lower, upper, center, instance)
    shift = shift_min + 0.1 * (shift_max - shift_min) * _rng_uniform(instance, dim)
    return rotation, shift


def make_directgolib_problem(function: DirectGOLibFunction, dim: int, instance: int) -> DirectGOLibProblem:
    if instance < 1 or instance > 5:
        raise ValueError(f"DIRECTGOLib instance must be in [1, 5], got {instance}")
    if dim < function.min_dimension:
        raise ValueError(f"{function.source_name} requires dimension >= {function.min_dimension}, got {dim}")
    lower = np.full(dim, function.lower, dtype=float)
    upper = np.full(dim, function.upper, dtype=float)
    bounds = np.column_stack([lower, upper])
    center = (lower + upper) / 2.0
    xmin = function.xmin(dim)
    rotation, shift = _compute_instance_parameters(function, dim, instance)
    temp_vec = -rotation @ shift - rotation @ center + center
    transformed_xmin = rotation.T @ (xmin - center) + shift + center

    def objective(theta: np.ndarray) -> np.ndarray:
        theta_batch = _as_batch(theta)
        transformed = theta_batch @ rotation.T + temp_vec
        transformed = np.minimum(np.maximum(transformed, lower), upper)
        transformed = np.where(np.isclose(transformed, xmin, atol=1e-12, rtol=0.0), xmin, transformed)
        return function.objective(transformed)

    rotated_label = "shift+rot" if instance >= 3 else "shift"
    name = f"DIRECTGOLib {function.source_name} ({rotated_label} inst={instance}) ({dim}D)"
    return DirectGOLibProblem(
        name=name,
        family=function.family,
        source_name=function.source_name,
        dimension=dim,
        instance=instance,
        objective=objective,
        bounds=bounds,
        fmin=float(function.fmin(dim)),
        xmin=transformed_xmin,
        shift=shift,
        rotation=rotation,
        features=function.features,
    )


def build_directgolib_abs_layeb_suite(
    dim: int,
    instances: Sequence[int] = (1, 2, 3, 4, 5),
    families: Sequence[str] = ("ABS", "Layeb"),
) -> List[DirectGOLibProblem]:
    problems: List[DirectGOLibProblem] = []
    for function in directgolib_abs_layeb_functions(families):
        if dim < function.min_dimension:
            continue
        for instance in instances:
            problems.append(make_directgolib_problem(function, dim, instance))
    return problems


def run_cds_box(problem: DirectGOLibProblem, budget: int, seed: int, step_size: float, num_cells: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    current_seed = seed
    while tracker.evaluations < budget:
        previous = tracker.evaluations
        optimizer = BoxCellularDirectSearch(problem.bounds, step_size, tracker, num_cells, current_seed)
        optimizer.run(evaluation_budget=budget - tracker.evaluations)
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def _box_arrays(problem: DirectGOLibProblem) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = problem.bounds[:, 0]
    upper = problem.bounds[:, 1]
    span = upper - lower
    return lower, upper, span


def run_cma_es_box(problem: DirectGOLibProblem, budget: int, seed: int, sigma_scale: float) -> Tuple[np.ndarray, float]:
    try:
        import cma
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, span = _box_arrays(problem)
    current_seed = seed
    sigma = sigma_scale * float(np.mean(span) / 2.0)
    while tracker.evaluations < budget:
        previous = tracker.evaluations
        rng = np.random.default_rng(current_seed)
        x0 = rng.uniform(lower, upper)
        strategy = cma.CMAEvolutionStrategy(
            x0,
            sigma,
            {"maxfevals": budget - tracker.evaluations, "verbose": -9, "bounds": [lower.tolist(), upper.tolist()], "seed": current_seed},
        )
        strategy.optimize(tracker)
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_pso_box(problem: DirectGOLibProblem, budget: int, seed: int, swarmsize: int) -> Tuple[np.ndarray, float]:
    try:
        from pyswarm import pso
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    current_seed = seed
    while tracker.evaluations < budget:
        previous = tracker.evaluations
        np.random.seed(current_seed)
        max_iterations = max(1, int((budget - tracker.evaluations) / swarmsize))
        pso(tracker, lower, upper, swarmsize=swarmsize, maxiter=max_iterations, debug=False)
        if tracker.evaluations <= previous + swarmsize:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_differential_evolution_box(problem: DirectGOLibProblem, budget: int, seed: int, popsize: int = 15) -> Tuple[np.ndarray, float]:
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    bounds = [tuple(row) for row in problem.bounds]
    current_seed = seed
    while tracker.evaluations < budget:
        previous = tracker.evaluations
        max_iterations = max(1, (budget - tracker.evaluations) // (popsize * problem.dimension) - 1)
        try:
            differential_evolution(tracker, bounds, maxiter=max_iterations, popsize=popsize, seed=current_seed, polish=False)
        except Exception:
            pass
        if tracker.evaluations <= previous + popsize:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_neldermead_box(problem: DirectGOLibProblem, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        from scipy.optimize import minimize
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    bounds = [tuple(row) for row in problem.bounds]
    current_seed = seed

    def objective_wrapper(x: np.ndarray) -> float:
        value = tracker(np.clip(x, lower, upper))
        return float(value) if np.isfinite(value) else 1e300

    while tracker.evaluations < budget:
        previous = tracker.evaluations
        x0 = np.random.default_rng(current_seed).uniform(lower, upper)
        try:
            minimize(objective_wrapper, x0, method="Nelder-Mead", bounds=bounds, options={"maxfev": budget - tracker.evaluations, "disp": False}, tol=1e-6)
        except Exception:
            pass
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_pdfo_box(problem: DirectGOLibProblem, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import pdfo
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    current_seed = seed
    while tracker.evaluations < budget:
        previous = tracker.evaluations
        x0 = np.random.default_rng(current_seed).uniform(lower, upper)
        try:
            pdfo.pdfo(tracker, x0, bounds=problem.bounds, options={"maxfev": budget - tracker.evaluations})
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"PDFO unavailable; skipping this run ({exc})")
            break
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_bads_box(problem: DirectGOLibProblem, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        from pybads import BADS
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, span = _box_arrays(problem)
    plausible_lower = lower + 0.25 * span
    plausible_upper = upper - 0.25 * span
    current_seed = seed

    def safe_obj(x: np.ndarray) -> float:
        value = tracker(np.clip(x, lower, upper))
        return float(value) if np.isfinite(value) else 1e300

    while tracker.evaluations < budget:
        remaining = budget - tracker.evaluations
        if remaining < problem.dimension + 2:
            break
        previous = tracker.evaluations
        x0 = np.random.default_rng(current_seed).uniform(plausible_lower, plausible_upper)
        try:
            bads = BADS(safe_obj, x0, lower, upper, plausible_lower, plausible_upper, options={"max_fun_evals": remaining, "display": "off"})
            bads.optimize()
        except Exception:
            print("Error in BADS optimization")
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_lshade_box(problem: DirectGOLibProblem, budget: int, seed: int, pop_factor: int) -> Tuple[np.ndarray, float]:
    try:
        from mealpy import FloatVar
        from mealpy.evolutionary_based import SHADE
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    current_seed = seed

    def obj_w(x: np.ndarray) -> float:
        value = tracker(np.clip(x, lower, upper))
        return float(value) if np.isfinite(value) else 1e300

    while tracker.evaluations < budget:
        previous = tracker.evaluations
        problem_dict = {"bounds": FloatVar(lb=lower.tolist(), ub=upper.tolist()), "minmax": "min", "obj_func": obj_w, "log_to": None}
        model = SHADE.L_SHADE(epoch=10000, pop_size=pop_factor * problem.dimension)
        model.solve(problem_dict, seed=current_seed, termination={"max_fe": budget - tracker.evaluations})
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_nomad_box(problem: DirectGOLibProblem, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    try:
        import PyNomad
    except ImportError:
        return np.empty((0, 2)), 0.0
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    current_seed = seed
    while tracker.evaluations < budget:
        previous = tracker.evaluations
        x0 = np.random.default_rng(current_seed).uniform(lower, upper).tolist()

        def bb(x):
            coords = np.array([x.get_coord(index) for index in range(x.size())])
            value = tracker(np.clip(coords, lower, upper))
            x.setBBO(str(value).encode("UTF-8"))
            return 1

        params = ["BB_OUTPUT_TYPE OBJ", f"MAX_BB_EVAL {budget - tracker.evaluations}", "DISPLAY_DEGREE 0", f"SEED {current_seed}", "DIRECTION_TYPE ORTHO 2n", "QUAD_MODEL_SEARCH NO"]
        try:
            PyNomad.optimize(bb, x0, lower.tolist(), upper.tolist(), params)
        except Exception:
            pass
        if tracker.evaluations <= previous + 1:
            break
        current_seed += 1
    return tracker.history_array(), tracker.elapsed_time()


def run_random_search_box(problem: DirectGOLibProblem, budget: int, seed: int) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    rng = np.random.default_rng(seed)
    while tracker.evaluations < budget:
        tracker(rng.uniform(lower, upper))
    return tracker.history_array(), tracker.elapsed_time()


def run_grid_search_box(problem: DirectGOLibProblem, budget: int, seed: int, step_size: float) -> Tuple[np.ndarray, float]:
    tracker = ObjectiveTracker(problem.objective)
    lower, upper, _ = _box_arrays(problem)
    rng = np.random.default_rng(seed)
    while tracker.evaluations < budget:
        point = rng.uniform(lower, upper)
        snapped = lower + np.rint((point - lower) / step_size) * step_size
        tracker(np.clip(snapped, lower, upper))
    return tracker.history_array(), tracker.elapsed_time()


def build_directgolib_optimizer_configs(settings: DirectGOLibBenchmarkSettings) -> List[OptimizerConfig]:
    configs: List[OptimizerConfig] = []
    if settings.include_cds:
        for h, n_cells in itertools.product(settings.cds_h_list, settings.cds_n_list):
            configs.append(OptimizerConfig(f"CDS-Box (h={h}, N={n_cells})", run_cds_box, {"step_size": h, "num_cells": n_cells}))
    if settings.include_cmaes:
        for sigma in settings.cma_sigma_list:
            configs.append(OptimizerConfig(f"CMA-ES (sigma={sigma})", run_cma_es_box, {"sigma_scale": sigma}))
    if settings.include_pso:
        for swarm in settings.pso_swarm_list:
            configs.append(OptimizerConfig(f"PSO (swarm={swarm})", run_pso_box, {"swarmsize": swarm}))
    if settings.include_lshade:
        for factor in settings.lshade_pop_factors:
            configs.append(OptimizerConfig(f"L-SHADE (pop={factor}d)", run_lshade_box, {"pop_factor": factor}))
    if settings.include_bads:
        configs.append(OptimizerConfig("BADS", run_bads_box, {}))
    if settings.include_nomad:
        configs.append(OptimizerConfig("NOMAD", run_nomad_box, {}))
    if settings.include_pdfo:
        configs.append(OptimizerConfig("Powell (PDFO)", run_pdfo_box, {}))
    if settings.include_neldermead:
        configs.append(OptimizerConfig("Nelder-Mead", run_neldermead_box, {}))
    if settings.include_de:
        configs.append(OptimizerConfig("DE (pop=15d)", run_differential_evolution_box, {"popsize": 15}))
    if settings.include_grid_search:
        configs.append(OptimizerConfig("Pure Grid Search (h=0.25)", run_grid_search_box, {"step_size": 0.25}))
    if settings.include_random:
        configs.append(OptimizerConfig("Random Search", run_random_search_box, {}))
    return configs


def run_directgolib_benchmark(settings: Optional[DirectGOLibBenchmarkSettings] = None, output_csv: Optional[str] = None):
    config = settings or DirectGOLibBenchmarkSettings()
    optimizer_configs = build_directgolib_optimizer_configs(config)
    columns = [
        "Problem",
        "Family",
        "Source",
        "Dimension",
        "Instance",
        "Rotated",
        "Optimizer",
        "Seed",
        "Final Loss",
        "True Opt",
        "Opt Gap",
        "Total Evals",
        "Time (s)",
        "Shift Norm",
    ]
    records = []
    csv_handle = None
    csv_writer = None
    if output_csv is not None:
        csv_handle = open(output_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_handle, fieldnames=columns)
        csv_writer.writeheader()
    try:
        for dim in config.dimensions:
            problems = build_directgolib_abs_layeb_suite(dim=dim, instances=config.instances, families=config.families)
            for problem in problems:
                print(f"\n===== DIRECTGOLib PROBLEM: {problem.name} =====")
                for optimizer in optimizer_configs:
                    opt_records = []
                    for seed in config.seeds:
                        print(f"  -> {optimizer.name} | Seed {seed}")
                        history, elapsed = optimizer.runner(problem, config.budget_evaluations, seed, **optimizer.params)
                        final_loss = float(history[-1, 1]) if history.size > 0 else float(np.inf)
                        opt_gap = max(0.0, final_loss - problem.fmin) if np.isfinite(final_loss) else float(np.inf)
                        record = {
                            "Problem": problem.name,
                            "Family": problem.family,
                            "Source": problem.source_name,
                            "Dimension": problem.dimension,
                            "Instance": problem.instance,
                            "Rotated": int(problem.instance >= 3),
                            "Optimizer": optimizer.name,
                            "Seed": seed,
                            "Final Loss": final_loss,
                            "True Opt": problem.fmin,
                            "Opt Gap": opt_gap,
                            "Total Evals": int(history[-1, 0]) if history.size > 0 else int(config.budget_evaluations),
                            "Time (s)": float(elapsed),
                            "Shift Norm": float(np.linalg.norm(problem.shift)),
                        }
                        records.append(record)
                        opt_records.append(record)
                        if csv_writer is not None:
                            csv_writer.writerow(record)
                            csv_handle.flush()
                        print(f"     done | loss={record['Final Loss']:.6e} | gap={record['Opt Gap']:.6e} | evals={record['Total Evals']} | time={record['Time (s)']:.2f}s")
                    mean_gap = float(np.mean([item["Opt Gap"] for item in opt_records]))
                    mean_time = float(np.mean([item["Time (s)"] for item in opt_records]))
                    print(f"  => {optimizer.name} summary | mean_gap={mean_gap:.6e} | mean_time={mean_time:.2f}s")
    finally:
        if csv_handle is not None:
            csv_handle.close()
    try:
        import pandas as pd

        return pd.DataFrame(records)
    except ImportError:
        return records


__all__ = [
    "DirectGOLibBenchmarkSettings",
    "DirectGOLibFunction",
    "DirectGOLibProblem",
    "FEATURE_NAMES",
    "build_directgolib_abs_layeb_suite",
    "directgolib_abs_layeb_functions",
    "make_directgolib_problem",
    "run_directgolib_benchmark",
]
