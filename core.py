from __future__ import annotations
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
CellIndex = Tuple[int, ...]

class ObjectiveTracker:

    def __init__(self, objective_function: Callable[[np.ndarray], np.ndarray | float]) -> None:
        self.objective_function = objective_function
        self.evaluations = 0
        self.best_value = np.inf
        self.history: List[Tuple[int, float]] = []
        self.start_time: Optional[float] = None

    def __call__(self, theta: np.ndarray) -> np.ndarray | float:
        if self.start_time is None:
            self.start_time = time.time()
        theta_array = np.asarray(theta, dtype=float)
        theta_batch = np.atleast_2d(theta_array)
        raw_values = self.objective_function(theta_batch)
        values = np.asarray(raw_values, dtype=float).reshape(-1)
        if values.size == 1 and theta_batch.shape[0] > 1:
            values = np.repeat(values, theta_batch.shape[0])
        for index in range(theta_batch.shape[0]):
            self.evaluations += 1
            current_value = float(values[index])
            if current_value < self.best_value:
                self.best_value = current_value
            self.history.append((self.evaluations, self.best_value))
        if theta_array.ndim == 1:
            return float(values[0])
        return values

    def reset(self) -> None:
        self.evaluations = 0
        self.best_value = np.inf
        self.history = []
        self.start_time = None

    def history_array(self) -> np.ndarray:
        if not self.history:
            return np.empty((0, 2), dtype=float)
        return np.asarray(self.history, dtype=float)

    def elapsed_time(self) -> float:
        if self.start_time is None:
            return 0.0
        return float(time.time() - self.start_time)

class AbstractCellularDirectSearch(ABC):

    def _as_cell(self, cell: Sequence[int] | np.ndarray) -> CellIndex:
        return tuple((int(value) for value in np.asarray(cell, dtype=int).tolist()))

    def _evaluation_count(self) -> int:
        if hasattr(self.objective_function, 'evaluations'):
            return int(getattr(self.objective_function, 'evaluations'))
        if hasattr(self.objective_function, 'n_evals'):
            return int(getattr(self.objective_function, 'n_evals'))
        return 0

    @abstractmethod
    def cell_to_coordinates(self, cell: CellIndex | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def is_in_domain(self, cell: CellIndex | np.ndarray) -> bool:
        raise NotImplementedError

    @abstractmethod
    def _initialize_active_cells(self) -> None:
        raise NotImplementedError

    def evaluate_cell(self, cell: CellIndex | np.ndarray) -> float:
        cell_index = self._as_cell(cell)
        if cell_index in self.objective_cache:
            self.cache_hits += 1
            return self.objective_cache[cell_index]
        self.cache_misses += 1
        coordinates = self.cell_to_coordinates(cell_index)
        value = float(self.objective_function(coordinates))
        self.objective_cache[cell_index] = value
        if value < self.best_value:
            self.best_value = value
        return value

    def _register_root_path(self, old_cell: CellIndex, new_cell: CellIndex) -> bool:
        if not self.track_roots:
            return False
        if new_cell in self.cell_to_root:
            return True
        if old_cell in self.cell_to_root:
            root = self.cell_to_root[old_cell]
            self.cell_paths[root].append(new_cell)
            self.cell_to_root[new_cell] = root
        return False

    def jealous_neighbor_update(self) -> None:
        surviving_cells: Set[CellIndex] = set()
        for cell in self.active_cells:
            cell_array = np.asarray(cell, dtype=int)
            neighbors = cell_array + self.neighbor_offsets
            active_neighbors = [self._as_cell(neighbor) for neighbor in neighbors if self._as_cell(neighbor) in self.active_cells]
            if not active_neighbors:
                surviving_cells.add(cell)
                continue
            cell_value = self.evaluate_cell(cell)
            best_neighbor_value = min((self.evaluate_cell(neighbor) for neighbor in active_neighbors))
            if cell_value <= best_neighbor_value:
                surviving_cells.add(cell)
        self.active_cells = surviving_cells

    def solitary_regeneration_update(self) -> None:
        updated_cells = set(self.active_cells)
        candidates = self.active_cells - self.fixed_cells
        for cell in candidates:
            cell_array = np.asarray(cell, dtype=int)
            neighbors = cell_array + self.neighbor_offsets
            has_active_neighbor = any((self._as_cell(neighbor) in self.active_cells for neighbor in neighbors))
            if has_active_neighbor:
                continue
            current_value = self.evaluate_cell(cell)
            best_neighbor: Optional[np.ndarray] = None
            best_neighbor_value = current_value
            for neighbor in neighbors:
                if not self.is_in_domain(neighbor):
                    continue
                neighbor_value = self.evaluate_cell(neighbor)
                if neighbor_value < best_neighbor_value:
                    best_neighbor_value = neighbor_value
                    best_neighbor = neighbor
            if best_neighbor is None:
                self.fixed_cells.add(cell)
                continue
            changed_axis = int(np.where(cell_array != best_neighbor)[0][0])
            denominator = max(self.step_size[changed_axis], 1e-12)
            step_magnitude = max(1, int((current_value - best_neighbor_value) / denominator))
            trial = cell_array.copy()
            direction = int(np.sign(best_neighbor[changed_axis] - cell_array[changed_axis]))
            trial[changed_axis] += direction * step_magnitude
            trial_cell = self._as_cell(trial)
            best_neighbor_cell = self._as_cell(best_neighbor)
            if self.is_in_domain(trial) and self.evaluate_cell(trial_cell) < current_value:
                target = trial_cell
            else:
                target = best_neighbor_cell
            if not self._register_root_path(cell, target):
                updated_cells.add(target)
            updated_cells.discard(cell)
        self.active_cells = updated_cells

    def step(self) -> bool:
        previous_state = set(self.active_cells)
        self.timestep += 1
        self.solitary_regeneration_update()
        self.jealous_neighbor_update()
        return self.active_cells != previous_state

    def run(self, evaluation_budget: int, max_steps: Optional[int]=None) -> None:
        steps = 0
        while self._evaluation_count() < evaluation_budget:
            changed = self.step()
            steps += 1
            if not changed:
                break
            if max_steps is not None and steps >= max_steps:
                break

class SphereCellularDirectSearch(AbstractCellularDirectSearch):

    def __init__(self, n_variables: int, radius: float, step_size: float | np.ndarray, objective_function: Callable[[np.ndarray], np.ndarray | float], num_active_cells: Optional[int]=None, seed: Optional[int]=None, track_roots: bool=True, legacy_initialization: bool=True) -> None:
        self.n = int(n_variables)
        self.radius = float(radius)
        self.step_size = np.full(self.n, step_size, dtype=float) if np.isscalar(step_size) else np.asarray(step_size, dtype=float)
        if self.step_size.shape != (self.n,):
            raise ValueError(f'step_size must have shape ({self.n},), got {self.step_size.shape}')
        self.objective_function = objective_function
        self.track_roots = bool(track_roots)
        self.legacy_initialization = bool(legacy_initialization)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.grid_shape = np.floor(2.0 * self.radius / self.step_size).astype(int) + 1
        offsets = []
        for axis in range(self.n):
            for direction in (-1, 1):
                offset = np.zeros(self.n, dtype=int)
                offset[axis] = direction
                offsets.append(offset)
        self.neighbor_offsets = np.asarray(offsets, dtype=int)
        self.active_cells: Set[CellIndex] = set()
        self.cell_paths: Dict[CellIndex, List[CellIndex]] = {}
        self.cell_to_root: Dict[CellIndex, CellIndex] = {}
        self.fixed_cells: Set[CellIndex] = set()
        self.objective_cache: Dict[CellIndex, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        total_cells = int(np.prod(self.grid_shape))
        self.num_active_cells = int(num_active_cells) if num_active_cells is not None else max(1, int(0.01 * total_cells))
        self.timestep = 0
        self.best_value = np.inf
        self._initialize_active_cells()

    def cell_to_coordinates(self, cell: CellIndex | np.ndarray) -> np.ndarray:
        return -self.radius + np.multiply(np.asarray(cell, dtype=float), self.step_size)

    def is_in_domain(self, cell: CellIndex | np.ndarray) -> bool:
        coordinates = self.cell_to_coordinates(cell)
        return bool(np.linalg.norm(coordinates) <= self.radius)

    def _initialize_active_cells(self) -> None:
        directions = np.random.normal(size=(self.num_active_cells, self.n))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        if self.legacy_initialization:
            uniform = np.random.uniform(0.0, 0.8 * self.radius, size=self.num_active_cells)
            coordinates = 0.8 * self.radius * uniform[:, np.newaxis] * directions / norms
        else:
            radii = np.random.uniform(0.0, 0.8 * self.radius, size=(self.num_active_cells, 1))
            coordinates = radii * directions / norms
        raw_indices = (coordinates + self.radius) / self.step_size
        indices = np.rint(raw_indices).astype(int)
        indices = np.clip(indices, 0, self.grid_shape - 1)
        for index in indices:
            cell = self._as_cell(index)
            self.active_cells.add(cell)
            self.cell_paths[cell] = [cell]
            self.cell_to_root[cell] = cell
            self.evaluate_cell(cell)

class BoxCellularDirectSearch(AbstractCellularDirectSearch):

    def __init__(self, bounds: Sequence[Tuple[float, float]], step_size: float | np.ndarray, objective_function: Callable[[np.ndarray], np.ndarray | float], num_active_cells: int=4, seed: Optional[int]=None) -> None:
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError(f'bounds must have shape (n, 2), got {self.bounds.shape}')
        self.n = int(len(bounds))
        self.step_size = np.full(self.n, step_size, dtype=float) if np.isscalar(step_size) else np.asarray(step_size, dtype=float)
        if self.step_size.shape != (self.n,):
            raise ValueError(f'step_size must have shape ({self.n},), got {self.step_size.shape}')
        span = self.bounds[:, 1] - self.bounds[:, 0]
        if np.any(span <= 0.0):
            raise ValueError('each bound must satisfy lower < upper')
        self.objective_function = objective_function
        self.track_roots = False
        self.legacy_initialization = False
        self.num_active_cells = int(num_active_cells)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.grid_shape = np.floor(span / self.step_size).astype(int) + 1
        offsets = []
        for axis in range(self.n):
            for direction in (-1, 1):
                offset = np.zeros(self.n, dtype=int)
                offset[axis] = direction
                offsets.append(offset)
        self.neighbor_offsets = np.asarray(offsets, dtype=int)
        self.active_cells: Set[CellIndex] = set()
        self.cell_paths: Dict[CellIndex, List[CellIndex]] = {}
        self.cell_to_root: Dict[CellIndex, CellIndex] = {}
        self.fixed_cells: Set[CellIndex] = set()
        self.objective_cache: Dict[CellIndex, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.active_agents: Dict[CellIndex, int] = {}
        self.next_agent_id = 0
        self.timestep = 0
        self.best_value = np.inf
        self.history: List[Tuple[int, float, int]] = []
        self._initialize_active_cells()

    @property
    def total_evaluations(self) -> int:
        return self.cache_misses

    def evaluate(self, cell: CellIndex | np.ndarray) -> float:
        return self.evaluate_cell(cell)

    def coordinates_to_cell(self, coordinates: np.ndarray) -> CellIndex:
        raw_index = (np.asarray(coordinates, dtype=float) - self.bounds[:, 0]) / self.step_size
        index = np.rint(raw_index).astype(int)
        index = np.clip(index, 0, self.grid_shape - 1)
        return self._as_cell(index)

    def cell_to_coordinates(self, cell: CellIndex | np.ndarray) -> np.ndarray:
        return self.bounds[:, 0] + np.multiply(np.asarray(cell, dtype=float), self.step_size)

    def is_in_domain(self, cell: CellIndex | np.ndarray) -> bool:
        coordinates = self.cell_to_coordinates(cell)
        return bool(np.all(coordinates >= self.bounds[:, 0]) and np.all(coordinates <= self.bounds[:, 1]))

    def is_valid(self, cell: CellIndex | np.ndarray) -> bool:
        return self.is_in_domain(cell)

    def _sync_active_agents(self) -> None:
        active_set = set(self.active_cells)
        for cell in list(self.active_agents.keys()):
            if cell not in active_set:
                self.active_agents.pop(cell)
        for cell in sorted(self.active_cells):
            if cell not in self.active_agents:
                self.active_agents[cell] = self.next_agent_id
                self.next_agent_id += 1

    def _initialize_active_cells(self) -> None:
        count = 0
        attempts = 0
        while count < self.num_active_cells and attempts < 2000:
            candidate = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            cell = self.coordinates_to_cell(candidate)
            if cell not in self.active_cells and self.is_in_domain(cell):
                self.active_cells.add(cell)
                self.cell_paths[cell] = [cell]
                self.cell_to_root[cell] = cell
                self.evaluate_cell(cell)
                count += 1
            attempts += 1
        self._sync_active_agents()
        self._record_history()

    def _record_history(self) -> None:
        best_so_far = self.best_value
        if self.history:
            best_so_far = min(best_so_far, self.history[-1][1])
        self.history.append((self.timestep, best_so_far, self.total_evaluations))

    def get_neighbors(self, cell: CellIndex) -> List[CellIndex]:
        cell_array = np.asarray(cell, dtype=int)
        neighbors = cell_array + self.neighbor_offsets
        return [self._as_cell(neighbor) for neighbor in neighbors if self.is_in_domain(neighbor)]

    def step(self) -> None:
        super().step()
        self._sync_active_agents()
        self._record_history()

    def run(self, evaluation_budget: int, max_steps: Optional[int]=None) -> None:
        steps = 0
        while self.total_evaluations < evaluation_budget:
            previous_evaluations = self.total_evaluations
            previous_agents = len(self.active_agents)
            self.step()
            steps += 1
            if not self.active_agents:
                break
            if self.total_evaluations == previous_evaluations and len(self.active_agents) == previous_agents:
                break
            if max_steps is not None and steps >= max_steps:
                break

    def best_curve(self) -> np.ndarray:
        points = [(evaluations, best) for _, best, evaluations in self.history if evaluations > 0]
        if not points:
            return np.empty((0, 2), dtype=float)
        return np.asarray(points, dtype=float)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
