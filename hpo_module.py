from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
from skopt import gp_minimize
from skopt.space import Real
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from core import BoxCellularDirectSearch, set_global_seed


def _configure_matplotlib_backend() -> None:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass


_configure_matplotlib_backend()


@dataclass(frozen=True)
class HPOSettings:
    seed_value: int = 42
    seeds: Tuple[int, ...] = (42, 100, 123, 999, 2025, 7, 13, 99, 101, 555)
    dimension: int = 5
    budget: int = 100
    step_size: float = 2.0
    num_active_cells: int = 1
    epochs: int = 4
    train_size: int = 4000
    validation_size: int = 1000
    train_batch_size: int = 64
    validation_batch_size: int = 1000
    dataset_dir: str = "./data"


class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate: float, n_filters: int, n_dense_units: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 3, 1)
        self.conv2 = nn.Conv2d(n_filters, n_filters * 2, 3, 1)
        self.dropout1 = nn.Dropout(dropout_rate)
        flat_dimension = n_filters * 2 * 5 * 5
        self.fc1 = nn.Linear(flat_dimension, n_dense_units)
        self.fc2 = nn.Linear(n_dense_units, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.conv1(inputs)
        output = torch.relu(output)
        output = torch.max_pool2d(output, 2)
        output = self.conv2(output)
        output = torch.relu(output)
        output = torch.max_pool2d(output, 2)
        output = torch.flatten(output, 1)
        output = self.dropout1(output)
        output = self.fc1(output)
        output = torch.relu(output)
        output = self.fc2(output)
        return torch.log_softmax(output, dim=1)


class FashionMNISTObjective:
    def __init__(self, settings: HPOSettings) -> None:
        self.settings = settings
        set_global_seed(settings.seed_value)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = datasets.FashionMNIST(settings.dataset_dir, train=True, download=True, transform=transform)

        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        train_end = settings.train_size
        validation_end = settings.train_size + settings.validation_size
        train_indices = indices[:train_end]
        validation_indices = indices[train_end:validation_end]

        generator = torch.Generator()
        generator.manual_seed(settings.seed_value)

        self.train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=settings.train_batch_size,
            shuffle=True,
            generator=generator,
        )
        self.validation_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=settings.validation_batch_size,
            shuffle=False,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

    @staticmethod
    def decode_parameters(params: np.ndarray) -> Tuple[float, float, float, int, int]:
        learning_rate = 10 ** (-5.0 + (params[0] / 10.0) * 4.0)
        dropout = (params[1] / 10.0) * 0.7
        weight_decay = 10 ** (-6.0 + (params[2] / 10.0) * 4.0)
        n_filters = int(4 + (params[3] / 10.0) * 60)
        n_dense_units = int(16 + (params[4] / 10.0) * 112)
        return learning_rate, dropout, weight_decay, n_filters, n_dense_units

    def __call__(self, params: np.ndarray) -> float:
        params_array = np.asarray(params, dtype=float)
        learning_rate, dropout, weight_decay, n_filters, n_dense_units = self.decode_parameters(params_array)

        torch.manual_seed(self.settings.seed_value)

        model = SimpleCNN(dropout, n_filters, n_dense_units).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.NLLLoss()

        model.train()
        for _ in range(self.settings.epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                correct += prediction.eq(target.view_as(prediction)).sum().item()

        accuracy = correct / len(self.validation_loader.dataset)
        return float(1.0 - accuracy)


def interpolate_run(history_evaluations: List[int], history_loss: List[float], common_x: np.ndarray) -> np.ndarray:
    interpolator = interpolate.interp1d(
        history_evaluations,
        history_loss,
        kind="previous",
        bounds_error=False,
        fill_value=(history_loss[0], history_loss[-1]),
    )
    return interpolator(common_x)


def process_multi_runs(
    results_dict: Dict[str, List[Tuple[List[int], List[float]]]],
    budget: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    common_x = np.arange(1, budget + 1)
    processed: Dict[str, Dict[str, np.ndarray]] = {}

    for algorithm_name, runs in results_dict.items():
        interpolated_runs = []

        for run_x, run_y in runs:
            if len(run_x) == 0:
                continue
            interpolated_curve = interpolate_run(run_x, run_y, common_x)
            interpolated_runs.append(interpolated_curve)

        if not interpolated_runs:
            continue

        matrix = np.asarray(interpolated_runs)
        processed[algorithm_name] = {
            "x": common_x,
            "mean": np.mean(matrix, axis=0),
            "min": np.min(matrix, axis=0),
            "max": np.max(matrix, axis=0),
        }

    return processed


def generate_pgfplots_code(processed_data: Dict[str, Dict[str, np.ndarray]], sample_rate: int = 5) -> None:
    print("\n% ================= PGFPLOTS CODE =================\n")
    print("\\begin{tikzpicture}")
    print("\\begin{axis}[")
    print("    xlabel={Evaluations},")
    print("    ylabel={Validation Error},")
    print("    grid=major,")
    print("    legend pos=north east,")
    print("    width=0.9\\columnwidth,")
    print("    height=6cm,")
    print("]")

    colors = {"CDS": "red", "Random Search": "gray", "Bayesian Optimization": "green"}

    for name, data in processed_data.items():
        x_values = data["x"][::sample_rate]
        y_values = data["mean"][::sample_rate]
        color = colors.get(name, "blue")

        coordinates = " ".join(f"({x},{y:.4f})" for x, y in zip(x_values, y_values))
        print(f"\\addplot[color={color}, line width=1.5pt] coordinates {{{coordinates}}};")
        print(f"\\addlegendentry{{{name}}}")

    print("\\end{axis}")
    print("\\end{tikzpicture}")
    print("\n% ===============================================\n")


def plot_processed_results(processed_data: Dict[str, Dict[str, np.ndarray]], n_seeds: int) -> None:
    plt.figure(figsize=(10, 6))
    colors = {"CDS": "red", "Random Search": "gray", "Bayesian Optimization": "green"}

    for name, data in processed_data.items():
        x_values = data["x"]
        y_mean = data["mean"]
        y_min = data["min"]
        y_max = data["max"]
        color = colors.get(name, "blue")

        plt.plot(x_values, y_mean, label=name, color=color, linewidth=2)
        plt.fill_between(x_values, y_min, y_max, color=color, alpha=0.15)

    plt.xlabel("Evaluations")
    plt.ylabel("Best Validation Error")
    plt.title(f"HPO on Fashion-MNIST (Average over {n_seeds} seeds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def run_robust_benchmark(
    settings: Optional[HPOSettings] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, List[float]], Dict[str, List[Tuple[List[int], List[float]]]]]:
    config = settings or HPOSettings()
    objective = FashionMNISTObjective(config)

    bounds = [(0.0, 10.0)] * config.dimension

    times: Dict[str, List[float]] = {
        "CDS": [],
        "Random Search": [],
        "Bayesian Optimization": [],
    }

    raw_results: Dict[str, List[Tuple[List[int], List[float]]]] = {
        "CDS": [],
        "Random Search": [],
        "Bayesian Optimization": [],
    }

    print(f"Starting robust benchmark on Fashion-MNIST with {len(config.seeds)} seeds...")

    for run_index, seed in enumerate(config.seeds, start=1):
        print(f"\n--- RUN {run_index}/{len(config.seeds)} (seed {seed}) ---")

        print("Running CDS...")
        start_time = time.time()

        cds_evaluations = 0
        cds_best = float("inf")
        cds_x: List[int] = []
        cds_y: List[float] = []

        def cds_wrapper(point: np.ndarray) -> float:
            nonlocal cds_evaluations, cds_best
            value = objective(point)
            cds_evaluations += 1
            if value < cds_best:
                cds_best = value
            cds_x.append(cds_evaluations)
            cds_y.append(cds_best)
            return value

        cds = BoxCellularDirectSearch(
            bounds=bounds,
            step_size=config.step_size,
            objective_function=cds_wrapper,
            num_active_cells=config.num_active_cells,
            seed=seed,
        )

        while cds_evaluations < config.budget:
            previous_evaluations = cds.total_evaluations
            cds.step()
            if len(cds.active_agents) == 0 or cds.total_evaluations == previous_evaluations:
                break

        raw_results["CDS"].append((cds_x, cds_y))
        times["CDS"].append(time.time() - start_time)

        print("Running Random Search...")
        start_time = time.time()

        np.random.seed(seed)
        rs_best = float("inf")
        rs_x: List[int] = []
        rs_y: List[float] = []

        for evaluation in range(1, config.budget + 1):
            point = np.random.uniform(0.0, 10.0, config.dimension)
            value = objective(point)
            if value < rs_best:
                rs_best = value
            rs_x.append(evaluation)
            rs_y.append(rs_best)

        raw_results["Random Search"].append((rs_x, rs_y))
        times["Random Search"].append(time.time() - start_time)

        print("Running Bayesian Optimization...")
        start_time = time.time()

        bo_best = float("inf")
        bo_x: List[int] = []
        bo_y: List[float] = []

        def bo_wrapper(point: List[float]) -> float:
            nonlocal bo_best
            value = objective(np.asarray(point, dtype=float))
            if value < bo_best:
                bo_best = value
            bo_x.append(len(bo_x) + 1)
            bo_y.append(bo_best)
            return value

        try:
            space = [Real(0.0, 10.0) for _ in range(config.dimension)]
            gp_minimize(
                bo_wrapper,
                space,
                n_calls=config.budget,
                n_initial_points=10,
                random_state=seed,
            )
            raw_results["Bayesian Optimization"].append((bo_x, bo_y))
            times["Bayesian Optimization"].append(time.time() - start_time)
        except Exception as error:
            print(f"Bayesian Optimization failed: {error}")

    processed = process_multi_runs(raw_results, config.budget)

    print("\n=== WALL-CLOCK TIME ANALYSIS ===")
    for algorithm_name, values in times.items():
        if values:
            print(f"{algorithm_name}: average {np.mean(values):.2f}s (std {np.std(values):.2f})")

    plot_processed_results(processed, n_seeds=len(config.seeds))
    generate_pgfplots_code(processed)

    return processed, times, raw_results


def run_quick_benchmark(
    settings: Optional[HPOSettings] = None,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, List[float]], Dict[str, List[Tuple[List[int], List[float]]]]]:
    config = settings or HPOSettings(
        seeds=(42,),
        budget=20,
        epochs=1,
        train_size=1000,
        validation_size=250,
        train_batch_size=64,
        validation_batch_size=250,
    )
    return run_robust_benchmark(config)


def main() -> None:
    run_robust_benchmark()


__all__ = [
    "FashionMNISTObjective",
    "HPOSettings",
    "SimpleCNN",
    "generate_pgfplots_code",
    "interpolate_run",
    "main",
    "plot_processed_results",
    "process_multi_runs",
    "run_quick_benchmark",
    "run_robust_benchmark",
]
