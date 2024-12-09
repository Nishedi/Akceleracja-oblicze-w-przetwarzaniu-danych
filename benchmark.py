import time
from typing import List, Dict, Tuple
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.ticker import ScalarFormatter

from miller_rabin import MillerRabinTest
from random_base_generator import RandomBaseGenerator
from strategies.cpu_parallel_strategy import CPUParallelPrimalityTestStrategy
from strategies.cpu_strategy import CPUPrimalityTestStrategy
from strategies.gpu_strategy import GPUPrimalityTestStrategy


class Benchmark:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cpu_strategy = CPUPrimalityTestStrategy()
        self.parallel_strategy = CPUParallelPrimalityTestStrategy()
        self.gpu_strategy = GPUPrimalityTestStrategy()

        self.cpu_test = MillerRabinTest(self.cpu_strategy)
        self.parallel_test = MillerRabinTest(self.parallel_strategy)
        self.gpu_test =MillerRabinTest(self.gpu_strategy)

    def generate_range_data(self,
                            num_ranges: List[Tuple[int, int]],
                            numbers_per_range: int,
                            seed: int = 42) -> Dict[str, List[int]]:
        random.seed(seed)
        test_data = {}

        for min_val, max_val in num_ranges:
            range_id = f"{min_val}-{max_val}"
            numbers = [sympy.randprime(min_val, max_val) for _ in range(numbers_per_range)]
            test_data[range_id] = numbers

        with open(self.output_dir / "test_data.json", "w") as f:
            json.dump(test_data, f)

        return test_data

    def generate_increasing_data(self,
                                 max_digits: int = 12,
                                 step: int = 1,
                                 samples_per_step: int = 3,
                                 iterations: int = 10) -> Dict[str, Dict[int, List[float]]]:
        results = {
            "cpu": {},
            "parallel": {},
            "input_sizes": {},
            "gpu_sizes": {}
        }

        print()
        for digits in range(1, max_digits + 1, step):
            min_num = 10 ** (digits - 1)
            max_num = 10 ** digits - 1

            test_numbers = sorted([sympy.randprime(min_num, max_num)
                                   for _ in range(samples_per_step)])

            results["input_sizes"][digits] = test_numbers

            self.measure_time(test_numbers, iterations, results, digits)

            print(f"Completed testing {digits}-digit numbers")

        with open(self.output_dir / "line_results.json", "w") as f:
            json.dump(results, f)

        return results

    def measure_time(self, test_numbers, iterations, results, index):
        cpu_times = []
        for num in test_numbers:
            random_bases = RandomBaseGenerator.generate_bases(num, iterations)
            start_time = time.time()
            self.cpu_test.is_prime(num, iterations, random_bases)
            cpu_times.append(time.time() - start_time)
            print("cpu", time.time() - start_time)
        results["cpu"][index] = cpu_times

        parallel_times = []
        for num in test_numbers:
            random_bases = RandomBaseGenerator.generate_bases(num, iterations)
            start_time = time.time()
            self.parallel_test.is_prime(num, iterations, random_bases)
            parallel_times.append(time.time() - start_time)
            print("parallel", time.time() - start_time)
        results["parallel"][index] = parallel_times

        gpu_times = []
        for num in test_numbers:
            random_bases = RandomBaseGenerator.generate_bases(num, iterations)
            start_time = time.time()
            self.gpu_test.is_prime(num, iterations, random_bases)
            gpu_times.append(time.time() - start_time)
            print("gpu", time.time() - start_time)
        results["gpu"][index] = gpu_times


        return cpu_times, parallel_times, gpu_times

    def run_benchmarks(self,
                       test_data: Dict[str, List[int]],
                       iterations: int = 10) -> Dict[str, Dict[str, List[float]]]:
        results = {
            "cpu": {},
            "parallel": {},
            "gpu": {}
        }

        for range_id, numbers in test_data.items():
            print(f"\nBenchmarking range {range_id}")

            cpu_times, parallel_times, gpu_times = self.measure_time(numbers, iterations, results, range_id)

            print(f"Average CPU time: {np.mean(cpu_times):.4f}s")
            print(f"Average Parallel time: {np.mean(parallel_times):.4f}s")
            print(f"Average GPI Parallel time: {np.mean(gpu_times):.4f}s")

        with open(self.output_dir / "bar_results.json", "w") as f:
            json.dump(results, f)

        return results

    def generate_bar_chart(self, results: Dict[str, Dict[str, List[float]]]):
        plt.figure(figsize=(12, 6))
        ranges = list(results["cpu"].keys())
        cpu_data = [results["cpu"][range_id] for range_id in ranges]
        parallel_data = [results["parallel"][range_id] for range_id in ranges]
        gpu_data = [results["gpu"][range_id] for range_id in ranges]

        cpu_means = [np.mean(times) for times in cpu_data]
        parallel_means = [np.mean(times) for times in parallel_data]
        gpu_means = [np.mean(times) for times in gpu_data]

        x = np.arange(len(ranges))
        width = 0.25

        plt.bar(x - width, cpu_means, width, label='CPU Strategy')
        plt.bar(x, parallel_means, width, label='Parallel Strategy')
        plt.bar(x + width, gpu_means, width, label='GPU Parallel Strategy')

        plt.xlabel('Number Ranges')
        plt.ylabel('Average Time (seconds)')
        plt.title('Average Execution Time by Strategy and Range')
        plt.xticks(x, ranges, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "bar_chart.png")
        plt.close()

    def generate_line_chart(self, results: Dict[str, Dict[int, List[float]]]):
        digits = sorted(list(results["cpu"].keys()))

        cpu_means = [np.mean(results["cpu"][d]) for d in digits]
        parallel_means = [np.mean(results["parallel"][d]) for d in digits]
        gpu_means = [np.mean(results["gpu"][d]) for d in digits]
        cpu_stds = [np.std(results["cpu"][d]) for d in digits]
        parallel_stds = [np.std(results["parallel"][d]) for d in digits]
        gpu_stds = [np.std(results["gpu"][d]) for d in digits]

        input_sizes = [np.mean(results["input_sizes"][d]) for d in digits]

        plt.figure(figsize=(12, 6))
        plt.errorbar(input_sizes, cpu_means, yerr=cpu_stds, label='CPU Strategy', marker='o')
        plt.errorbar(input_sizes, parallel_means, yerr=parallel_stds, label='Parallel Strategy', marker='s')
        plt.errorbar(input_sizes, gpu_means, yerr=gpu_stds, label='GPU Parallel Strategy', marker='s')

        plt.xscale("log")

        plt.xlabel('Input Size (Number)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time vs Input Size (Linear Scale)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "line_chart.png")
        plt.close()

    def generate_iterations_test(self,
                                  test_range: Tuple[int, int],
                                  start_iterations: int = 1000,
                                  end_iterations: int = 10000000,
                                  samples_per_step: int = 3) -> Dict[str, List[float]]:
        min_val, max_val = test_range
        numbers = [sympy.randprime(min_val, max_val) for _ in range(samples_per_step)]


        results = {
            "iterations": [],
            "cpu_times": [],
            "parallel_times": [],
            "gpu_times": []
        }
        print("\nStarting iteration test...")

        iterations = start_iterations
        while iterations <= end_iterations:
            print(f"Testing {iterations} iterations...")

            cpu_times, parallel_times, gpu_times = self.measure_time_for_iterations(numbers, iterations)

            results["iterations"].append(iterations)
            results["cpu_times"].append(np.mean(cpu_times))
            results["parallel_times"].append(np.mean(parallel_times))
            results["gpu_times"].append(np.mean(gpu_times))

            iterations *= 10  # Zwiększamy liczbę iteracji o jedno zero

        with open(self.output_dir / "iterations_results.json", "w") as f:
            json.dump(results, f)

        return results

    def measure_time_for_iterations(self, test_numbers, iterations):
        cpu_times = []
        for num in test_numbers:
            random_bases = RandomBaseGenerator.generate_bases(num, iterations)
            start_time = time.time()
            self.cpu_test.is_prime(num, iterations, random_bases)
            cpu_times.append(time.time() - start_time)

        parallel_times = []
        for num in test_numbers:
            random_bases = RandomBaseGenerator.generate_bases(num, iterations)
            start_time = time.time()
            self.parallel_test.is_prime(num, iterations, random_bases)
            parallel_times.append(time.time() - start_time)

        gpu_times = []
        for num in test_numbers:
            random_bases = RandomBaseGenerator.generate_bases(num, iterations)
            start_time = time.time()
            self.gpu_test.is_prime(num, iterations, random_bases)
            gpu_times.append(time.time() - start_time)

        return cpu_times, parallel_times, gpu_times

    def generate_iterations_chart(self, results: Dict[str, List[float]]):
        iterations = results["iterations"]
        cpu_times = results["cpu_times"]
        parallel_times = results["parallel_times"]
        gpu_times = results["gpu_times"]

        plt.figure(figsize=(12, 6))

        plt.plot(iterations, cpu_times, label="CPU Strategy", marker='o')
        plt.plot(iterations, parallel_times, label="Parallel Strategy", marker='s')
        plt.plot(iterations, gpu_times, label="GPU Parallel Strategy", marker='s')

        plt.xscale("log")

        plt.xlabel('Number of Iterations')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title('Execution Time vs Number of Iterations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "iterations_chart.png")
        plt.close()


def main():
    benchmark = Benchmark()

    ranges = [
        (1000, 10000),
        (10000, 100000),
        (100000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000)
    ]

    range_data = benchmark.generate_range_data(num_ranges=ranges, numbers_per_range=2)
    range_results = benchmark.run_benchmarks(test_data=range_data, iterations=10000000)
    benchmark.generate_bar_chart(range_results)

    results = benchmark.generate_increasing_data(max_digits=12, step=2, samples_per_step=3, iterations=10000000)
    benchmark.generate_line_chart(results)

    iteration_results = benchmark.generate_iterations_test(test_range=ranges[2], start_iterations=1000, end_iterations=100000000, samples_per_step=3)
    benchmark.generate_iterations_chart(iteration_results)


if __name__ == "__main__":
    main()
