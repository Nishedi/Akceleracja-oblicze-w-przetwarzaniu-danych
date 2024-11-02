import time
from typing import List, Dict, Tuple
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from miller_rabin import MillerRabinTest
from strategies.cpu_parallel_strategy import CPUParallelPrimalityTestStrategy
from strategies.cpu_strategy import CPUPrimalityTestStrategy


class Benchmark:
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cpu_strategy = CPUPrimalityTestStrategy()
        self.parallel_strategy = CPUParallelPrimalityTestStrategy()

        self.cpu_test = MillerRabinTest(self.cpu_strategy)
        self.parallel_test = MillerRabinTest(self.parallel_strategy)

    def generate_range_data(self,
                            num_ranges: List[Tuple[int, int]],
                            numbers_per_range: int,
                            seed: int = 42) -> Dict[str, List[int]]:
        random.seed(seed)
        test_data = {}

        for min_val, max_val in num_ranges:
            range_id = f"{min_val}-{max_val}"
            numbers = [random.randint(min_val, max_val) for _ in range(numbers_per_range)]
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
            "input_sizes": {}
        }

        print()
        for digits in range(1, max_digits + 1, step):
            min_num = 10 ** (digits - 1)
            max_num = 10 ** digits - 1

            test_numbers = sorted([random.randint(min_num, max_num)
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
            start_time = time.time()
            self.cpu_test.is_prime(num, iterations)
            cpu_times.append(time.time() - start_time)
        results["cpu"][index] = cpu_times

        parallel_times = []
        for num in test_numbers:
            start_time = time.time()
            self.parallel_test.is_prime(num, iterations)
            parallel_times.append(time.time() - start_time)
        results["parallel"][index] = parallel_times

        return cpu_times, parallel_times

    def run_benchmarks(self,
                       test_data: Dict[str, List[int]],
                       iterations: int = 10) -> Dict[str, Dict[str, List[float]]]:
        results = {
            "cpu": {},
            "parallel": {}
        }

        for range_id, numbers in test_data.items():
            print(f"\nBenchmarking range {range_id}")

            cpu_times, parallel_times = self.measure_time(numbers, iterations, results, range_id)

            print(f"Average CPU time: {np.mean(cpu_times):.4f}s")
            print(f"Average Parallel time: {np.mean(parallel_times):.4f}s")

        with open(self.output_dir / "bar_results.json", "w") as f:
            json.dump(results, f)

        return results

    def generate_bar_chart(self, results: Dict[str, Dict[str, List[float]]]):
        plt.figure(figsize=(12, 6))
        ranges = list(results["cpu"].keys())
        cpu_data = [results["cpu"][range_id] for range_id in ranges]
        parallel_data = [results["parallel"][range_id] for range_id in ranges]

        cpu_means = [np.mean(times) for times in cpu_data]
        parallel_means = [np.mean(times) for times in parallel_data]

        x = np.arange(len(ranges))
        width = 0.35

        plt.bar(x - width / 2, cpu_means, width, label='CPU Strategy')
        plt.bar(x + width / 2, parallel_means, width, label='Parallel Strategy')

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
        cpu_stds = [np.std(results["cpu"][d]) for d in digits]
        parallel_stds = [np.std(results["parallel"][d]) for d in digits]

        input_sizes = [np.mean(results["input_sizes"][d]) for d in digits]

        plt.figure(figsize=(12, 6))
        plt.errorbar(input_sizes, cpu_means, yerr=cpu_stds, label='CPU Strategy', marker='o')
        plt.errorbar(input_sizes, parallel_means, yerr=parallel_stds, label='Parallel Strategy', marker='s')

        plt.xlabel('Input Size (Number)')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time vs Input Size (Linear Scale)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "line_chart.png")
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

    range_data = benchmark.generate_range_data(num_ranges=ranges, numbers_per_range=10)
    range_results = benchmark.run_benchmarks(test_data=range_data, iterations=10)
    benchmark.generate_bar_chart(range_results)

    results = benchmark.generate_increasing_data(max_digits=12, step=2, samples_per_step=3, iterations=10)
    benchmark.generate_line_chart(results)


if __name__ == "__main__":
    main()
