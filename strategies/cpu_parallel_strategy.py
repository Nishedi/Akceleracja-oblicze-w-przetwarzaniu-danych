from strategies.base_strategy import PrimalityTestStrategy
from multiprocessing import Process, Manager
import numpy as np
import psutil
import math


class CPUParallelPrimalityTestStrategy(PrimalityTestStrategy):
    def is_prime(self, n: int, k: int, bases: np.ndarray) -> bool:
        if n == 2 or n == 3:
            return True
        if n <= 1 or n % 2 == 0:
            return False

        manager = Manager()
        value_is_not_prime = manager.Value('b', False)
        # Wyznaczenie liczby rdzeni
        number_of_cores = psutil.cpu_count(logical=False)
        repetitions_per_core = math.ceil(k / number_of_cores)
        # Przygotowanie listy procesów do uruchomienia na każdym rdzeniu
        processes = []
        for i in range(number_of_cores):
            start_idx = i * repetitions_per_core
            end_idx = min(start_idx + repetitions_per_core, len(bases))
            core_bases = bases[start_idx:end_idx]
            processes.append(Process(target=self.single_process, args=(core_bases, n, value_is_not_prime)))

        # Uruchom wszystkie procesy i poczekaj na ich zakończenie
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # Sprawdź, czy któryś z procesów znalazł liczbę złożoną
        if value_is_not_prime.value:
            return False

        return True

    @staticmethod
    def single_process(bases: np.ndarray, n: int, value_is_not_prime: bool) -> bool:
        d = n - 1
        while d % 2 == 0:
            d //= 2

        for base in bases:
            x = pow(int(base), d, n)
            if x == 1 or x == n - 1:
                continue

            while d != n - 1:
                x = (x * x) % n
                d *= 2
                if x == 1:
                    value_is_not_prime.value = True
                    return False
                if x == n - 1:
                    break
            else:
                value_is_not_prime.value = True
                return False

        value_is_not_prime.value = False
        return True
