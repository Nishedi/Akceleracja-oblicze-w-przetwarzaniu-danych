from strategies.base_strategy import PrimalityTestStrategy
import random
from multiprocessing import Process, Manager
import psutil
import math

class CPUParrarelPrimalityStrategyStrategy(PrimalityTestStrategy):
    def is_prime(self, n: int, k: int) -> bool:
        manager = Manager()
        valueIsNotPrime = manager.Value('b', False)
        # Wyznaczenie liczby rdzeni
        coreCount = psutil.cpu_count(logical=False)
        # Przygotowanie listy procesów do uruchomienia na każdym rdzeniu
        processes = []
        for i in range(coreCount):
            processes.append(Process(target=self.cpu_check_number, args=(coreCount, n, k, valueIsNotPrime)))

        #Uruchom wszystkie procesy i poczekaj na ich zakończenie
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        # Check if any of the processes found a composite number
        if valueIsNotPrime.value:
            return False

        # If we got this far, testValue is probably prime
        return True

    def cpu_check_number(self, coreCount, testValue, allRepetitions, valueIsNotPrime):
        d = testValue - 1
        coreRepetitions = math.ceil(allRepetitions / coreCount)
        for _ in range(coreRepetitions):
            if not self._miller_test(d, testValue):
                valueIsNotPrime.value = True
                return False
        valueIsNotPrime.value = False
        return True

    def _miller_test(self, d, n):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        while d != n - 1:
            x = (x * x) % n
            d *= 2
            if x == 1:
                return False
            if x == n - 1:
                return True
        return False