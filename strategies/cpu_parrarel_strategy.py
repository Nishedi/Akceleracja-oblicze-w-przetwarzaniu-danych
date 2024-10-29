from strategies.base_strategy import PrimalityTestStrategy
from random import random
from multiprocessing import Process, Manager
import psutil
import math

class CPUParrarelPrimalityStrategyStrategy(PrimalityTestStrategy):
    def is_prime(self, n: int, k: int) -> bool:
        testValue =n
        repetitions = k
        """Returns True if testValue is probably prime, False if it's definitely composite."""
        manager = Manager()
        valueIsNotPrime = manager.Value('b', False)

        if testValue < 3:
            return False

        # Find max power of 2 that divides testValue - 1
        powerOfTwo = 0
        testValueMinusOne = testValue - 1
        while testValueMinusOne % 2 == 0:
            testValueMinusOne //= 2
            powerOfTwo += 1

        # Get number of cores
        coreCount = psutil.cpu_count(logical=False)

        # Create a list of processes
        processes = []

        # Create a process for each core
        for i in range(coreCount):
            processes.append(Process(target=self.cpu_check_number, args=(
                i, coreCount, testValue, powerOfTwo, repetitions, valueIsNotPrime)))

        # Start all processes
        for process in processes:
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        # Check if any of the processes found a composite number
        if valueIsNotPrime.value:
            return False

        # If we got this far, testValue is probably prime
        return True

    def cpu_check_number(self, coreCount, testValue, powerOfTwo, allRepetitions, valueIsNotPrime):
        """Returns True(0) if testValue is probably prime, False(1) if it's definitely composite."""

        # Calculate the number of repetitions for each process
        coreRepetitions = math.ceil(allRepetitions / coreCount)

        for _ in range(1, coreRepetitions):
            # Choose a random number between 2 and testValue - 2
            randomValue = int(random() * (testValue - 3)) + 2

            # Calculate randomValue^testValueMinusOne mod testValue
            x = pow(randomValue, testValue - 1, testValue)

            # If x is 1 or testValue - 1, we don't know if testValue is prime, so we'll try again
            if x == 1 or x == testValue - 1:
                continue

            # Otherwise, we'll keep squaring x and checking if it's equal to testValue - 1
            for _ in range(1, powerOfTwo):
                if x == testValue - 1:
                    break

                x = pow(x, 2, testValue)
                if x == 1:
                    valueIsNotPrime.value = True
                    return

            # If x != testValue - 1, then we know that testValue is definitely composite
            if x != testValue - 1:
                valueIsNotPrime.value = True
                return

        # If we got this far, testValue is probably prime
        valueIsNotPrime.value = False
        return
