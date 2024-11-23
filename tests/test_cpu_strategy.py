import unittest
from miller_rabin import MillerRabinTest
from random_base_generator import RandomBaseGenerator
from strategies.cpu_strategy import CPUPrimalityTestStrategy
from utils import load_numbers_from_file


class TestCPUPrimalityTest(unittest.TestCase):
    def setUp(self):
        self.cpu_strategy = CPUPrimalityTestStrategy()
        self.miller_rabin_test = MillerRabinTest(self.cpu_strategy)

    def test_prime(self):
        k = 100
        random_bases = RandomBaseGenerator.generate_bases(31, k)
        self.assertTrue(self.miller_rabin_test.is_prime(31, k, random_bases))
        random_bases = RandomBaseGenerator.generate_bases(35, k)
        self.assertFalse(self.miller_rabin_test.is_prime(35, k, random_bases))

    def test_non_primes(self):
        self._test_from_file("data/non_primes.csv", False)

    def test_primes_20_bits(self):
        self._test_from_file("data/primes_20_bits.csv", True)

    def test_primes_43_bits(self):
        self._test_from_file("data/primes_43_bits.csv", True)

    def test_primes_63_bits(self):
        self._test_from_file("data/primes_63_bits.csv", True)

    def test_edge_cases_non_primes(self):
        self._test_from_file("data/edge_cases_false.csv", False)

    def test_edge_cases_primes(self):
        self._test_from_file("data/edge_cases_true.csv", True)

    def test_pseudo_primes(self):
        self._test_from_file("data/pseudo_primes.csv", False)

    def _test_from_file(self, file_path, expected: bool):
        numbers = load_numbers_from_file(file_path)
        k = 100
        for number in numbers:
            random_bases = RandomBaseGenerator.generate_bases(number, k)
            result = self.miller_rabin_test.is_prime(number, k, random_bases)
            self.assertEqual(result, expected, f"Failed on {number} (expected {expected})")


if __name__ == '__main__':
    unittest.main()
