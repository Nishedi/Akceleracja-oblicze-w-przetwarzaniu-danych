from miller_rabin import MillerRabinTest
from strategies.cpu_strategy import CPUPrimalityTestStrategy
from strategies.cpu_parrarel_strategy import CPUParrarelPrimalityStrategyStrategy
from utils import load_numbers_from_file, stopwatch


def main():
    print("Choose strategy:")
    print("1. CPU")

    choice = input("Enter the number of your choice: ")

    if choice == '1':
        strategy = CPUPrimalityTestStrategy()
    if choice == '2':
        strategy = CPUParrarelPrimalityStrategyStrategy()
    else:
        print("Invalid choice, defaulting to CPU strategy.")
        strategy = CPUPrimalityTestStrategy()

    miller_rabin_test = MillerRabinTest(strategy)

    print("\nTest numbers:")
    print("1. Enter number manually")
    print("2. Load numbers from a file")
    input_mode = input("Enter the number of your choice: ")

    if input_mode == '1':
        n = int(input("Enter the number to check: "))
        k = int(input("Enter the number of iterations (default 10): ") or 10)
        is_prime = stopwatch(miller_rabin_test.is_prime)(n, k)
        print(f"Number {n} is prime: {is_prime}")
    elif input_mode == '2':
        file_path = input("Enter the file path: ") or 'numbers.txt'
        try:
            numbers = load_numbers_from_file(file_path)
            k = int(input("Enter the number of iterations (default 10): ") or 10)
            for number in numbers:
                is_prime = stopwatch(miller_rabin_test.is_prime)(number, k)
                print(f"Number {number} is prime: {is_prime}")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    else:
        print("Invalid input mode selected.")


if __name__ == '__main__':
    main()
