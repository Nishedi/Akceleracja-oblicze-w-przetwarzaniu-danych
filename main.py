from miller_rabin import MillerRabinTest
from strategies.cpu_strategy import CPUPrimalityTestStrategy
from strategies.cpu_parrarel_strategy import CPUParrarelPrimalityStrategyStrategy
from utils import load_numbers_from_file, stopwatch


def main():
    strategy_mapping = {
        '1': CPUPrimalityTestStrategy,
        '2': CPUParrarelPrimalityStrategyStrategy
    }

    while True:
        print("Choose strategy:")
        print("1. CPU")
        print("2. Parrael CPU")

        choice = input("Enter the number of your choice: ")
        print(choice)

        # Get the strategy based on the user's choice
        strategy = strategy_mapping.get(choice, CPUPrimalityTestStrategy)()

        if choice not in strategy_mapping:
            print("Invalid choice, defaulting to CPU strategy.")

        miller_rabin_test = MillerRabinTest(strategy)

        print("\nTest numbers:")
        print("1. Enter number manually")
        print("2. Load numbers from a file")
        print("3. Load default value for test purposes (4754597)")
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
        elif input_mode == '3':
            n = 4754597
            k = 100000000
            is_prime = stopwatch(miller_rabin_test.is_prime)(n, k)
            print(f"Number {n} is prime: {is_prime}")
        else:
            print("Invalid input mode selected.")


if __name__ == '__main__':
    main()
