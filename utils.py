from functools import wraps
import time


def load_numbers_from_file(file_path: str):
    with open(file_path, 'r') as file:
        numbers = [int(line.strip()) for line in file.readlines()]
    return numbers


def load_numbers_to_test(file_path: str):
    with open(file_path, 'r') as file:
        data = [line.strip().split() for line in file.readlines()]
        numbers = [(int(number), int(expected)) for number, expected in data]
    return numbers


def stopwatch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"\n{func.__name__} took {end - start} seconds")

        return result

    return wrapper
