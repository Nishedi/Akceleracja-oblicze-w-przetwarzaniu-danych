class MillerRabinTest:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def is_prime(self, n: int, k: int = 10) -> bool:
        return self.strategy.is_prime(n, k)
