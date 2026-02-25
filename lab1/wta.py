from random import random


def normalize(arr: list[float]) -> list[float]:
    norm = sum(x ** 2 for x in arr) ** 0.5
    return [x / norm for x in arr]


class Neuron:
    def __init__(self, weights: list[float]):
        self.weights = weights

    @classmethod
    def create(cls, size: int) -> "Neuron":
        return cls(normalize([random() for _ in range(size)]))

    def __call__(self, input: list[float]) -> float:
        return sum(w * i for w, i in zip(self.weights, input))

    def train(self, input: list[float], eta: float):
        self.weights = [w + eta * (i - w) for w, i in zip(self.weights, input)]
