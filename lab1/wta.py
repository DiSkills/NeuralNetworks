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
        return sum(self.weights[i] * input[i] for i in range(len(input)))
