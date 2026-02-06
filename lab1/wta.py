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


class Network:
    def __init__(self, input_size: int, number_of_neurons: int):
        self.neurons = [Neuron.create(input_size) for _ in range(number_of_neurons)]

    def __call__(self, input: list[float]) -> list[float]:
        u = [neuron(input) for neuron in self.neurons]
        y = [0] * len(u)
        y[u.index(max(u))] = 1
        return y
