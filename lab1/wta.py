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

    def train(self, input: list[float], eta: float):
        self.weights = [self.weights[i] + eta * (input[i] - self.weights[i]) for i in range(len(input))]


class Network:
    def __init__(self, input_size: int, number_of_neurons: int):
        self.neurons = [Neuron.create(input_size) for _ in range(number_of_neurons)]

    def __call__(self, input: list[float]) -> list[float]:
        u = [neuron(input) for neuron in self.neurons]
        y = [0] * len(u)
        y[u.index(max(u))] = 1
        return y

    def train(self, inputs: list[list[float]], eta: float):
        for input in inputs:
            u = [neuron(input) for neuron in self.neurons]
            winner = self.neurons[u.index(max(u))]
            winner.train(input, eta)


def main():
    network = Network(2, 4)
    training_set = [
        [0.97, 0.2], [1.0, 0.0], [-0.72, 0.7], [-0.67, 0.74],
        [-0.8, 0.6], [0.0, -1.0], [0.2, -0.97], [-0.3, -0.95],
    ]
    network.train(training_set, 0.5)
    for i, neuron in enumerate(network.neurons):
        print(i, neuron.weights)


if __name__ == "__main__":
    main()
