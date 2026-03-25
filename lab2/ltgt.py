from random import random


def generate(n: int) -> list[tuple[float, float, int]]:
    data = []
    for _ in range(n):
        x1, x2 = random(), random()
        label = int(x1 > x2)
        data.append((x1, x2, label))
    return data


class Neuron:
    def __init__(self, weights: list[float]):
        self.weights = weights

    def __call__(self, input: list[float]) -> float:
        return int(sum(w * i for w, i in zip(self.weights, [1] + input)) > 0)

    def train(self, input: list[float], eta: float, desired_response: float):
        error = desired_response - self.__call__(input)
        self.weights = [w + eta * error * i for w, i in zip(self.weights, [1] + input)]


class AdalineNeuron(Neuron):
    def output(self, input: list[float]) -> float:
        return sum(w * i for w, i in zip(self.weights, [1] + input))

    def __call__(self, input: list[float]) -> float:
        return int(self.output(input) > 0)

    def train(self, input: list[float], eta: float, desired_response: float):
        error = desired_response - self.output(input)
        self.weights = [w + eta * error * i for w, i in zip(self.weights, [1] + input)]


def main():
    weights = [0] * 3
    neuron, adaline = Neuron(weights.copy()), AdalineNeuron(weights.copy())
    for *input, label in generate(20):
        neuron.train(input, 0.5, label)
        adaline.train(input, 0.5, label)
    corrected_adaline = corrected = 0
    for *input, label in generate(1000):
        corrected += (neuron(input) == label)
        corrected_adaline += (adaline(input) == label)
    print(f"Neuron: {corrected / 1000 * 100}%, Adaline: {corrected_adaline / 1000 * 100}%")


if __name__ == "__main__":
    main()
