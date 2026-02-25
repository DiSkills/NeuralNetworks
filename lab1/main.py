from lab1.hebbian import HebbianNeuron
from lab1.penalty import PenaltyNeuron
from lab1.wta import Neuron


class Network:
    def __init__(self, neuron: type[Neuron], input_size: int, number_of_neurons: int):
        self.neurons = [neuron.create(input_size) for _ in range(number_of_neurons)]

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


def task(neuron: type[Neuron]):
    network = Network(neuron, 2, 4)
    training_set = [
        [0.97, 0.2], [1.0, 0.0], [-0.72, 0.7], [-0.67, 0.74],
        [-0.8, 0.6], [0.0, -1.0], [0.2, -0.97], [-0.3, -0.95],
    ]
    network.train(training_set, 0.5)
    for i, neuron in enumerate(network.neurons):
        print(i, neuron.weights)


def main():
    for neuron in Neuron, PenaltyNeuron, HebbianNeuron:
        print('=' * 100)
        task(neuron)


if __name__ == "__main__":
    main()
