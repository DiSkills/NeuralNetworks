import random
import math

class Neuron:
    def __init__(self, size: int):
        self.weights = [random.uniform(-1, 1) for _ in range(size)]
        self.bias = random.uniform(-1, 1)
        self.output = 0
        self.error = 0

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        return x * (1 - x)

    def __call__(self, inputs: list[float]) -> float:
        v = sum(i * w for i, w in zip(inputs, self.weights)) + self.bias
        self.output = self.sigmoid(v)
        return self.output

    def adjust(self, inputs: list[float], learning_rate: float):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.error * self.sigmoid_derivative(self.output) * inputs[i]
        self.bias += learning_rate * self.error * self.sigmoid_derivative(self.output)


class Network:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_size)]
        self.output_layer = [Neuron(hidden_size) for _ in range(output_size)]

    def __call__(self, inputs: list[float]) -> tuple[list[float], list[float]]:
        hidden_outputs = [neuron(inputs) for neuron in self.hidden_layer]
        output = [neuron(hidden_outputs) for neuron in self.output_layer]
        return output, hidden_outputs

    def backpropagate(
        self,
        inputs: list[float],
        expected: list[float],
        hidden_outputs: list[float],
        learning_rate: float,
    ):
        for desired, output in zip(expected, self.output_layer):
            output.error = desired - output.output
        for i, hidden in enumerate(self.hidden_layer):
            hidden.error = sum(output.error * output.weights[i] for output in self.output_layer)
        for neuron in self.output_layer:
            neuron.adjust(hidden_outputs, learning_rate)
        for neuron in self.hidden_layer:
            neuron.adjust(inputs, learning_rate)

    def train(
        self,
        training_data: list[tuple[list[float], list[float]]],
        epochs: int,
        learning_rate: float,
    ):
        for epoch in range(epochs):
            error = 0
            for inputs, expected_output in training_data:
                output, hidden_outputs = self.__call__(inputs)
                self.backpropagate(inputs, expected_output, hidden_outputs, learning_rate)
                error += sum((expected_output[i] - output[i]) ** 2 for i in range(len(self.output_layer)))

    def predict(self, inputs: list[float]) -> list[float]:
        output = self.__call__(inputs)[0]
        result = [0] * len(output)
        result[output.index(max(output))] = 1
        return result


def main():
    training_data = [
        ([1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 0, 1]),  # X -> 0001
        ([0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0]),  # I -> 0010
        ([1, 0, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 0]),  # Y -> 0100
        ([1, 0, 0, 1, 0, 0, 1, 1, 1], [1, 0, 0, 0]),  # L -> 1000
    ]
    network = Network(9, 5, 4)
    network.train(training_data, epochs=1000, learning_rate=0.1)
    test_input_with_noise = [1, 0, 1, 0, 1, 1, 1, 0, 1]  # X
    print(network.predict(test_input_with_noise))


if __name__ == "__main__":
    main()
