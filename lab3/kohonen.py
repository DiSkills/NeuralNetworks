import math
import random
from collections import defaultdict


def convert(raw: list[str | float]) -> list[float]:
    return [
        1 if raw[2] == "М" else 0,
        1 if raw[3] == "Да" else 0,
        *raw[4:9],
    ]


def linear_normalize(data: list[list[float]], num_features: int) -> list[list[float]]:
    mins = [min(row[j] for row in data) for j in range(num_features)]
    maxs = [max(row[j] for row in data) for j in range(num_features)]
    normalized = []
    for row in data:
        norm_row = []
        for j in range(num_features):
            denom = maxs[j] - mins[j]
            norm_row.append((row[j] - mins[j]) / denom if denom != 0 else 0.0)
        normalized.append(norm_row)
    return normalized


class Neuron:
    def __init__(self, weights: list[float]):
        self.weights = weights

    @classmethod
    def create(cls, input_size: int) -> "Neuron":
        return cls([random.random() for _ in range(input_size)])

    def distance(self, inputs: list[float]) -> float:
        return math.sqrt(sum((x - w) ** 2 for x, w in zip(inputs, self.weights)))

    def update(self, inputs: list[float], learning_rate: float):
        self.weights = [w + learning_rate * (x - w) for x, w in zip(inputs, self.weights)]


class Network:
    def __init__(self, input_size: int, num_clusters: int):
        self.neurons = [Neuron.create(input_size) for _ in range(num_clusters)]

    def winner(self, inputs: list[float]) -> int:
        distances = [n.distance(inputs) for n in self.neurons]
        return distances.index(min(distances))

    def train(self, data: list[list[float]], initial: float, step: float):
        data = data.copy()
        learning_rate = initial
        while learning_rate > 0:
            random.shuffle(data)
            for sample in data:
                idx = self.winner(sample)
                self.neurons[idx].update(sample, learning_rate)
            learning_rate -= step


def analyze(network: Network, raw_dataset: list[list[float | str]], dataset: list[list[float]]) -> None:
    summary = defaultdict(list)

    for raw, row in zip(raw_dataset, dataset):
        cluster = network.winner(row)
        summary[cluster].append(raw[-1])

    for cluster, scholarships in summary.items():
        print(f"Cluster {cluster + 1}: Average scholarship = {sum(scholarships) / len(scholarships):.2f}")


def main():
    network = Network(7, 4)

    raw_dataset = [
        [1, "Варданян", "М", "Да", 60, 79, 60, 72, 63, 1.00],
        [2, "Горбунов", "М", "Нет", 60, 61, 30, 5, 17, 0.00],
        [3, "Гуменюк", "Ж", "Нет", 60, 61, 30, 66, 58, 0.00],
        [4, "Егоров", "М", "Да", 85, 78, 72, 70, 85, 1.25],
        [5, "Захарова", "Ж", "Да", 65, 78, 60, 67, 65, 1.00],
        [6, "Иванова", "Ж", "Да", 60, 78, 77, 81, 60, 1.25],
        [7, "Ишонина", "Ж", "Да", 55, 79, 56, 69, 72, 0.00],
        [8, "Климчук", "М", "Нет", 55, 56, 50, 56, 60, 0.00],
        [9, "Лисовский", "М", "Нет", 55, 60, 21, 64, 50, 0.00],
        [10, "Нетреба", "М", "Нет", 60, 56, 30, 16, 17, 0.00],
        [11, "Остапова", "Ж", "Да", 85, 89, 85, 92, 85, 1.75],
        [12, "Пашкова", "Ж", "Да", 60, 88, 76, 66, 60, 1.25],
        [13, "Попов", "М", "Нет", 55, 64, 0, 9, 50, 0.00],
        [14, "Сазон", "Ж", "Да", 80, 83, 62, 72, 72, 1.25],
        [15, "Степоненко", "М", "Нет", 55, 10, 3, 8, 50, 0.00],
        [16, "Терентьева", "Ж", "Да", 60, 67, 57, 64, 50, 0.00],
        [17, "Титов", "М", "Да", 75, 98, 86, 82, 85, 1.50],
        [18, "Чернова", "Ж", "Да", 85, 85, 81, 85, 72, 1.25],
        [19, "Четкин", "М", "Да", 80, 56, 50, 69, 50, 0.00],
        [20, "Шевченко", "М", "Нет", 55, 60, 30, 8, 60, 0.00],
    ]
    dataset = linear_normalize([convert(row) for row in raw_dataset], 7)

    network.train(dataset, 0.3, 0.05)
    analyze(network, raw_dataset, dataset)


if __name__ == "__main__":
    main()
