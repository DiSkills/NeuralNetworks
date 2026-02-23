from lab1.wta import Neuron


class PenaltyNeuron(Neuron):
    def __init__(self, weights: list[float]):
        super().__init__(weights)
        self.number_of_victories = 0

    def __call__(self, input: list[float]) -> float:
        return super().__call__(input) - self.number_of_victories

    def train(self, input: list[float], eta: float):
        super().train(input, eta)
        self.number_of_victories += 1
