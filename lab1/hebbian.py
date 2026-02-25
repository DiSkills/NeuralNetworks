from lab1.wta import Neuron


class HebbianNeuron(Neuron):
    @classmethod
    def create(cls, size: int) -> "Neuron":
        return super().create(size + 1)

    def __call__(self, input: list[float]) -> float:
        return super().__call__([1] + input)

    def train(self, input: list[float], eta: float):
        output = self.__call__(input)
        self.weights = [w + eta * output * i for w, i in zip(self.weights, [1] + input)]
