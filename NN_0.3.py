# Work in progress library for Neural Networks
# Currently supports simple feedforward networks of arbitrary depth with ReLu activation function
# TODO: Convolutional and Recurring Layers; sigmoid, tanh and leaky ReLu activation functions
from random import randint
from NN_supplies import *
import time
import matplotlib.pyplot as plt


class Network:
    def __init__(self, layout):
        self.architecture = []
        self.global_error = 1
        self.learning_rate_acceleration = 0.0001

        # Creating the Neurons and filling the self.architecture matrix
        for i in layout:
            temp = []
            for j in range(i):
                temp.append(Neuron())
            self.architecture.append(temp)

    def output(self):
        temp = []
        for i in self.architecture[-1]:
            temp.append(i.output)
        return temp

    def forward_pass(self, inputs):
        for i, j in enumerate(self.architecture[0]):
            j.output = inputs[i]
        for i in self.architecture[1:]:
            for j in i:
                j.compute()
        temp = []
        for i in self.architecture[-1]:
            temp.append(i.output)
        return temp

    def backwards_pass(self, outputs):
        # Sanity Check
        if len(outputs) != len(self.architecture[-1]):
            raise Exception("The number of desired outputs isn't equal to the number of outputs of the network")

        # Calibrating the last layer
        for i, neuron in enumerate(self.architecture[-1]):
            neuron.calibrate(outputs[i])
            self.global_error += neuron.error

        # Calibrating all the hidden layers
        for layer in self.architecture[-2:0:-1]:
            for neuron in layer:
                neuron.calibrate()
                self.global_error += neuron.error

        # Calibrating the learning rate
        for layer in self.architecture:
            for neuron in layer:
                neuron.learning_rate *= 1 + self.learning_rate_acceleration / self.global_error
        # print(self.learning_rate_acceleration / self.global_error)
        # print(self.architecture[2][0].learning_rate)
        self.global_error = 1

    def standard_layer(self, layer):
        for neuron in self.architecture[layer]:
            if layer != 0:
                temp = []
                for j in self.architecture[layer - 1]:
                    temp.append([j, randint(0, 1000) / 1000])
                neuron.inputs = temp
            if layer != len(self.architecture) - 1:
                temp = []
                for j in self.architecture[layer + 1]:
                    temp.append(j)
                neuron.outputs = temp


class Neuron:
    def __init__(self, activ_func="ReLu", output_bounds=None, weight_bounds=None):
        self.learning_rate = 0.0001
        self.activ_func = activ_func
        self.output = 0
        self.error = 0
        self.inputs = []
        self.outputs = []

        if output_bounds is None:
            self.output_bounds = [-1000, 1000]
        else:
            self.output_bounds = output_bounds

        if weight_bounds is None:
            self.weight_bounds = [-10, 10]
        else:
            self.weight_bounds = weight_bounds

    def compute(self):
        temp_sum = 0
        for i in self.inputs:
            temp_sum += i[0].output * i[1]
        temp_sum = 0.1 * temp_sum / len(self.inputs)
        # Accessing the activation function dictionary from the NN_supplies.py
        output = Activation_Functions[self.activ_func][0](temp_sum)
        self.output = min(max(output, self.output_bounds[0]), self.output_bounds[1])

    def calibrate(self, requirement=None):
        error = 0
        if requirement is None:
            for i in self.outputs:
                error += i.error * find_integer_value(i.inputs, self)
            error = error / len(self.outputs)
        else:
            error = self.output - requirement

        # Finding the error using the derivative of the activation function from the NN_supplies.py dictionary
        self.error = Activation_Functions[self.activ_func][1](self.output) * error
        for i in self.inputs:
            i[1] -= self.error * self.learning_rate * i[0].output
            i[1] = min(max(i[1], self.weight_bounds[0]), self.weight_bounds[1])

    def __repr__(self):
        return f"(N, {round(self.output, 1)}, {self.activ_func})"


def generate_data():
    question = [randint(0, 50), randint(0, 35)]
    if question[0] > question[1] ** 2:
        answer = [0, 50]
    else:
        answer = [50, 0]
    return question, answer


def accuracy_test(answer, output, accuracy_type):
    if accuracy_type == "MAX":
        return answer.index(max(answer)) == output.index(max(output))


def main():
    total_accuracy = []
    accuracy = []
    for _ in range(100):
        accuracy.append(False)

    network_1 = Network([2, 8, 5, 2])
    for i in range(len(network_1.architecture)):
        network_1.standard_layer(i)
    start_time = time.perf_counter()
    n = 50000
    for a_ in range(n):
        print(str(a_ * 100 / n) + "%")
        x, y = generate_data()
        # print(x)
        network_1.forward_pass(x)
        if a_ > 2500:
            network_1.backwards_pass(y)

        for i in range(len(accuracy) - 1):
            accuracy[i] = accuracy[i + 1]
        accuracy[len(accuracy) - 1] = accuracy_test(y, network_1.output(), "MAX")
        total_accuracy.append(accuracy.count(True))
        # print(network_1.architecture)

    print(network_1.architecture)
    print(time.perf_counter() - start_time)
    plt.plot(total_accuracy)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
