from random import random
import numpy as np


# we are after the maximum of this f(x) function
def f(x):
    return (x - 0.3) * (x - 0.3) * (x - 0.3) - 5 * x + x * x - 2


class SimulatedAnnealing:

    # we solve the problem withing the interval [min,max]
    # while T is high: we accept "bad" moves as well
    # if T is low: we keep accepting "good" moves exclusively
    def __init__(self, min_coordinate, max_coordinate, min_temp, max_temp, cooling_rate=0.02):
        self.min_coordinate = min_coordinate
        self.max_coordinate = max_coordinate
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.cooling_rate = cooling_rate
        # this is the x coordinate of the actual state
        self.actual_state = 0
        # this is the neighboring state
        self.next_state = 0
        self.best_state = 0

    def run(self):
        temp = self.max_temp

        while temp > self.min_temp:
            new_state = self.generate_next_state()

            # because we are looking for the maximum of f(x) this is why
            # the higher the value of the function at x the higher the energy
            actual_energy = self.get_energy(self.actual_state)
            new_energy = self.get_energy(new_state)

            # we are accepting better states with 100% probability
            if random() < self.accept_prob(actual_energy, new_energy, temp):
                self.actual_state = new_state

            if f(self.actual_state) > f(self.best_state):
                self.best_state = self.actual_state

            # decrement the T temperature
            temp = temp * (1 - self.cooling_rate)

        print('Global maximum: x=%s f(x)=%s' % (self.best_state, f(self.best_state)))

    # random x coordinate within the range [min,max]
    def generate_next_state(self):
        return self.min_coordinate + (self.max_coordinate - self.min_coordinate) * random()

    @staticmethod
    def accept_prob(actual_energy, next_energy, temp):
        # the next state is better (maximization problem) we accept
        if next_energy > actual_energy:
            return 1

        # we accept "bad" moves with a given probability
        return np.exp((actual_energy - next_energy) / temp)

    @staticmethod
    def get_energy(x):
        return f(x)


if __name__ == '__main__':
    algorithm = SimulatedAnnealing(-2, 2, 1, 100)
    algorithm.run()
