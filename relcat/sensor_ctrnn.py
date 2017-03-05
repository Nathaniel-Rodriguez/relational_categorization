import numpy as np

def sigmoid(x):

    return 1.0 / (1.0 + np.exp(-x))

class SensorCTRNN:
    """
    A class for continuous-time recurrent neural networks
    that uses sensors for input.

    Adapted from Randall Beer's C++ code

    """

    def __init__(self, circuit_size, num_of_sensors):
        """
        Initializes the CTRNN and its parameters to zero
        """

        self.circuit_size = circuit_size
        self.num_of_sensors = num_of_sensors

        self.ctrnn_states = np.zeros(self.circuit_size)
        self.ctrnn_outputs = np.zeros(self.circuit_size)
        self.biases = np.zeros(self.circuit_size)
        self.gains = np.zeros(self.circuit_size)
        self.taus = np.zeros(self.circuit_size)
        self.rtaus = np.zeros(self.circuit_size)
        self.sensor_states = np.zeros(self.circuit_size)
        self.circuit_weights = np.zeros((self.circuit_size, \
                                        self.circuit_size))
        self.sensor_weights = np.zeros((self.circuit_size, \
                                        self.circuit_size))

    def randomize_state(self, random_variable_lower_bound=-1.0, \
            random_variable_upper_bound=1.0):
        """
        Randomizes the state of the network given a set of lower and upper 
        bounds that the neuron states can take.
        """

        self.ctrnn_states = np.random.uniform(random_variable_lower_bound, 
                        random_variable_upper_bound, size=self.circuit_size)
        self.ctrnn_outputs = \
                    sigmoid(self.gains * self.ctrnn_states + self.biases)
        self.sensor_states = np.zeros(self.circuit_size)

    def set_neuron_time_constants(self, time_constants):

        self.taus = np.array(time_constants)
        self.rtaus = 1. / self.taus

    def euler_step(self, step_size):
        """
        Steps the network's states and outputs using the Euler method.
        This uses Beer's standard CTRNN equation.
        """

        inputs = np.dot(self.sensor_weights, self.sensor_states)
        self.ctrnn_states += step_size * self.rtaus * (inputs - self.ctrnn_states)
        self.ctrnn_outputs = sigmoid(self.gains * self.ctrnn_states + self.biases)

    def set_sensor(self, index, value):

        self.sensor_states[index] = value

    def neuron_output(self, index):

        return self.ctrnn_outputs[index]

if __name__ == '__main__':
    """
    For testing
    """

    pass