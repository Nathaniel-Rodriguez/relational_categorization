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

        self.ctrnn_states = np.zeros((self.circuit_size, 1))
        self.ctrnn_outputs = np.zeros((self.circuit_size, 1))
        self.biases = np.zeros((self.circuit_size, 1))
        self.gains = np.ones((self.circuit_size, 1))
        self.taus = np.ones((self.circuit_size, 1))
        self.rtaus = np.ones((self.circuit_size, 1))
        self.sensor_states = np.zeros((self.num_of_sensors, 1))
        self.circuit_weights = np.zeros((self.circuit_size, \
                                        self.circuit_size))
        # Note that sensor weights is larger than it should be
        # This is because the original code didn't factor in
        # the motor neurons. The weight matrix entries for the motor 
        # neurons are initialized to 0.0 and just never updated
        # during evolution
        self.sensor_weights = np.zeros((self.num_of_sensors, \
                                        self.circuit_size))

    def randomize_state(self, random_variable_lower_bound=-1.0, \
            random_variable_upper_bound=1.0):
        """
        Randomizes the state of the network given a set of lower and upper 
        bounds that the neuron states can take.
        """

        self.ctrnn_states = np.random.uniform(random_variable_lower_bound, 
                        random_variable_upper_bound, 
                        size=(self.circuit_size,1))
        self.ctrnn_outputs = \
                    sigmoid(self.gains * self.ctrnn_states + self.biases)
        self.sensor_states = np.zeros((self.num_of_sensors, 1))

    def initialize(self):

        self.randomize_state(0.0, 0.0)
        for i in range(self.num_of_sensors):
            self.set_sensor(i, 0.0)

    def euler_step(self, step_size):
        """
        Steps the network's states and outputs using the Euler method.
        This uses Beer's standard CTRNN equation.
        """

        inputs = np.dot(self.sensor_weights.T, self.sensor_states) \
                    + np.dot(self.circuit_weights.T, self.ctrnn_outputs)
        self.ctrnn_states += step_size * self.rtaus \
                            * (inputs - self.ctrnn_states)
        self.ctrnn_outputs = sigmoid(self.gains 
                                        * self.ctrnn_states + self.biases)

    def neuron_output(self, index):

        return self.ctrnn_outputs[index, 0]

    def set_sensor(self, index, value):

        self.sensor_states[index, 0] = value

    def set_biases(self, bias_sequence):

        if len(bias_sequence) != self.circuit_size:
            raise IndexError("Error: Bias sequence length != circuit size")
        else:
            self.biases = np.array(bias_sequence).reshape(
                                                    self.circuit_size, 1)
    def set_gains(self, gain_sequence):

        if len(gain_sequence) != self.circuit_size:
            raise IndexError("Error: Gain sequence length != circuit size")
        else:
            self.gains = np.array(gain_sequence).reshape(
                                                    self.circuit_size, 1)

    def set_time_constants(self, time_constants):

        if len(time_constants) != self.circuit_size:
            raise IndexError("Error: Time constants len != circuit size")
        else:
            self.taus = np.array(time_constants).reshape(
                                                    self.circuit_size, 1)
            self.rtaus = 1. / self.taus
    
if __name__ == '__main__':
    """
    For testing
    """

    my_sensor = SensorCTRNN(4, 5)
    my_sensor.randomize_state()
    print(my_sensor.ctrnn_states)
    print(my_sensor.ctrnn_outputs)
    print(my_sensor.sensor_states)
    my_sensor.initialize()
    for i in range(my_sensor.num_of_sensors):
        my_sensor.set_sensor(i, np.random.uniform(0,1))
    print(my_sensor.sensor_states)
    my_sensor.euler_step(0.1)
    print(my_sensor.ctrnn_states)
    print(my_sensor.ctrnn_outputs)
    my_sensor.set_biases([0.1,0.2,0.3,0.5])
    print(my_sensor.biases)
    pass