"""
Main module for running the relational categorization task.
Contains the class for creating a simulation.
"""

import numpy as np
import math
from .sensor_agent import SensorAgent
from .visual_objects import Circle

class RelationalCategorization:

    def __init__(self, **kwargs):
        """
        Parameter list:
        world_left
        world_right
        world_top
        world_bottom
        bilateral_symmetry
        agent_radius
        mass
        visual_angle
        num_rays
        num_interneurons
        max_distance
        max_ray_length
        max_velocity
        circle_size
        obj_velocity
        circle_min_diameter
        circle_max_diameter
        circle_difference
        step_size
        min_weight
        max_weight
        min_bias
        max_bias
        min_tau
        max_tau

        """

        # Default parameters
        parameter_defaults = {
        'world_left': 0,
        'world_right': 400,
        'world_top': 0,
        'world_bottom': 300,
        'bilateral_symmetry': False,
        'agent_radius': 15,
        'mass': 0.2,
        'visual_angle': np.pi / 6,
        'num_rays': 7,
        'num_interneurons': 3,
        'max_distance': 75,
        'max_ray_length': 220,
        'max_velocity': 5,
        'obj_velocity': 3.0,
        'circle_size': 20,
        'circle_min_diameter': 20.0,
        'circle_max_diameter': 50.0,
        'circle_difference': 5.0,
        'step_size': 0.1,
        'min_weight': -16.,
        'max_weight': 16.,
        'min_bias': -16.,
        'max_bias': 16.,
        'min_tau': 1.,
        'max_tau': 30.,
        'min_search_value': 0.0,
        'max_search_value': 1.0
        }

        for key, default in parameter_defaults.items():
            setattr(self, key, kwargs.get(key, default))

        self.circuit_size = self.num_interneurons + 2
        self.initial_agent_x = (self.world_right - self.world_left) / 2.
        self.initial_agent_y = self.world_bottom - self.agent_radius
        self.agent_top = self.initial_agent_y + self.agent_radius
        self.vertical_offset = self.agent_top - self.max_ray_length
        if self.bilateral_symmetry:
            if self.num_rays % 2 == 0:
                self.num_sensor_weights = int(self.num_rays / 2
                                        * self.num_interneurons)
            else:
                self.num_sensor_weights = int((self.num_rays - 1) / 2 
                                * self.num_interneurons 
                                + math.ceil(self.num_interneurons / 2))

            num_motor_weights = self.num_interneurons
            num_loops = math.ceil(self.num_interneurons / 2)
            num_cross_connections = (self.num_interneurons // 2)**2
            num_side_connections = math.ceil(self.num_interneurons / 2) \
                                * (math.ceil(self.num_interneurons / 2) - 1)
            self.num_circuit_weights = num_motor_weights + num_loops \
                                    + num_cross_connections\
                                    + num_side_connections
            self.num_parameters = self.num_sensor_weights \
                                    + self.num_circuit_weights \
                                    + math.ceil(self.num_interneurons / 2) \
                                    + math.ceil(self.num_interneurons / 2) \
                                    + 2 # for motor neurons

        else:
            # sensor weights : num_interneurons * num_rays
            # circuit weights : num_interneurons * (num_interneurons + 2)
            # biases: num_interneurons + 2
            # time constants: num_interneurons + 2
            self.num_sensor_weights = self.num_interneurons * self.num_rays
            self.num_circuit_weights = self.num_interneurons \
                                        * (self.num_interneurons + 2)
            self.num_parameters = self.num_sensor_weights \
                + self.num_circuit_weights \
                + self.num_interneurons + 2 \
                + self.num_interneurons + 2

    def __call__(self, x):

        # Generate agent
        agent = SensorAgent(self.agent_radius,
            self.mass, self.visual_angle, self.num_rays,
            self.max_ray_length, self.initial_agent_x, self.initial_agent_y,
            self.circuit_size, self.max_velocity)

        # Generate circle
        ball = Circle(self.circle_size,
                    self.initial_agent_x, self.world_top,
                    0.0, self.obj_velocity)

        # Map parameter values
        self.map_search_parameters(x, agent.nervous_system)

        # Run trials
        fitness = self.run_trials(agent, ball)

        # Convert to cost
        return 1. / fitness

    def map_search_parameters(self, x, nervous_system):
        """
        x : numpy array of search parameter values
        """


        if self.bilateral_symmetry:

            sensor_index_end = self.num_sensor_weights
            circuit_index_end = sensor_index_end + self.num_circuit_weights
            bias_index_end = circuit_index_end \
                                + math.ceil(self.circuit_size / 2)

            # Sensor Weights
            sensor_weights = rescale_parameter(x[:sensor_index_end], 
                self.min_weight,
                self.max_weight, self.min_search_value,
                self.max_search_value)
            index_counter = 0
            for i in range(int(self.num_rays / 2)):
                for j in range(self.num_interneurons):
                    nervous_system.sensor_weights[i,j] = \
                        sensor_weights[index_counter]
                    nervous_system.sensor_weights[self.num_rays - i - 1, 
                        self.num_interneurons - j - 1] = \
                        sensor_weights[index_counter]
                    index_counter += 1

            if self.num_rays % 2 == 1:
                for j in range(math.ceil(self.num_interneurons / 2)):
                    nervous_system.sensor_weights[int(self.num_rays 
                        / 2), j] = sensor_weights[index_counter]
                    nervous_system.sensor_weights[int(self.num_rays 
                        / 2), self.num_interneurons - j - 1] \
                                = sensor_weights[index_counter]
                    index_counter += 1

            # index_counter == sensor_index_end when over

            # Circuit weights
            circuit_weights = rescale_parameter(\
                x[sensor_index_end:circuit_index_end], 
                self.min_weight,
                self.max_weight, self.min_search_value,
                self.max_search_value)

            index_counter = 0
            for i in range(int(self.num_interneurons / 2)):
                for j in range(self.num_interneurons):
                    nervous_system.circuit_weights[i,j] = \
                        circuit_weights[index_counter]
                    nervous_system.circuit_weights[self.num_interneurons 
                        - i - 1, self.num_interneurons - j - 1] = \
                        circuit_weights[index_counter]
                    index_counter += 1

            if self.num_interneurons % 2 == 1:
                for j in range(math.ceil(self.num_interneurons / 2)):
                    nervous_system.circuit_weights[int(\
                        self.num_interneurons / 2), j] \
                        = circuit_weights[index_counter]
                    nervous_system.circuit_weights[int(\
                        self.num_interneurons / 2), 
                        self.num_interneurons - j - 1] \
                        = circuit_weights[index_counter]
                    index_counter += 1

            for i in range(int(self.num_interneurons / 2)):
                for j in range(2):
                    nervous_system.circuit_weights[i, 
                        self.num_interneurons + j] \
                        = circuit_weights[index_counter]
                    nervous_system.circuit_weights[self.num_interneurons \
                        - i - 1, self.circuit_size - j - 1] \
                        = circuit_weights[index_counter]
                    index_counter += 1

            if self.num_interneurons % 2 == 1:
                for j in range(1):
                    nervous_system.circuit_weights[int(\
                        self.num_interneurons / 2), 
                        self.num_interneurons + j] \
                        = circuit_weights[index_counter]
                    nervous_system.circuit_weights[int(\
                        self.num_interneurons / 2), 
                        self.circuit_size - j - 1] \
                        = circuit_weights[index_counter]
                    index_counter += 1                    

            # Biases
            bias_pars = \
                rescale_parameter(x[circuit_index_end:bias_index_end], 
                self.min_bias,
                self.max_bias, self.min_search_value,
                self.max_search_value)
            biases = np.zeros(self.circuit_size)
            for i in range(math.ceil(self.num_interneurons / 2)):
                biases[i] = bias_pars[i]
                biases[self.num_interneurons - i - 1] \
                    = bias_pars[i]
            # Motor biases
            biases[-2] = bias_pars[-1]
            biases[-1] = bias_pars[-1]
            nervous_system.set_biases(biases)

            # Time constants
            time_constant_pars = rescale_parameter(x[bias_index_end:], 
                self.min_tau,
                self.max_tau, self.min_search_value,
                self.max_search_value)
            time_constants = np.zeros(self.circuit_size)
            for i in range(math.ceil(self.num_interneurons / 2)):
                time_constants[i] = time_constant_pars[i]
                time_constants[self.num_interneurons - i - 1] \
                    = time_constant_pars[i]

            # Motor time constants
            time_constants[-2] = time_constant_pars[-1]
            time_constants[-1] = time_constant_pars[-1]
            nervous_system.set_time_constants(time_constants)

        else:

            sensor_index_end = self.num_sensor_weights
            circuit_index_end = sensor_index_end + self.num_circuit_weights
            bias_index_end = circuit_index_end + self.circuit_size

            # Sensor Weights
            sensor_weights = rescale_parameter(x[:sensor_index_end], 
                        self.min_weight,
                        self.max_weight, self.min_search_value,
                        self.max_search_value)
            for i in range(self.num_rays):
                for j in range(self.num_interneurons):
                    nervous_system.sensor_weights[i,j] = \
                        sensor_weights[self.num_interneurons * i + j]
            # Circuit weights
            circuit_weights = rescale_parameter(\
                x[sensor_index_end:circuit_index_end], 
                self.min_weight,
                self.max_weight, self.min_search_value,
                self.max_search_value)
            for i in range(self.num_interneurons):
                for j in range(self.num_interneurons):
                    nervous_system.circuit_weights[i,j] = \
                        circuit_weights[self.num_interneurons * i + j]

            for i in range(self.num_interneurons):
                for j in range(2):
                    nervous_system.circuit_weights[i, \
                        self.num_interneurons + j] \
                            = circuit_weights[self.num_interneurons 
                                                * self.num_interneurons
                                                + 2 * i + j]
            # Biases
            nervous_system.set_biases(
                rescale_parameter(x[circuit_index_end:bias_index_end], 
                self.min_bias,
                self.max_bias, self.min_search_value,
                self.max_search_value))

            # Time constants
            nervous_system.set_time_constants(
                rescale_parameter(x[bias_index_end:], 
                self.min_tau,
                self.max_tau, self.min_search_value,
                self.max_search_value))

    def run_trials(self, agent, ball):

        result_matrix = np.zeros((int(self.circle_max_diameter 
                                        / self.circle_difference), 
                                int(self.circle_max_diameter 
                                    / self.circle_difference)))
        for i in range(result_matrix.shape[0]):
            for j in range(result_matrix.shape[1]):
                if i != j:
                    ball_size = i * self.circle_difference \
                                    + self.circle_min_diameter
                    compare_ball_size = j * self.circle_difference \
                                    + self.circle_min_diameter
                    result_matrix[i,j] = self.trial(agent, ball, 
                                                ball_size, compare_ball_size)

        return self.eval_fitness(result_matrix)

    def _record_data(self, agent, ball, start=True):

        if start:
            self.time_records = [0]
            self.object_records = {}
            self.object_records['ball'] = {'x': [ball.center_xpos],
                                            'y': [ball.center_ypos],
                                            'radius':[ball.size]}
            self.object_records['agent'] = {'x': [agent.xpos], 
                                            'y': [agent.ypos],
                                            'radius': [agent.radius]}
            for i, ray in enumerate(agent.rays):
                self.object_records[i] = {'x1':[ray.x1], 
                                        'y1':[ray.y1], 
                                        'x2':[ray.y2], 
                                        'y2':[ray.x2]}

        else:

            self.time_records.append(self.time_records[-1] + self.step_size)
            self.object_records['ball']['x'].append(ball.center_xpos)
            self.object_records['ball']['y'].append(ball.center_ypos)
            self.object_records['ball']['radius'].append(ball.size)
            self.object_records['agent']['x'].append(agent.xpos)
            self.object_records['agent']['y'].append(agent.ypos)
            self.object_records['agent']['radius'].append(agent.radius)
            for i, ray in enumerate(agent.rays):
                self.object_records[i]['x1'].append(ray.x1)
                self.object_records[i]['y1'].append(ray.y1)
                self.object_records[i]['x2'].append(ray.x2)
                self.object_records[i]['y2'].append(ray.y2)

    def trial(self, agent, ball, presented_ball_size, comparison_ball_size,
        record=False):

        agent.set_position(self.initial_agent_x, self.initial_agent_y)
        agent.reset_rays()
        agent.nervous_system.initialize()
        agent.velocity_x = 0.0

        initial_object_y = self.initial_agent_y - (agent.radius 
                                                    + agent.max_ray_length 
                                                    + presented_ball_size)
        ball.set_position(self.initial_agent_x, initial_object_y)
        ball.set_size(presented_ball_size / 2.0)
        agent.initialize_ray_sensors(ball)

        # If recording create initial setup
        if record:
            self._record_data(agent, ball, start=True)

        # First drop presented ball, hold agent still
        while (ball.leading_edge_y() < agent.ypos - agent.radius):
            ball.step(self.step_size)
            agent.one_obj_step(self.step_size, ball, True)

            if record:
                self._record_data(agent, ball, start=False)

        initial_object_y = self.initial_agent_y - (agent.radius 
                                                    + agent.max_ray_length
                                                    + comparison_ball_size)
        ball.set_position(self.initial_agent_x, initial_object_y)
        ball.set_size(comparison_ball_size / 2.0)
        agent.reset_rays()

        # Second drop comparison ball, let agent move
        while (ball.leading_edge_y() < agent.ypos - agent.radius):
            ball.step(self.step_size)
            agent.one_obj_step(self.step_size, ball, False)
            # keep ball within world
            agent.clip_position(self.world_left, self.world_right)
            if record:
                self._record_data(agent, ball, start=False)

        normalized_distance = abs(ball.center_xpos - agent.xpos) \
                                / self.max_distance
        normalized_distance = 1 if normalized_distance > 1 \
                                else normalized_distance

        return 1 - normalized_distance \
                if presented_ball_size > comparison_ball_size \
                else normalized_distance

    def eval_fitness(self, fitness_matrix):

        col_avg = 0.0
        row_avg = 0.0
        for i in range(1, fitness_matrix.shape[0] - 1):

            col_sum = 0.0
            row_sum = 0.0
            num_sums = 0
            for j in range(fitness_matrix.shape[1]):

                if i == j:
                    col_avg += col_sum / num_sums
                    row_avg += row_sum / num_sums
                    col_sum = 0.0
                    row_sum = 0.0
                    num_sums = 0
                else:
                    num_sums += 1
                    col_sum += fitness_matrix[j,i]
                    row_sum += fitness_matrix[i,j]

            col_avg += col_sum / num_sums
            row_avg += row_sum / num_sums

        return min(col_avg / (fitness_matrix.shape[0] - 2) / 2, 
                    row_avg / (fitness_matrix.shape[0] - 2) / 2)

    def run_test_trial(self, x, ball_size, comparison_ball_size):

        # Generate agent
        agent = SensorAgent(self.agent_radius,
            self.mass, self.visual_angle, self.num_rays,
            self.max_ray_length, self.initial_agent_x, self.initial_agent_y,
            self.circuit_size, self.max_velocity)

        # Generate circle
        ball = Circle(self.circle_size,
                    self.initial_agent_x, self.world_top,
                    0.0, self.obj_velocity)

        # Map parameter values
        self.map_search_parameters(x, agent.nervous_system)

        return self.trial(agent, ball, ball_size, 
                                    comparison_ball_size, True)

def rescale_parameter(search_value, min_param_value, max_param_value,
    min_search_value, max_search_value):

    scale = (max_param_value - min_param_value) \
                / (max_search_value - min_search_value)
    bias = min_param_value - scale * min_search_value

    return scale * search_value + bias

if __name__ == '__main__':
    """
    For testing
    """

    pass