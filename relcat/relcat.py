"""
Main module for running the relational categorization task.
Contains the class for creating a simulation.
"""

import numpy as np
import math
import .sensor_agent
import .visual_objects

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
        parameters = {
        'world_left': 0,
        'world_right': 400,
        'world_top': 0,
        'world_bottom': 300,
        'bilateral_symmetry': True,
        'agent_radius': 15,
        'mass': 0.2,
        'visual_angle': np.pi / 6,
        'num_rays': 7,
        'num_interneurons': 5,
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

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.circuit_size = self.num_interneurons + 5
        self.initial_agent_x = (self.world_right - self.world_left) / 2.
        self.initial_agent_y = self.world_bottom - self.agent_radius
        self.agent_top = self.initial_agent_y + self.agent_radius
        self.vertical_offset = self.agent_top - self.max_ray_length

        if self.bilateral_symmetry:
            if self.num_rays % 2 == 0:
                num_sensor_weights = int(self.num_rays / 2
                                        * self.num_interneurons)
            else:
                num_sensor_weights = (self.num_rays - 1) / 2 \
                                * self.num_interneurons \
                                + math.ceil(self.num_interneurons / 2)

            num_motor_weights = self.num_interneurons
            num_loops = math.ceil(self.num_interneurons / 2)
            num_cross_connections = int(math.ceil(self.num_interneurons / 2) \
                                        * self.num_interneurons / 2)
            num_side_connections = math.ceil(self.num_interneurons / 2) \
                                * (math.ceil(self.num_interneurons / 2) - 1)

            self.num_parameters = num_sensor_weights + num_motor_weights \
                                    + num_loops + num_cross_connections \
                                    + num_side_connections \
                                    + math.ceil(self.num_interneurons / 2) \
                                    + math.ceil(self.num_interneurons / 2)
        else:
            # sensor weights : num_interneurons * num_rays
            # circuit weights : num_interneurons * (num_interneurons + 2)
            # biases: num_interneurons + 2
            # time constants: num_interneurons + 2
            self.num_parameters = self.num_interneurons * self.num_rays \
                + self.num_interneurons * (self.num_interneurons + 2) \
                + self.num_interneurons + 2 \
                + self.num_interneurons + 2

    def __call__(self, x):

        # Generate agent
        agent = sensor_agent.SensorAgent(self.agent_radius,
            self.mass, self.visual_angle, self.num_rays,
            self.max_ray_length, self.initial_agent_x, self.initial_agent_y,
            self.circuit_size, self.max_velocity)

        # Generate circle
        ball = visual_objects.Circle(self.circle_size,
                                    self.initial_agent_x, self.world_top)

        # Map parameter values
        map_search_parameters(x, agent.nervous_system)

        # Run trials

        # Return fitness

    def map_search_parameters(self, x, nervous_system):
        """
        x : numpy array of search parameter values
        """

        if self.bilateral_symmetry:

            # Sensor Weights
            rescale_parameter(x[:], self.min_weight,
                self.max_weight, self.min_search_value,
                self.max_search_value)

            # Circuit weights

            # Biases

            # Time constants

        else:

            sensor_index_end = self.num_rays
            circuit_index_end = self.num_rays + self.num_interneurons
            bias_index_end = self.num_rays + 2 * self.num_interneurons

            # Sensor Weights
            nervous_system.sensor_weights = \
                rescale_parameter(x[:sensor_index_end], 
                self.min_weight,
                self.max_weight, self.min_search_value,
                self.max_search_value)

            # Circuit weights
            nervous_system.circuit_weights = \
                rescale_parameter(x[sensor_index_end:circuit_index_end], 
                self.min_weight,
                self.max_weight, self.min_search_value,
                self.max_search_value)

            # Biases
            nervous_system.biases = \
                rescale_parameter(x[circuit_index_end:bias_index_end], 
                self.min_bias,
                self.max_bias, self.min_search_value,
                self.max_search_value)

            # Time constants
            nervous_system.taus = rescale_parameter(x[bias_index_end:], 
                self.min_tau,
                self.max_tau, self.min_search_value,
                self.max_search_value)

    def run_trials(self, agent, ball):

    def 

def recale_parameter(parameter, min_param_value, max_param_value,
    min_search_value, max_search_value):

    scale = (max_param_value - min_param_value) \
                / (max_search_value - min_search_value)
    bias = min_param_value - scale * min_search_value

    return scale * parameter + bias

if __name__ == '__main__':
    """
    For testing
    """

    pass