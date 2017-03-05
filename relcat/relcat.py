"""
Main module for running the relational categorization task.
Contains the class for creating a simulation.
"""

import numpy as np
import math

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
        'circle_size': 20,
        'obj_velocity': 3.0,
        'circle_min_diameter': 20.0,
        'circle_max_diameter': 50.0,
        'circle_difference': 5.0,
        'step_size': 0.1,
        'min_weight': -16.,
        'max_weight': 16.,
        'min_bias': -16.,
        'max_bias': 16.,
        'min_tau': 1.,
        'max_tau': 30.
        }

        for key, default in parameter_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.circuit_size = self.num_interneurons + 5
        self.initial_agent_x = (self.world_right - self.world_left) / 2.
        self.initial_agent_y = self.world_bottom - self.agent_radius
        self.agent_top = self.initial_agent_y + self.agent_radius
        self.vertical_offset = self.agent_top - self.max_ray_length

        # Hold these, might ditch them... not sure why they needed....
        self.last_ray_index = self.num_rays - 1
        self.last_int_index = self.num_interneurons - 1
        self.left_motor_index = self.circuit_size - 2
        self.right_motor_index = self.circuit_size - 1
        # End trash

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

    def __call__():



if __name__ == '__main__':
    """
    For testing
    """

    pass