import numpy as np
import math
from .sensor_ctrnn import SensorCTRNN
from .visual_objects import Ray

def reset_ray(ray, theta, center_xpos, center_ypos, radius, max_ray_length):

    ray.angle = theta
    ray.init_relative_end_x = radius * math.sin(ray.angle) \
                            + max_ray_length * math.sin(ray.angle)
    ray.x2 = center_xpos + ray.init_relative_end_x
    ray.init_relative_end_y = radius * math.cos(ray.angle) \
                            + max_ray_length * math.cos(ray.angle)
    ray.y2 = center_ypos - ray.init_relative_end_y
    ray.x1 = center_xpos + radius * math.sin(ray.angle)
    ray.y1 = center_ypos - radius * math.cos(ray.angle)
    dx = ray.x2 - ray.x1
    dy = ray.y2 - ray.y1
    ray.length = math.sqrt(dx * dx + dy * dy)

class SensorAgent:
    """
    A class for the agent that partakes in the relational
    categorization task. It is adapted from code made by
    Randall Beer in C++.

    """

    def __init__(self, agent_radius, agent_mass, agent_visual_angle,
        num_of_rays, max_ray_length, agent_xpos, agent_ypos, circuit_size,
        max_velocity):

        self.radius = agent_radius
        self.mass = agent_mass
        if num_of_rays == 1:
            self.visual_angle = 0
        else:
            self.visual_angle = agent_visual_angle
        self.num_of_rays = num_of_rays
        self.max_ray_length = max_ray_length
        self.xpos = agent_xpos
        self.ypos = agent_ypos
        self.circuit_size = circuit_size
        self.max_velocity = max_velocity
        self.velocity_x = 0.0

        self.nervous_system = SensorCTRNN(self.circuit_size, self.num_of_rays)
        self.rays = [ Ray() for i in range(self.num_of_rays) ]

        self.reset()

    def reset(self):

        for i, theta in enumerate(np.linspace(-self.visual_angle/2.0, 
                                    self.visual_angle/2.0, self.num_of_rays)):
            reset_ray(self.rays[i], theta, self.xpos, self.ypos, 
                    self.radius, self.max_ray_length)

    def set_position_x(self, x):

        self.set_position(x, self.ypos)

    def set_position_y(self, y):

        self.set_position(self.xpos, y)

    def set_position(self, x, y):

        self.xpos = x
        self.ypos = y

        for ray in self.rays:
            ray.x2 += x - self.xpos
            ray.x1 += x - self.xpos
            ray.y1 += y - self.ypos

    def clip_position(self, left, right):

        if (self.xpos - self.radius < left):
            dx = left - self.xpos + self.radius
            self.xpos = left + self.radius

            for ray in self.rays:
                ray.x1 += dx
                ray.x2 += dx

        elif (self.xpos + self.radius > right):
            dx = self.xpos + self.radius - right
            self.xpos = right - self.radius
            
            for ray in self.rays:
                ray.x1 -= dx
                ray.x2 -= dx

    def initialize_ray_sensors(self, visual_obj, visual_obj2=None):

        # Reset the ray positions
        for ray in self.rays:
            ray.x2 = self.xpos + ray.init_relative_end_x
            ray.y2 = self.ypos - ray.init_relative_end_y

        # Clip each ray to the visual object
        for ray in self.rays:
            visual_obj.ray_intersection(ray)
            if visual_obj2 != None:
                visual_obj2.ray_intersection(ray)
            dx = ray.x2 - ray.x1
            dy = ray.y2 - ray.y1
            ray.length = math.sqrt(dx * dx + dy * dy)

        # Update the visual sensors states (fraction of part cut off)
        for i, ray in enumerate(self.rays):
            self.nervous_system.set_sensor(i, 
                (self.max_ray_length - ray.length) / self.max_ray_length)

    def step(self, step_size, locked):

        # Update the nervous system
        self.nervous_system.euler_step(step_size)

        # Update the body effectors
        if locked:
            left_force = 0.0
            right_force = 0.0
        else:
            left_force = self.nervous_system.neuron_output(\
                                        self.nervous_system.circuit_size-2)
            right_force = self.nervous_system.neuron_output(\
                                        self.nervous_system.circuit_size-1)

        # Update the agent's position
        self.velocity_x = (left_force - right_force) / self.mass
        if (self.velocity_x < -self.max_velocity):
            self.velocity_x = -self.max_velocity
        if (self.velocity_x > self.max_velocity):
            self.velocity_x = self.max_velocity

        self.set_position(self.xpos + step_size * self.velocity_x, self.ypos)

    def one_obj_step(self, step_size, visual_obj, locked):
        """
        Step the agent with one visual object
        """

        self.initialize_ray_sensors(visual_obj)
        self.step(step_size, locked)

    def two_obj_step(self, step_size, visual_obj1, visual_obj2, locked):
        """
        Step the agent with two visual objects
        """

        self.initialize_ray_sensors(visual_obj1, visual_obj2)
        self.step(step_size, locked)

if __name__ == '__main__':
    """
    testing
    """

    pass