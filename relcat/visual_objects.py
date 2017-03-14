import numpy as np
import math

class Ray:
    """
    Represents a ray, which is a line segment in space.
    """

    def __init__(self, x1=0, y1=0, x2=0, y2=0, angle=0, length=0,
        init_relative_end_x=0, init_relative_end_y=0):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.angle = angle
        self.length = length
        self.init_relative_end_x = init_relative_end_x
        self.init_relative_end_y = init_relative_end_y

class VisualObject:
    """
    Based class for the other objects
    """

    def __init__(self, size, center_xpos, center_ypos):

        self.size = size
        self.center_xpos = center_xpos
        self.center_ypos = center_ypos
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def clip_position(self, left, right, top, bottom):

        self.center_xpos =  (left + self.size) if \
            ((self.center_xpos - self.size) < left) else \
            (right - self.size if (self.center_xpos + self.size) > right \
                else self.center_xpos)
        self.center_ypos = (top + self.size) if \
            ((self.center_ypos - self.size) < top) else \
            ((bottom - self.size) if \
                (self.center_ypos + self.size) > bottom else self.center_ypos)

    def leading_edge(self):

        return self.center_ypos + self.size

    def step(self, step_size):

        self.center_xpos += step_size * self.velocity_x
        self.center_ypos += step_size * self.velocity_y

    def set_position(self, x, y):

        self.center_xpos = x
        self.center_ypos = y 

    def set_positionx(self, x):

        self.set_position(self, x, self.center_ypos)

    def set_positiony(self, y):

        self.set_position(self, self.center_xpos, y)

    def set_size(self, size):

        self.size = size

    def leading_edge_y(self):

        return self.center_ypos + self.size

class Circle(VisualObject):

    def __init__(self, size, center_xpos, center_ypos):
        super().__init__(size, center_xpos, center_ypos)

    def ray_intersection(self, ray):
        """
        Determines whether the ray has intersected the object
        and handles the ray's end-points in the event of
        an intersection.
        """

        dx = ray.x2 - ray.x1
        dy = ray.y2 - ray.y1
        u = ((self.center_xpos - ray.x1) * dx \
            + (self.center_ypos - ray.y1) * dy) / (dx * dx + dy * dy)
        nearX = ray.x1 + u * dx
        nearY = ray.y1 + u * dy
        if (math.sqrt((self.center_xpos - nearX) \
            * (self.center_xpos - nearX) \
            + (self.center_ypos - nearY) \
            * (self.center_ypos - nearY)) > self.size):

            return None

        a = dx * dx + dy * dy
        b = 2 * (dx * (ray.x1 - self.center_xpos) \
            + dy * (ray.y1 - self.center_ypos))
        c = self.center_xpos * self.center_xpos \
            + self.center_ypos * self.center_ypos \
            + ray.x1 * ray.x1 + ray.y1 * ray.y1 - 2 * \
            (self.center_xpos * ray.x1 + self.center_ypos * ray.y1) \
            - self.size * self.size
        i = b * b - 4 * a * c

        if (i == 0):
            u = -b / (2 * a)
            if (u >= 0 and u <= 1):
                ray.x2 = ray.x1 + u * dx
                ray.y2 = ray.y1 + u * dy

        elif (i > 0):
            distance1 = 9999999
            u = (-b + math.sqrt(i)) / (2 * a)
            if (u >= 0 and u <= 1):
                ray.x2 = ray.x1 + u * dx
                ray.y2 = ray.y1 + u * dy
                distance1 = math.sqrt((ray.x2 - ray.x1) \
                            * (ray.x2 - ray.x1) + (ray.y2 - ray.y1) \
                            * (ray.y2 - ray.y1))

            if (u >= 0 and u <= 1):
                end_x2 = ray.x1 + u * dx
                end_y2 = ray.y1 + u * dy
                distance2 = math.sqrt((end_x2 - ray.x1) \
                            * (end_x2 - ray.x1) + (end_y2 - ray.y1) \
                            * (end_y2 - ray.y1))

                if (distance2 < distance1):
                    ray.x2 = end_x2
                    ray.y2 = end_y2

        return None

class Diamond(VisualObject):

    def __init__(self, size, center_xpos, center_ypos):
        super().__init__(size, center_xpos, center_ypos)

    def ray_intersection(self, ray):
        """
        Determines whether the ray has intersected the object
        and handles the ray's end-points in the event of
        an intersection.
        """

        x3 = self.center_xpos - self.size
        y4 = self.center_ypos + self.size
        denom = (y4 - self.center_ypos) * (ray.x2 - ray.x1) \
                - (self.center_xpos - x3) * (ray.y2 - ray.y1)
   
        if (denom != 0):
            ua = ((self.center_xpos - x3) * (ray.y1 - self.center_ypos) \
                - (y4 - self.center_ypos) * (ray.x1 - x3)) / denom
            ub = ((ray.x2 - ray.x1) * (ray.y1 - self.center_ypos) \
                - (ray.y2 - ray.y1) * (ray.x1 - x3)) / denom
            if (ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1):
                ray.x2 = ray.x1 + ua * (ray.x2 - ray.x1)
                ray.y2 = ray.y1 + ua * (ray.y2 - ray.y1)
                return None
                
            x3 = self.center_xpos + self.size
            denom = (y4 - self.center_ypos) * (ray.x2 - ray.x1) \
                    - (self.center_xpos - x3) * (ray.y2 - ray.y1)

            if (denom == 0):
                return None

            ua = ((self.center_xpos - x3) * (ray.y1 - self.center_ypos) \
                - (y4 - self.center_ypos) * (ray.x1 - x3)) / denom
            ub = ((ray.x2 - ray.x1) * (ray.y1 - self.center_ypos) \
                - (ray.y2 - ray.y1) * (ray.x1 - x3)) / denom
            if (ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1):
                ray.x2 = ray.x1 + ua * (ray.x2 - ray.x1)
                ray.y2 = ray.y1 + ua * (ray.y2 - ray.y1)
                return None

class Line(VisualObject):

    def __init__(self, size, center_xpos, center_ypos):
        super().__init__(size, center_xpos, center_ypos)

    def ray_intersection(self, ray):
        x3 = self.center_xpos - self.size
        x4 = self.center_xpos + self.size
        denom = -(x4 - x3) * (ray.y2 - ray.y1)
            
        if (denom == 0):
            return None

        ua = (x4 - x3) * (ray.y1 - self.center_ypos) / denom
        ub = ((ray.x2 - ray.x1) * (ray.y1 - self.center_ypos) \
            - (ray.y2 - ray.y1) * (ray.x1 - x3)) / denom

        if (ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1):
            ray.x2 = ray.x1 + ua * (ray.x2 - ray.x1)
            ray.y2 = ray.y1 + ua * (ray.y2 - ray.y1)
            return None

    def leading_edge(self):

        return self.center_ypos