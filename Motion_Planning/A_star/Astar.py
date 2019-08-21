"""
A* grid planning
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)
See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)
"""

import math

import matplotlib.pyplot as plt

import time

show_animation = True


class AStarPlanner:

    def __init__(self, ox, oy, reso, rr):
        """
        Initialize grid map for a star planning
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x       # index of grid
            self.y = y       # index of grid
            self.cost = cost # cost
            self.pind = pind # previous index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search
        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position  [m]
            gx: goal x position  [m]
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        

        nstart = self.Node(self.calc_xyindex(sx, self.minx),
                           self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx),
                          self.calc_xyindex(gy, self.miny), 0.0, -1)

        open_set = dict()
        closed_set = dict()

        open_set[self.calc_grid_index(nstart)] = nstart

        tic_planning = time.time()
        
        check_iter = 0
        
        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break
            
            # Find minimum index(Key) in open_set(Dict) w.r.t. (cost+heuristic). MinHeap.
            tic_minheap = time.time()

            c_id = min(
                open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(open_set[o], ngoal))
            toc_minheap = time.time()
            
            current = open_set[c_id]

            # show graph
            tic_show = time.time()

            # if show_animation:  # pragma: no cover
            #     plt.plot(self.calc_grid_position(current.x, self.minx),
            #              self.calc_grid_position(current.y, self.miny), "xc")
            #     # if len(closed_set.keys()) % 10 == 0:
            #     #     plt.pause(0.001)
            toc_show = time.time()
            

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                # motion : list of [dx, dy, cost]
                node = self.Node(current.x + self.motion[i][0],     # x + dx
                                 current.y + self.motion[i][1],     # y + dy
                                 current.cost + self.motion[i][2],  # current cost + new cost
                                 c_id)                              # min cost node

                n_id = self.calc_grid_index(node)


                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

            check_iter = check_iter + 1
        

        rx, ry = self.calc_final_path(ngoal, closed_set)

        toc_planning = time.time()

        print("check iteration :", check_iter)
        print("min heap :", toc_minheap - tic_minheap)
        print("show graph :", toc_show - tic_show)
        print("Planning time :", toc_planning - tic_planning)

        return rx, ry

    def calc_final_path(self, ngoal, closedset):
        # generate final course
        rx = [self.calc_grid_position(ngoal.x, self.minx)]
        ry = [self.calc_grid_position(ngoal.y, self.miny)]
        pind = ngoal.pind

        while pind != -1:
            # Keep append indices until the Previous index is -1. => Path from start to goal.
            n = closedset[pind]
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            pind = n.pind

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        # A star heuristic
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)
        return d

    def calc_grid_position(self, index, minp):
        # calculate grid position. from grid index to real position.
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        # x,y_index = (pos - min_pos) / resolution
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        # grid_index = num. of y * width + num. of x
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        """
        Check Boundary and Obstacles in map.

        Collision : return False
        No Collision : return True
        """
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        # boundary check (min, max boundary)
        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        # collision check
        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        tic_obs = time.time()

        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        print("minx:", self.minx)
        print("miny:", self.miny)
        print("maxx:", self.maxx)
        print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx) / self.reso) # num. of planner node (x)
        self.ywidth = round((self.maxy - self.miny) / self.reso) # num. of planner node (y)
        print("xwidth:", self.xwidth)
        print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                             for i in range(self.xwidth)] # initialize obstacle nodes in map as False.

        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)

            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)

                for iox, ioy in zip(ox, oy):
                    # Make Configuration space with robot radius(rr). additional obstacle space as robot size(rr).
                    d = math.sqrt((iox - x) ** 2 + (ioy - y) ** 2)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break
        
        toc_obs = time.time()
        print("Process time for obstacle map :", toc_obs - tic_obs)

    @staticmethod
    def get_motion_model():
        """
        Motion model
        : left, right, up, down, left_up, left_down, right_up, right_down
        Cost == distance

        motion : list of [dx, dy, cost]
        """
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 0.  # [m]
    sy = 0.  # [m]
    gx = -5.0  # [m]
    gy = 45.0  # [m]
    grid_size = 1.0  # [m]
    robot_radius = 2.0  # [m]

    # set obstable positions
    ox, oy = [], []

    # Vehicles on a road

    # Road
    for i in range(0, 51):
        ox.append(-10)
        oy.append(i)
    for i in range(0, 51):
        ox.append(10)
        oy.append(i)

    # Vehicles
    ox.append(-1)
    ox.append(-1)
    ox.append(1)
    ox.append(1)

    oy.append(5)
    oy.append(8)
    oy.append(5)
    oy.append(8)
    
    # for i in range(-1, 2):
    #     for j in range(5, 9):
    #         ox.append(i)
    #         oy.append(j)
    for i in range(-5, -2):
        for j in range(0, 4):
            ox.append(i)
            oy.append(j)


    # # Wall obstacle map
    # for i in range(-10, 60):
    #     ox.append(i)
    #     oy.append(-10.0)
    # for i in range(-10, 60):
    #     ox.append(60.0)
    #     oy.append(i)
    # for i in range(-10, 61):
    #     ox.append(i)
    #     oy.append(60.0)
    # for i in range(-10, 61):
    #     ox.append(-10.0)
    #     oy.append(i)
    # for i in range(-10, 40):
    #     ox.append(20.0)
    #     oy.append(i)
    # for i in range(0, 40):
    #     ox.append(40.0)
    #     oy.append(60.0 - i)

    # for i in range(0, 20):
    #     ox.append(i)
    #     oy.append(40)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)

    tic = time.time()
    # a_star.calc_obstacle_map(ox, oy)
    rx, ry = a_star.planning(sx, sy, gx, gy)
    toc = time.time()

    print("Process time :", (toc - tic))

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()