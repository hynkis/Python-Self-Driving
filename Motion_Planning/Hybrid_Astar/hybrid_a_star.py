"""
Hybrid A* path planning
author: Zheng Zh (@Zhengzh)
"""

import heapq
import scipy.spatial
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("../ReedsSheppPath/")
try:
    from a_star import dp_planning  # , calc_obstacle_map
    import reeds_shepp_path_planning as rs
    from car import move, check_car_collision, WB, plot_car
except:
    raise

import time

import obstacle_map.obstacle_map as obs_map


XY_GRID_RESOLUTION   = 1.0   # [m]
YAW_GRID_RESOLUTION  = np.deg2rad(5.0)  # 5.0 15.0 [rad]
MOTION_RESOLUTION    = 0.1   # 0.5 best? 0.1 [m] path interporate resolution
N_STEER              = 30.0  # 20.0 number of steer command
H_COST               = 1.0
VR                   = 2.0   # robot radius
MAX_STEER            = np.deg2rad(15) # Maximum Steer Angle

SB_COST           = 100.0  # 100.0 switch back penalty cost
BACK_COST         = 5.0    # 5.0 backward penalty cost
STEER_CHANGE_COST = 5.0    # 5.0 steer angle change penalty cost
STEER_COST        = 1.0    # 1.0 steer angle change penalty cost
H_COST            = 5.0    # 5.0. Heuristic cost

show_animation = False

class Node:
    def __init__(self, xind, yind, yawind, direction,
                 xlist, ylist, yawlist, directions,
                 steer=0.0, pind=None, cost=None):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directions = directions
        self.steer = steer
        self.pind = pind
        self.cost = cost


class Path:
    def __init__(self, xlist, ylist, yawlist, directionlist, cost):
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directionlist = directionlist
        self.cost = cost


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN
        inp: input data, single frame or multi frame
        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


class Config:

    def __init__(self, ox, oy, xyreso, yawreso):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.minx = round(min_x_m / xyreso)
        self.miny = round(min_y_m / xyreso)
        self.maxx = round(max_x_m / xyreso)
        self.maxy = round(max_y_m / xyreso)

        self.xw = round(self.maxx - self.minx)
        self.yw = round(self.maxy - self.miny)

        self.minyaw = round(- math.pi / yawreso) - 1
        self.maxyaw = round(math.pi / yawreso)
        self.yaww = round(self.maxyaw - self.minyaw)


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi

    return angle

def calc_motion_inputs():
    """
    Calculate motion input for succeeding neighbor states.
    using steer and direction
    """
    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER, N_STEER),[0.0])):
        for d in [1, -1]:
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kdtree):
    """
    Succeeding neighbor states of current node.
    using steer and direction from "calc_motion_inputs()"
    """
    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kdtree)
        # Boundary and Collision check.
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kdtree):
    """
    Calculate cost and indices of the node with steer, direction primitives.
        - current   : current node
        - steer     : steer primitives
        - direction : direction primitives

    """

    x, y, yaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    xlist, ylist, yawlist = [], [], []
    for dist in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        xlist.append(x)
        ylist.append(y)
        yawlist.append(yaw)

    if not check_car_collision(xlist, ylist, yawlist, ox, oy, kdtree):
        return None

    # d = direction == 1 # check whether direction control input is forward of not.

    xind = round(x / XY_GRID_RESOLUTION)
    yind = round(y / XY_GRID_RESOLUTION)
    yawind = round(yaw / YAW_GRID_RESOLUTION)

    addedcost = 0.0

    # # direction switch penalty
    # if d != current.direction:
    #     addedcost += SB_COST

    # direction switch penalty
    if current.direction != direction:
        addedcost += SB_COST

    # backward drive penalty
    if direction != 1:
        addedcost += BACK_COST

    # steer penalty
    addedcost += STEER_COST * abs(steer)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + addedcost + arc_l

    node = Node(xind, yind, yawind, direction, xlist,
                ylist, yawlist, [direction],
                pind=calc_index(current, config),
                cost=cost, steer=steer)
    # node = Node(xind, yind, yawind, d, xlist,
    #             ylist, yawlist, [d],
    #             pind=calc_index(current, config),
    #             cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.xind == n2.xind and n1.yind == n2.yind and n1.yawind == n2.yawind:
        return True
    return False


def analytic_expantion(current, goal, c, ox, oy, kdtree):

    sx = current.xlist[-1]
    sy = current.ylist[-1]
    syaw = current.yawlist[-1]

    gx = goal.xlist[-1]
    gy = goal.ylist[-1]
    gyaw = goal.yawlist[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(sx, sy, syaw, gx, gy, gyaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kdtree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analystic_expantion(current, goal,
                                         c, ox, oy, kdtree):
    apath = analytic_expantion(current, goal, c, ox, oy, kdtree)

    if apath:
        plt.plot(apath.x, apath.y)
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw = apath.yaw[1:]

        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:]:
            fd.append(d >= 0)

        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fd,
                     cost=fcost, pind=fpind, steer=fsteer)
        return True, fpath

    return False, None


def calc_rs_path_cost(rspath):

    cost = 0.0
    # for l in rspath.lengths:
    #     if l >= 0:  # forward
    #         print("forward cost", l)
    #         cost += l
    #     else:  # back
    #         print("back cost")
    #         cost += abs(l) * BACK_COST

    # backward penalty
    for length in rspath.lengths:
        if length >= 0:  # forward
            cost += length
        else:  # back
            cost += abs(length) * BACK_COST

    # swich back penalty
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:  # switch back
            cost += SB_COST

    # steer penalyty
    for ctype in rspath.ctypes:
        if ctype != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    nctypes = len(rspath.ctypes)
    ulist = [0.0] * nctypes
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = - MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xyreso, yawreso):
    """
    start
    goal
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xyreso: grid resolution [m]
    yawreso: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obkdtree = KDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xyreso, yawreso)

    nstart = Node(round(start[0] / xyreso), round(start[1] / xyreso), round(start[2] / yawreso),
                  True, [start[0]], [start[1]], [start[2]], [True], cost=0)
    ngoal = Node(round(goal[0] / xyreso), round(goal[1] / xyreso), round(goal[2] / yawreso),
                 True, [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    _, _, h_dp = dp_planning(nstart.xlist[-1], nstart.ylist[-1],
                             ngoal.xlist[-1], ngoal.ylist[-1], ox, oy, xyreso, VR)

    pq = []
    openList[calc_index(nstart, config)] = nstart
    heapq.heappush(pq, (calc_cost(nstart, h_dp, ngoal, config),
                        calc_index(nstart, config)))

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.xlist[-1], current.ylist[-1], "xc")
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        isupdated, fpath = update_node_with_analystic_expantion(
            current, ngoal, config, ox, oy, obkdtree)

        # If update node with analysitic expantion, break while loop.
        if isupdated:
            break

        # If no update node with analysitic expantion, check neighbor nodes.
        for neighbor in get_neighbors(current, config, ox, oy, obkdtree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, ngoal, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, fpath, nstart, config)
    return path

def a_star_planning(start, goal, ox, oy, xyreso, yawreso):
    """
    start
    goal
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xyreso: grid resolution [m]
    yawreso: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])

    nstart = Node(round(start[0] / xyreso), round(start[1] / xyreso), round(start[2] / yawreso),
                  True, [start[0]], [start[1]], [start[2]], [True], cost=0)
    ngoal = Node(round(goal[0] / xyreso), round(goal[1] / xyreso), round(goal[2] / yawreso),
                 True, [goal[0]], [goal[1]], [goal[2]], [True])

    path_x, path_y, _ = dp_planning(nstart.xlist[-1], nstart.ylist[-1],
                             ngoal.xlist[-1], ngoal.ylist[-1], ox, oy, xyreso, VR)

    return path_x, path_y


def calc_cost(n, h_dp, goal, c):
    ind = (n.yind - c.miny) * c.xw + (n.xind - c.minx)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, ngoal, nstart, config):
    rx, ry, ryaw = list(reversed(ngoal.xlist)), list(
        reversed(ngoal.ylist)), list(reversed(ngoal.yawlist))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while nid:
        n = closed[nid]
        rx.extend(list(reversed(n.xlist)))
        ry.extend(list(reversed(n.ylist)))
        ryaw.extend(list(reversed(n.yawlist)))
        direction.extend(list(reversed(n.directions)))

        nid = n.pind

    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    # Added by Seong.
    # Normalize Yaw angle.
    for i in range(len(ryaw)):
        while ryaw[i] > np.pi:
            ryaw[i] -= 2*np.pi
        while ryaw[i] < - np.pi:
            ryaw[i] += 2*np.pi

    path = Path(rx, ry, ryaw, direction, finalcost)

    return path


def verify_index(node, c):
    xind, yind = node.xind, node.yind
    if xind >= c.minx and xind <= c.maxx and yind >= c.miny \
            and yind <= c.maxy:
        return True

    return False


def calc_index(node, c):
    ind = (node.yawind - c.minyaw) * c.xw * c.yw + \
        (node.yind - c.miny) * c.xw + (node.xind - c.minx)

    # if ind <= 0:
    #     print("Error(calc_index):", ind)

    return ind


# Added by seong. 19.07.23.
def constraint_search(obs_X, obs_Y, result_path):
    """
    Constraint search algorithm.
    find lower bound and upper bound of each node of result path.

    Inputs:
        - obs_X
        - obs_Y
        - bound_X
        - bound_Y
        - result_path

    Outputs:
        - lower_bounds_x
        - upper_bounds_x
        - lower_bounds_y
        - upper_bounds_y
    """
    lower_bounds_x = []
    lower_bounds_y = []
    upper_bounds_x = []
    upper_bounds_y = []

    # obs_X.extend(bound_X) # concat two lists. obstacle x + boundary x
    # obs_Y.extend(bound_Y) # concat two lists. obstacle y + boundary y
    obs_X = np.array(obs_X) # for index searching.
    obs_Y = np.array(obs_Y)

    print("obs_X :", len(obs_X))
    print("obs_Y :", len(obs_Y))

    print("result_path.xlist :", len(result_path.xlist))
    print("result_path.ylist :", len(result_path.ylist))
    
    iii = 0
    for path_x, path_y in zip(result_path.xlist, result_path.ylist):
        # For Y constraints
        indices_Y = []
        for ind_x, obs_x in enumerate(obs_X):
            # if obs_x == path_x:
            if obs_x-1 < path_x and obs_x+1 > path_x: # path_x is float while obs_x is int. 
                indices_Y.append(ind_x)

        if len(indices_Y) == 0:
            print("No constraint!. path_x :", path_x)

        const_Y = obs_Y[indices_Y]
        sorted_const_Y = sorted(const_Y)
        
        ii = 0
        for y in sorted_const_Y:
            if path_y > y:
                ii = ii + 1
            else:
                lower_bounds_y.append(sorted_const_Y[ii-1])
                upper_bounds_y.append(sorted_const_Y[ii])
                break

        # For X constraints
        indices_X = []
        for ind_y, obs_y in enumerate(obs_Y):
            # if obs_y == path_y:
            if obs_y-1 < path_y and obs_y+1 > path_y:
                indices_X.append(ind_y)

        if len(indices_X) == 0:
            print("No constraint!. path_y :", path_y)

        const_X = obs_X[indices_X]
        sorted_const_X = sorted(const_X)
        
        jj = 0
        for x in sorted_const_X:
            if path_x > x:
                jj = jj + 1
            else:
                lower_bounds_x.append(sorted_const_X[jj-1])
                upper_bounds_x.append(sorted_const_X[jj])
                break

        iii = iii + 1

    return lower_bounds_x, upper_bounds_x, lower_bounds_y, upper_bounds_y

def constraint_search_(obs_X, obs_Y, pred_path):
    """
    Constraint search algorithm.
    find lower bound and upper bound of each node of result path.

    Inputs:
        - obs_X
        - obs_Y
        - pred_path : [num of x, N+1]


    Outputs:
        - lower_bounds_x
        - upper_bounds_x
        - lower_bounds_y
        - upper_bounds_y
    """
    lower_bounds_x = []
    lower_bounds_y = []
    upper_bounds_x = []
    upper_bounds_y = []

    # obs_X.extend(bound_X) # concat two lists. obstacle x + boundary x
    # obs_Y.extend(bound_Y) # concat two lists. obstacle y + boundary y
    obs_X = np.array(obs_X) # for index searching.
    obs_Y = np.array(obs_Y)

    print("obs_X :", len(obs_X))
    print("obs_Y :", len(obs_Y))

    print("pred_path.xlist :", len(pred_path[0,:]))
    print("pred_path.ylist :", len(pred_path[1,:]))
    
    iii = 0
    for path_x, path_y in zip(pred_path[0,:], pred_path[1,:]):
        # For Y constraints
        indices_Y = []
        for ind_x, obs_x in enumerate(obs_X):
            # if obs_x == path_x:
            if obs_x-1 < path_x and obs_x+1 > path_x: # path_x is float while obs_x is int. 
                indices_Y.append(ind_x)

        if len(indices_Y) == 0:
            print("No constraint!. path_x :", path_x)

        const_Y = obs_Y[indices_Y]
        sorted_const_Y = sorted(const_Y)
        
        ii = 0
        for y in sorted_const_Y:
            if path_y > y:
                ii = ii + 1
            else:
                lower_bounds_y.append(sorted_const_Y[ii-1])
                upper_bounds_y.append(sorted_const_Y[ii])
                break

        # For X constraints
        indices_X = []
        for ind_y, obs_y in enumerate(obs_Y):
            # if obs_y == path_y:
            if obs_y-1 < path_y and obs_y+1 > path_y:
                indices_X.append(ind_y)

        if len(indices_X) == 0:
            print("No constraint!. path_y :", path_y)

        const_X = obs_X[indices_X]
        sorted_const_X = sorted(const_X)
        
        jj = 0
        for x in sorted_const_X:
            if path_x > x:
                jj = jj + 1
            else:
                lower_bounds_x.append(sorted_const_X[jj-1])
                upper_bounds_x.append(sorted_const_X[jj])
                break

        iii = iii + 1

    return lower_bounds_x, upper_bounds_x, lower_bounds_y, upper_bounds_y

def constraint_search__(obs_X, obs_Y, path_x, path_y):
    """
    Constraint search algorithm.
    find lower bound and upper bound of each node of result path.

    Inputs:
        - obs_X
    """
    lower_bounds_x = []
    lower_bounds_y = []
    upper_bounds_x = []
    upper_bounds_y = []

    # obs_X.extend(bound_X) # concat two lists. obstacle x + boundary x
    # obs_Y.extend(bound_Y) # concat two lists. obstacle y + boundary y
    obs_X = np.array(obs_X) # for index searching.
    obs_Y = np.array(obs_Y)

    print("obs_X :", len(obs_X))
    print("obs_Y :", len(obs_Y))

    print("path_x :", len(path_x))
    print("path_y :", len(path_y))
    
    iii = 0
    for ix, iy in zip(path_x, path_y):
        # For Y constraints
        indices_Y = []
        for ind_x, obs_x in enumerate(obs_X):
            # if obs_x == ix:
            if obs_x-1 < ix and obs_x+1 > ix: # ix is float while obs_x is int. 
                indices_Y.append(ind_x)

        if len(indices_Y) == 0:
            print("No constraint!. ix :", ix)

        const_Y = obs_Y[indices_Y]
        sorted_const_Y = sorted(const_Y)
        
        ii = 0
        for y in sorted_const_Y:
            if iy > y:
                ii = ii + 1
            else:
                lower_bounds_y.append(sorted_const_Y[ii-1])
                upper_bounds_y.append(sorted_const_Y[ii])
                break

        # For X constraints
        indices_X = []
        for ind_y, obs_y in enumerate(obs_Y):
            # if obs_y == iy:
            if obs_y-1 < iy and obs_y+1 > iy:
                indices_X.append(ind_y)

        if len(indices_X) == 0:
            print("No constraint!. iy :", iy)

        const_X = obs_X[indices_X]
        sorted_const_X = sorted(const_X)
        
        jj = 0
        for x in sorted_const_X:
            if ix > x:
                jj = jj + 1
            else:
                lower_bounds_x.append(sorted_const_X[jj-1])
                upper_bounds_x.append(sorted_const_X[jj])
                break

        iii = iii + 1

    return lower_bounds_x, upper_bounds_x, lower_bounds_y, upper_bounds_y


def main():
    print("Start Hybrid A* planning")

    # ox, oy = [], []
    # ox, oy = obs_map.map_maze(ox, oy)

    # start = [17.0, 20.0, np.deg2rad(-90.0)]
    # goal = [20.0, 42.0, np.deg2rad(0.0)]
    #goal = [17.0, 20.0, np.deg2rad(90.0)]

    #start = [20.0, 42.0, np.deg2rad(0.0)]
    #goal = [55.0, 50.0, np.deg2rad(0.0)]

    start = [0.0, 0.0, np.deg2rad(90.0)]
    goal = [5.0, 20.0, np.deg2rad(90.0)]

    ox, oy = [], []
    pos_cars = [(0,15), (5,2)]
    ox, oy = obs_map.map_road(ox, oy, pos_cars)

    plt.plot(ox, oy, ".k")
    rs.plot_arrow(start[0], start[1], start[2], fc='g')
    rs.plot_arrow(goal[0], goal[1], goal[2])

    plt.grid(True)
    plt.axis("equal")

    # Path Planning - Hybrid A*
    tic = time.time()
    path = hybrid_a_star_planning(
        start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
    toc = time.time()
    print("Process time (hybrid A*):", (toc - tic))
    
    # Constraint Search
    tic = time.time()
    lb_x, ub_x, lb_y, ub_y = constraint_search(ox, oy, path)
    toc = time.time()
    print("Process time (constraint_search):", (toc - tic))

    x = path.xlist
    y = path.ylist
    yaw = path.yawlist
    ind = 0 # index for lower/upper bounds.

    for ix, iy, iyaw in zip(x, y, yaw):
        # plotting during animation.
        plt.cla()
        plt.plot(ox, oy, ".k")
        
        plt.plot(lb_x[ind], iy, "xr", markersize=10) # Plot Constraints
        plt.plot(ub_x[ind], iy, "xr", markersize=10) # Plot Constraints
        plt.plot(ix, lb_y[ind], "xb", markersize=10) # Plot Constraints
        plt.plot(ix, ub_y[ind], "xb", markersize=10) # Plot Constraints
        ind = ind + 1

        plt.plot(x, y, "-r", label="Hybrid A* path")
        plt.grid(True)
        plt.axis("equal")
        plot_car(ix, iy, iyaw)
        plt.pause(0.0001)

    print(__file__ + " done!!")
    plt.show()


if __name__ == '__main__':
    main()