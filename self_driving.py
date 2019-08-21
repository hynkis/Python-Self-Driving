import numpy as np

import math
import time
import matplotlib.pyplot as plt
import sys

# for MPC
import scipy as sp
import scipy.sparse as sparse

# ===== for Motion Planning =====
sys.path.append("./Motion_Planning/Hybrid_Astar/")
sys.path.append("./Motion_Planning/ReedsSheppPath/")
from Motion_Planning.Hybrid_Astar import hybrid_a_star
from Motion_Planning.Hybrid_Astar.obstacle_map import obstacle_map as obs_map
from Motion_Planning.ReedsSheppPath import reeds_shepp_path_planning as reeds_sheep

# ===== for Dynamics =====
sys.path.append("./Vehicle_Dynamics/")
from Vehicle_Dynamics import vehicle_dynamics

# ===== for Control =====
sys.path.append("./Control/MPC")
import mpc_path_tracking
from visualization_vehicle import plot_car

# ===== Vehicle parameters =====
XY_GRID_RESOLUTION   = 1.0   # [m]
YAW_GRID_RESOLUTION  = np.deg2rad(5.0)  # 5.0 15.0 [rad]
MOTION_RESOLUTION    = 0.5   # 0.5 best? 0.1 [m] path interporate resolution
N_STEER              = 20.0  # 20.0 number of steer command
VR                   = 2.0   # robot radius

SB_COST           = 100.0  # 100.0 switch back penalty cost
BACK_COST         = 5.0    # 5.0 backward penalty cost
STEER_CHANGE_COST = 5.0    # 5.0 steer angle change penalty cost
STEER_COST        = 1.0    # 1.0 steer angle change penalty cost
H_COST            = 5.0    # 5.0. Heuristic cost

L_F = 1.25
L_R = 1.40
DT = 0.02   # sampling time

M = 1300
WIDTH = 1.78
LENGTH = 4.25
TURNING_CIRCLE = 10.4
C_D = 0.34
A_F = 2.0
C_ROLL = 0.015

def main():

    # ===== Vehicle Dynamics ===== #
    vehicle = vehicle_dynamics.Vehicle_Dynamics(m=M, l_f=L_F, l_r=L_R, width = WIDTH, length = LENGTH, turning_circle=TURNING_CIRCLE,
                                                C_d = C_D, A_f = A_F, C_roll = C_ROLL, dt = DT)

    ox, oy = [], []

    """
    Initialization
        - Initial State
        - Goal State

        - MPC init
            -- Horizon
            -- Constraints
            -- Objective function Weights Matrix

    """

    # ==========================================
    # ========== Initial & Goal State ==========
    # ==========================================

    start = [0.0, 0.0, np.deg2rad(90.0)]
    goal = [-2.0, 30.0, np.deg2rad(90.0)]
    pos_cars = [(0,15), (5,2), (5, 18)]
    

    # start = [17.0, 20.0, np.deg2rad(-90.0)]
    # goal = [5.0, 20.0, np.deg2rad(90.0)]
    # goal = [20.0, 42.0, np.deg2rad(0.0)]
    #goal = [55.0, 50.0, np.deg2rad(0.0)]

    reeds_sheep.plot_arrow(start[0], start[1], start[2], fc='g')
    reeds_sheep.plot_arrow(goal[0], goal[1], goal[2])

    # Initial state
    # States  : [x; y; v; yaw]
    # Actions : [steer; accel]
    x = np.array([[start[0]],
                  [start[1]],
                  [5.0],
                  [start[2]]])    #  [X; Y; V; Yaw]
    u = np.array([[0*math.pi/180],
                  [0.01]])               #  [steer; accel]

    x_vec = np.squeeze(x, axis=1) # (N,) shape for QP solver, NOT (N,1).

    nx = x.shape[0]
    nu = u.shape[0]

    # =============================================
    # ========== MPC initialization ===============
    # =============================================

    # Prediction horizon
    N = 100

    # Initialize predictive states
    pred_x = np.zeros((nx, N+1))
    for i in range(N+1):
        pred_x[:,i] = x.T

    pred_u = np.zeros((nu, N))
    for i in range(N):
        pred_u[:,i] = u.T

    # MPC Constraints
    umin = np.array([-np.deg2rad(15), -3.]) # u : [steer, accel]
    umax = np.array([ np.deg2rad(15),  2.])
    xmin = np.array([-np.inf,-np.inf, -100., -2*np.pi]) #  [X; Y; vel_x; Yaw]
    xmax = np.array([ np.inf, np.inf,  100.,  2*np.pi])

    # MPC Objective function
    # MPC weight matrix
    Q = sparse.diags([1.0, 1.0, 5.0, 5.0])         # weight matrix for state
    # QN = Q
    QN = sparse.diags([100.0, 100.0, 50.0, 100.0])   # weight matrix for terminal state
    R = sparse.diags([0.01, 0.1])                      # weight matrix for control input
    # R_before = 10*sparse.eye(nu)                    # weight matrix for control input

    # ========================================
    # ============== Simulation ==============
    # ========================================

    # Simulation Setup
    sim_time = 1000
    plt_tic = np.linspace(0, sim_time, sim_time)
    plt_states = np.zeros((nx, sim_time))
    plt_actions = np.zeros((nu, sim_time))

    for i in range(sim_time):
        print("===================== sim_time :", i, "=====================")

        tic = time.time()
        
        # ========================================
        # =========== Obstacle map ===============
        # ========================================
        tic = time.time()

        ox, oy = obs_map.map_road(ox, oy, pos_cars)
        # ox, oy = obs_map.map_maze(ox, oy)

        toc = time.time()
        print("Process time (Obstacle map):", (toc - tic))

        # ================================================
        # ======== Motion Planning (Hybrid A*) ===========
        # ================================================
        tic = time.time()

        curr_pos = [x_vec[0], x_vec[1], x_vec[3]]  # list for hybrid A*
        goal_pos = goal

        if i == 0:
            path = hybrid_a_star.hybrid_a_star_planning(
                curr_pos, goal_pos, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        
        # if i == 0:
        #     path_x, path_y = hybrid_a_star.a_star_planning(
        #                 curr_pos, goal_pos, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

        toc = time.time()
        print("Process time (Hybrid A*):", (toc - tic))

        # ========================================
        # ========= Constraint Search ============
        # ========================================
        tic = time.time()

        # lb_x, ub_x, lb_y, ub_y = hybrid_a_star.constraint_search(ox, oy, path)
        # lb_x, ub_x, lb_y, ub_y = hybrid_a_star.constraint_search_(ox, oy, pred_x)
        # lb_x, ub_x, lb_y, ub_y = hybrid_a_star.constraint_search__(ox, oy, path_x, path_y)

        toc = time.time()
        print("Process time (constraint_search):", (toc - tic))

        # ========================================
        # ========== Control (MPC) ===============
        # ========================================
        tic = time.time()

        # Discrete time model of the vehicle lateral dynamics

        # Reference states
        # Xr, _ = mpc_path_tracking.reference_search(path_x, path_y, pred_x, DT, N)
        Xr, _ = mpc_path_tracking.reference_search_(path.xlist, path.ylist, path.yawlist, pred_x, DT, N)

        # Discrete time model of the vehicle lateral dynamics
        Ad_mat, Bd_mat, gd_mat = vehicle.get_kinematics_model(x, u)

        # ========== Constraints ==========
        umin = np.array([-np.deg2rad(15), -3.]) # u : [steer, accel]
        umax = np.array([ np.deg2rad(15),  1.])
        xmin = np.array([-np.inf,-np.inf, -100., -2*np.pi]) #  [X; Y; vel_x; Yaw]
        xmax = np.array([ np.inf, np.inf,  100.,  2*np.pi])

        # # ========== Objective function ==========
        # # MPC weight matrix
        # Q = sparse.diags([1.0, 1.0, 5.0, 10.0])         # weight matrix for state
        # # QN = Q
        # QN = sparse.diags([10.0, 10.0, 50.0, 50.0])   # weight matrix for terminal state
        # R = sparse.diags([0.1, 0.1])                      # weight matrix for control input
        # # R_before = 10*sparse.eye(nu)                    # weight matrix for control input

        # Solve MPC
        res = mpc_path_tracking.mpc(Ad_mat, Bd_mat, gd_mat, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax)
        # res = mpc_path_tracking.mpc_(Ad_mat, Bd_mat, gd_mat, x_vec, Xr, Q, QN, R, N, lb_x, ub_x, lb_y, ub_y, umin, umax)

        print("solution info")
        print(res.info.obj_val)

        # Check solver status
        if res.info.status != 'solved':
            print('OSQP did not solve the problem!')
            # raise ValueError('OSQP did not solve the problem!')
            plt.pause(5.0)
            continue

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]

        toc = time.time()
        print("Process time (MPC):", (toc - tic))


        # for Plotting predictive states
        sol_state = res.x[:-N*nu]
        sol_control = res.x[-N*nu:]

        for ii in range((N+1)*nx):
            if ii % 4 == 0:
                pred_x[0,ii//4] = sol_state[ii]
            elif ii % 4 == 1:
                pred_x[1,ii//4] = sol_state[ii]
            elif ii % 4 == 2:
                pred_x[2,ii//4] = sol_state[ii]
            else: # ii % 4 == 3:
                # Normalize angle
                temp_yaw = sol_state[ii]
                while temp_yaw > np.pi:
                    temp_yaw -= 2*np.pi
                while temp_yaw < -np.pi:
                    temp_yaw += 2*np.pi
                pred_x[3,ii//4] = temp_yaw

        for jj in range((N)*nu):
            if jj % 2 == 0:
                pred_u[0,jj//2] = sol_control[jj]
            elif jj % 2 == 1:
                pred_u[1,jj//2] = sol_control[jj]
            
        print("pred_u")
        print(pred_u[0,:])

        # ==================================================
        # ========= Simulate Vehicle & Plotting ============
        # ==================================================

        # Plot
        print("Current   x :", x[0], "y :", x[1], "v :", x[2], "yaw :", x[3])
        print("------------------------------------------------------------")
        # print("Reference x :", xr[0], "y :", xr[1], "v :", xr[2], "yaw :", xr[3])
        print("Reference x :", Xr[0,0], "y :", Xr[1,0], "v :", Xr[2,0], "yaw :", Xr[3,0])
        print("------------------------------------------------------------")
        print("steer :", u[0], "accel :", u[1])

        plt.cla()
        plt.cla()
        plt.grid(True)
        plt.axis("equal")

        plt.plot(ox, oy, ".k")                               # plotting Obstacle map
        plt.plot(plt_states[0,:i], plt_states[1,:i], "-b", label="Drived") # plotting driven path
        plot_car(x[0], x[1], x[3], steer=u[0])               # plotting Vehicle w.r.t. rear axle.
        plt.plot(pred_x[0,:], pred_x[1,:], "r")              # Predictive Trajectory
        plt.plot(path.xlist, path.ylist, label="Local_Path") # Path from Hybrid A*
        # plt.plot(path_x, path_y, label="Local_Path") # Path from A*
        plt.plot(Xr[0,:], Xr[1,:], "g")                      # Local Path for MPC
        plt.pause(0.0001)
        
        # Update States
        u = np.expand_dims(ctrl, axis=1) # from (N,) to (N,1)
        x_next = np.matmul(Ad_mat, x) + np.matmul(Bd_mat, u) + gd_mat

        # Normalize angle
        while x_next[3] > np.pi:
            x_next[3] -= 2*np.pi
        while x_next[3] < -np.pi:
            x_next[3] += 2*np.pi

        plt_states[:,i] = x.T
        plt_actions[:,i] = u.T

        x = x_next
        x_vec = np.squeeze(x, axis=1) # (N,) shape for QP solver, NOT (N,1).


        # if check_goal(x, xr, goal_dist=1, stop_speed=5):
        #     print("Goal")
        #     break




if __name__ == '__main__':
    main()