"""
Date : 2019.08.22
Author : Hyunki Seong

Python MPC
    - based on Dynamics model
    - using predictive linearized matrix

TODO (19.08.30)
    - Modify Vehicle model (from front wheel to rear wheel)
    - Check code block of OSQP
"""

import osqp
import numpy as np
import scipy as sp
import scipy.sparse as sparse

import matplotlib.pyplot as plt
import math
import time
import sys
sys.path.append("../../Vehicle_Dynamics/")
try:
    import vehicle_models
    from visualization_vehicle import plot_car
except:
    raise

def nearest_point(path_x, path_y, x, y, look_ind=0):
    min_d = np.inf
    min_ind = -1

    for i in reversed(range(len(path_x))):
        d = np.sqrt( (path_x[i]-x)**2 + (path_y[i]-y)**2 )
        if d < min_d:
            min_d = d
            min_ind = i
    
    min_ind = min_ind + look_ind

    return min_ind

def reference_search(path_x, path_y, pred_state, dt, N):
    """
    Find reference for MPC
        States  : [X; Y; Yaw; V_x; V_y; Yaw_rate]
        Actions : [steer; accel]
    """
    x_ref = np.zeros((6, N+1)) # nx : 6
    u_ref = np.zeros((2, N+1)) # nu : 2

    cumul_d = 0
    # ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0])
    ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0], look_ind=1) # look ahead 1 index.
    path_d = np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

    # Reference points from x0 to xN
    for i in range(N+1):
        # Calculate Reference points
        cumul_d = cumul_d + abs(pred_state[3,i])*dt # vx == x[3]

        if cumul_d < path_d:
            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = np.deg2rad(0)
            x_ref[3, i] = 10.0
            x_ref[4, i] = 0.0
            x_ref[5, i] = np.deg2rad(0)

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

        else:
            # go forward until cumul_d < path_d
            while(cumul_d >= path_d):
                ind = ind + 1
                path_d = path_d + np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = np.deg2rad(0)
            x_ref[3, i] = 10.0
            x_ref[4, i] = 0.0
            x_ref[5, i] = np.deg2rad(0)

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

    return x_ref, u_ref

def reference_search_(path_x, path_y, path_yaw, pred_state, dt, N):
    """
    Find reference for MPC
        States  : [X; Y; Yaw; V_x; V_y; Yaw_rate]
        Actions : [steer; accel]
    """
    x_ref = np.zeros((6, N+1)) # nx : 6
    u_ref = np.zeros((2, N+1)) # nu : 2

    cumul_d = 0
    ind = nearest_point(path_x, path_y, pred_state[0,0], pred_state[1,0], look_ind=1) # look ahead 1 index.
    path_d = np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

    # Reference points from x0 to xN
    for i in range(N+1):
        # Calculate Reference points
        cumul_d = cumul_d + abs(pred_state[3,i])*dt # vx == x[3]

        if cumul_d < path_d:
            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = path_yaw[ind]
            x_ref[3, i] = 10.0
            x_ref[4, i] = 0.0
            x_ref[5, i] = np.deg2rad(0)

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

        else:
            # go forward until cumul_d < path_d
            while(cumul_d >= path_d):
                ind = ind + 1
                path_d = path_d + np.sqrt( (path_x[ind+1]-path_x[ind])**2 + (path_y[ind+1]-path_y[ind])**2 )

            x_ref[0, i] = path_x[ind]
            x_ref[1, i] = path_y[ind]
            x_ref[2, i] = path_yaw[ind]
            x_ref[3, i] = 10.0
            x_ref[4, i] = 0.0
            x_ref[5, i] = np.deg2rad(0)

            u_ref[0, i] = 0.0 # steer operational point should be 0.
            u_ref[1, i] = 0.0

    return x_ref, u_ref

def check_goal(state, goal, goal_dist, stop_speed):
    # check goal
    dx = state[0] - goal[0]
    dy = state[1] - goal[1]
    d = math.sqrt(dx**2 + dy**2)

    if (d <= goal_dist):
        isgoal = True
    else:
        isgoal = False

    if (state[2] <= stop_speed):
        isstop = True
    else:
        isstop = False

    if isgoal and isstop:
        return True

    return False

def mpc(Ad_list, Bd_list, gd_list, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax):
    """
    Initialize Nonlinear Dynamics with Shooting Method

    """
    # ========== Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1)) ==========
    nx = Ad_list[0].shape[0]
    nu = Bd_list[0].shape[1]

    # ----- quadratic objective -----
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)]).tocsc()

    # ----- linear objective -----
    # xr_vec = np.squeeze(xr, axis=1)
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
    #               np.zeros(N*nu)])
    # xr_vec = Xr[:,0]
    # q = np.hstack([np.kron(np.ones(N), -Q.dot(xr_vec)), -QN.dot(xr_vec),
    #               np.zeros(N*nu)])

    q = -Q.dot(Xr[:,0])                                    # index 0
    for ii in range(N-1):
        q = np.hstack([q, -Q.dot(Xr[:,ii+1])])             # index 1 ~ N-1
    q = np.hstack([q, -QN.dot(Xr[:,-1]), np.zeros(N*nu)])  # index N

    # ----- linear dynamics -----
    Ax_Ad = sparse.csc_matrix(Ad_list[0])
    Ax_diag = sparse.kron(sparse.eye(N+1),-sparse.eye(nx))
    Bu_Bd = sparse.csc_matrix(Bd_list[0])
    
    for i in range(N-1):
        Ad = sparse.csc_matrix(Ad_list[i+1])
        Bd = sparse.csc_matrix(Bd_list[i+1])
        Ax_Ad = sparse.block_diag([Ax_Ad, Ad])
        Bu_Bd = sparse.block_diag([Bu_Bd, Bd])

    Ax_Ad_top = sparse.kron(np.ones(N+1), np.zeros((nx,nx)))
    Ax_Ad_side = sparse.kron(np.ones((N,1)), np.zeros((nx,nx)))
    Ax = Ax_diag + sparse.vstack([Ax_Ad_top, sparse.hstack([Ax_Ad, Ax_Ad_side])])
    Bu_Bd_top = sparse.kron(np.ones(N), np.zeros((nx,nu)))
    Bu = sparse.vstack([Bu_Bd_top, Bu_Bd])
    Aeq = sparse.hstack([Ax, Bu])

    leq = -x_vec # later ueq == leq
    for i in range(N):
        gd = np.squeeze(gd_list[i], axis=1) # from (N,1) to (N,)
        leq = np.hstack([leq, -gd])
    # leq = np.hstack([-x_vec, np.zeros(N*nx)])
    ueq = leq

    # Original Code
    # Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad_list[0])
    # Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd_list[0])
    # Aeq = sparse.hstack([Ax, Bu])
    # gd = np.squeeze(gd_list[0], axis=1) # from (N,1) to (N,)
    # leq = np.hstack([-x_vec, np.kron(np.ones(N), -gd)])
    # ueq = leq

    # ----- input and state constraints -----
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    # lineq = []
    # uineq = []

    # for i in range(len(lb_x)):
    #     xmin = [lb_x[i], lb_y[i], -10, -np.pi]
    #     xmax = [ub_x[i], ub_y[i],  10,  np.pi]
    #     lineq = np.hstack([lineq, xmin])
    #     uineq = np.hstack([uineq, xmax])
    # lineq = np.hstack([lineq, np.kron(np.ones(N), umin)])
    # uineq = np.hstack([uineq, np.kron(np.ones(N), umax)])

    # ----- OSQP constraints -----
    A = sparse.vstack([Aeq, Aineq]).tocsc()
    lb = np.hstack([leq, lineq])
    ub = np.hstack([ueq, uineq])

    # ==========Create an OSQP object and Setup workspace ==========
    prob = osqp.OSQP()
    prob.setup(P, q, A, lb, ub, verbose=True, warm_start=False) # verbose: print output.

    # Solve
    res = prob.solve()

    return res

def main():
    # ===== Vehicle parameters =====
    l_f = 1.25
    l_r = 1.40

    dt = 0.05

    m=1300
    width = 1.78
    length = 4.25
    turning_circle=10.4
    C_d = 0.34
    A_f = 2.0
    C_roll = 0.015

    vehicle = vehicle_models.Vehicle_Dynamics(m=m, l_f=l_f, l_r=l_r, width = width, length = length,
                            turning_circle=turning_circle, C_d = C_d, A_f = A_f, C_roll = C_roll, dt = dt)
    # ========== MPC parameters ==========
    N = 10 # Prediction horizon

    # ========== Initialization ==========
    # Path
    path_x = np.linspace(-10, 100, 100/0.5)
    path_y = np.linspace(-10, 100, 100/0.5) * 0.0 - 0.0

    # Initial state
    # States  : [X; Y; Yaw; V_x; V_y; Yaw_rate]
    # Actions : [steer; accel]
    x = np.array([[0.0],
                [0.0],
                [np.deg2rad(0)],
                [20.0],
                [0.0],
                [np.deg2rad(0)]])   # [X; Y; Yaw; V_x; V_y; Yaw_rate]
    u = np.array([[0*math.pi/180],
                [0.01]])            # [steer; accel]

    x0 = x
    u0 = u
    x_vec = np.squeeze(x, axis=1) # (N,) shape for QP solver, NOT (N,1).

    nx = x.shape[0]
    nu = u.shape[0]

    # ========== Initialial guess states and controls ==========
    # u_noise = np.zeros((nu, 1))
    # mu_steer = 0.0
    # sigma_steer = np.deg2rad(1)
    # mu_accel = 0.0
    # sigma_accel = 0.1

    pred_u = np.zeros((nu, N+1))
    for i in range(N):
        # u_noise[0] = np.random.normal(mu_steer, sigma_steer, 1)
        # u_noise[1] = np.random.normal(mu_accel, sigma_accel, 1)
        # pred_u[:,i] = np.transpose(u0 + u_noise)
        pred_u[:,i] = np.transpose(u0)
    pred_u[:,-1] = pred_u[:,-2] # append last pred_u for N+1

    pred_x = np.zeros((nx, N+1))
    pred_x[:,0] = x0.T
    for i in range(0, N):
        x0, _, _ = vehicle.update_dynamics_model(x0, pred_u[:,i]) # get x_k+1 from x_k and u_k-1
        x0[2,:] = vehicle_models.normalize_angle(x0[2,:])
        pred_x[:,i+1] = x0.T

    # ========== Reference state ==========
    Xr, _ = reference_search(path_x, path_y, pred_x, dt, N)

    # ========== Constraints ==========
    umin = np.array([-np.deg2rad(15), -3.]) # u : [steer, accel]
    umax = np.array([ np.deg2rad(15),  1.])
    xmin = np.array([-np.inf,-np.inf, -2*np.pi, -100., -100., -2*np.pi]) #  [X; Y; Yaw; V_x; V_y; Yaw_rate]
    xmax = np.array([ np.inf, np.inf, -2*np.pi,  100.,  100.,  2*np.pi])

    # ========== Objective function ==========
    # MPC weight matrix
    Q = sparse.diags([5.0, 5.0, 1.0, 5.0, 5.0, 1.0])         # weight matrix for state
    QN = sparse.diags([100.0, 100.0, 10.0, 100.0, 100.0, 10.0])   # weight matrix for terminal state
    R = sparse.diags([50, 50])                      # weight matrix for control input

    # ========== Simulation Setup ==========
    sim_time = 1000
    plt_tic = np.linspace(0, sim_time, sim_time)
    plt_states = np.zeros((nx, sim_time))
    plt_actions = np.zeros((nu, sim_time))

    for i in range(sim_time):
        tic = time.time()
        print("===================== sim_time :", i, "=====================")
        
        # Discrete time model of the vehicle lateral dynamics

        # Reference states
        Xr, _ = reference_search(path_x, path_y, pred_x, dt, N)

        # Discrete time model of the vehicle lateral dynamics
        Ad_list, Bd_list, gd_list = [], [], []
        for ii in range(N):
            Ad, Bd, gd = vehicle.get_dynamics_model(pred_x[:,ii], pred_u[:,ii])
            Ad_list.append(Ad)
            Bd_list.append(Bd)
            gd_list.append(gd)

        # ========== Constraints ==========
        umin = np.array([-np.deg2rad(15), -3.]) # u : [steer, accel]
        umax = np.array([ np.deg2rad(15),  1.])
        xmin = np.array([-np.inf,-np.inf, -2*np.pi, -100., -100., -2*np.pi]) #  [X; Y; Yaw; V_x; V_y; Yaw_rate]
        xmax = np.array([ np.inf, np.inf, -2*np.pi,  100.,  100.,  2*np.pi])

        # Solve MPC
        res = mpc(Ad_list, Bd_list, gd_list, x_vec, Xr, Q, QN, R, N, xmin, xmax, umin, umax)

        # Check solver status
        if res.info.status != 'solved':
            print('OSQP did not solve the problem!')
            # raise ValueError('OSQP did not solve the problem!')
            plt.pause(1.0)
            # continue

        # Apply first control input to the plant
        ctrl = res.x[-N*nu:-(N-1)*nu]
        toc = time.time()
        print("ctrl :", ctrl)

        # Predictive States and Actions
        sol_state = res.x[:-N*nu]
        sol_action = res.x[-N*nu:]
        
        for ii in range((N+1)*nx):
            if ii % nx == 0:
                pred_x[0,ii//nx] = sol_state[ii]
            elif ii % nx == 1:
                pred_x[1,ii//nx] = sol_state[ii]
            elif ii % nx == 2:
                pred_x[2,ii//nx] = sol_state[ii]
            elif ii % nx == 3:
                pred_x[3,ii//nx] = sol_state[ii]
            elif ii % nx == 4:
                pred_x[4,ii//nx] = sol_state[ii]
            else: # ii % 6 == 5:
                pred_x[5,ii//nx] = sol_state[ii]

        for jj in range((N)*nu):
            if jj % nu == 0:
                pred_u[0,jj//nu] = sol_action[jj]
            else: # jj % nu == 1
                pred_u[1,jj//nu] = sol_action[jj]
        pred_u[:,-1] = pred_u[:,-2] # append last control

        # Plot
        print("Current   x :", x[0], "y :", x[1], "yaw :", x[2], "vx :", x[3], "vy :", x[4], "yawrate :", x[5])
        print("------------------------------------------------------------")
        print("Reference x :", Xr[0,0], "y :", Xr[1,0], "yaw :", Xr[2,0], "vx :", Xr[3,0], "vy :", Xr[4,0], "yawrate :", Xr[5,0])
        print("------------------------------------------------------------")
        print("steer :", u[0], "accel :", u[1])

        print("Process time :", toc - tic)

        plt.cla()
        plt.plot(plt_states[0,:i], plt_states[1,:i], "-b", label="Drived") # plot from 0 to i
        plt.grid(True)
        plt.axis("equal")
        plot_car(x[0], x[1], x[2], steer=u[0]) # plotting w.r.t. rear axle.
        plt.plot(pred_x[0,:], pred_x[1,:], "r")
        plt.plot(path_x, path_y, label="Path")
        plt.plot(Xr[0,:], Xr[1,:], "g")
        plt.pause(5)
        
        # Update States
        u = np.expand_dims(ctrl, axis=1) # from (N,) to (N,1)
        x_next = np.matmul(Ad_list[0], x) + np.matmul(Bd_list[0], u) + gd_list[0]

        plt_states[:,i] = x.T
        plt_actions[:,i] = u.T

        x = x_next
        x_vec = np.squeeze(x, axis=1) # (N,) shape for QP solver, NOT (N,1).

        # Update Predictive States and Actions
        temp_pred_x = pred_x
        temp_pred_u = pred_u
        # index 0
        pred_x[:,0] = x_next.T                   
        pred_u[:,0] = temp_pred_u[:,1] # before u.T
        # index 1 ~ N-2
        for ii in range(0, N-2):
            pred_x[:,ii+1] = temp_pred_x[:,ii+2]
            pred_u[:,ii+1] = temp_pred_u[:,ii+2]
        # index N-1
        pred_x[:,-2] = temp_pred_x[:,N]
        pred_u[:,-2] = temp_pred_u[:,N-1]

        # index N
        # append last state using last A, B matrix and last pred state
        # append last control with last pred control
        last_state = np.expand_dims(temp_pred_x[:,N], axis=1) # from (N,) to (N,1)
        last_control = np.expand_dims(temp_pred_u[:,N-1], axis=1)
        pred_x[:,-1] = np.transpose(vehicle.update_dynamics_model(last_state, last_control))
        pred_u[:,-1] = pred_u[:,N-1]


        # if check_goal(x, xr, goal_dist=1, stop_speed=5):
        #     print("Goal")
        #     break
    
if __name__ == "__main__":
    main()
