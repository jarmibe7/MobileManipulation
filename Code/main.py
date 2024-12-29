# main.py
# 
# Main function for youBot mobile manipulation.
#
# Author: Jared Berry
# Date: 11/20/2024
#
# To run:
# python Code/main.py
#
# To run each task, change the 'task' parameter in main() to one of the 
# listed options. Make sure to change the saving filepath if you run the code!
#
import numpy as np
import modern_robotics as mr
import matplotlib.pyplot as plt
import os
import logging

from trajectory_generator import TrajectoryGenerator
from simulator import NextState
from feedback_control import FeedbackControl, TestJointLimits

def c(theta): return np.cos(theta)
def s(theta): return np.sin(theta)

#
# load_parameters
#
# Helper function to load system parameters and robot kinematics
#
def load_parameters():
    # Robot kinematics
    T_sb_i = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.0963],
                       [0, 0, 0, 1]])
    T_b0 = np.array([[1, 0, 0, 0.1662],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.0026],
                     [0, 0, 0, 1]])
    M_0e = np.array([[1, 0, 0, 0.033],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0.6546],
                     [0, 0, 0, 1]])
    T_se_i = np.array([[0, 0, 1, 0.5],
                       [0, 1, 0, -0.50],
                       [-1, 0, 0,0.5],
                       [0, 0, 0, 1]])
    B_mat = np.array([[0, 0, 1, 0, 0.033, 0],
                      [0, -1, 0, -0.5076, 0, 0],
                      [0, -1, 0, -0.3526, 0, 0],
                      [0, -1, 0, -0.2176, 0, 0],
                      [0, 0, 1, 0, 0, 0]]).T
    mjs = np.pi/10
    thetadot_max = np.array([mjs, mjs, mjs, mjs, mjs])

    # Robot H(0) Matrix
    r = 0.0475  # Wheel radius
    l = 0.47/2
    w = 0.3/2
    H = (1/r) * np.array([[-l - w, 1, -1],
                          [ l + w, 1,  1],
                          [ l + w, 1, -1],
                          [-l - w, 1,  1]])

    eta = 3* np.pi / 4 # Angle of end effector w.r.t cube
    T_ce_grasp = np.array([[c(eta), 0, s(eta), 0.025],
                           [0, 1, 0, 0],
                           [-s(eta), 0, c(eta), 0.031],
                           [0, 0, 0, 1]])
    T_ce_stand = np.array([[c(eta), 0, s(eta), 0.025],
                           [0, 1, 0, 0],
                           [-s(eta), 0, c(eta), 0.25],
                           [0, 0, 0, 1]])
    
    return T_b0, M_0e, T_se_i, B_mat, thetadot_max, H, T_ce_grasp, T_ce_stand

#
# plot_errors
#
# Helper function to plot error twists from each timestep
#
def plot_errors(errors, save_folder, task_name):
    # Separate angular and linear error
    angular_errors = errors[:, 0:3] 
    linear_errors = errors[:, 3:]   

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle('Error Components Over Time')

    # Plot angular error components
    axes = ['X', 'Y', 'Z']
    for i, axis in enumerate(axes):
        axs[0].plot(angular_errors[:, i], label=f'Angular Error {axis}')
    axs[0].set_ylabel('Angular Error (rad)')
    axs[0].set_xlabel('Timestep (dt=10ms)')
    axs[0].legend(loc='lower right')
    axs[0].grid(True)

    # Plot linear error components
    for i, axis in enumerate(axes):
        axs[1].plot(linear_errors[:, i], label=f'Linear Error {axis}')
    axs[1].set_ylabel('Linear Error (m)')
    axs[1].set_xlabel('Timestep (dt=10ms)')
    axs[1].legend(loc='lower right')
    axs[1].grid(True)

    # Finalize and save plots
    plt.tight_layout()
    filepath = os.path.join(save_folder, f'{task_name}_errors_fig.png')
    plt.savefig(filepath)
    logging.info(f'Exported errors figure to {filepath}\n')

    return

#
# calculate_jacobian
#
# Helper function to calculate manipulator jacobian
#
def calculate_jacobian(T_0e, T_b0, B_mat, H, thetalist):
    # Calculate arm Jacobian
    J_arm = mr.JacobianBody(B_mat, thetalist)

    # Calculate F6 (m = 4)
    H_psi = np.linalg.pinv(H, rcond=1e-4)
    zero_vec = np.zeros(4)
    F6 = np.array([zero_vec, zero_vec, H_psi[0], H_psi[1], H_psi[2], zero_vec])

    # Calculate base jacobian
    T_eb = mr.TransInv(T_0e) @ mr.TransInv(T_b0)
    J_base = mr.Adjoint(T_eb) @ F6

    # Compute full manipulator jacobian
    Je = np.hstack([J_base, J_arm])

    return Je

def run_simulation(task, T_sc_i, T_sc_f, iconfig, save_folder):
    # Load robot parameters
    T_b0, M_0e, T_se_i, B_mat, thetadot_max, H, T_ce_grasp, T_ce_stand = load_parameters()

    # Initial conditions
    dt = 0.01
    iphi, ix, iy = iconfig[0], iconfig[1], iconfig[2]
    T_sb = np.array([[c(iphi), -s(iphi), 0,   ix],
                        [s(iphi),  c(iphi), 0,iy],
                        [0,   0,       1, 0.0963],
                        [0,   0,       0,      1]])

    # Generate end effector trajectory
    logging.info('Generating Trajectories...\n')
    k = 1
    trajectory_array = TrajectoryGenerator(T_se_i, T_sc_i, T_sc_f, T_ce_grasp, T_ce_stand, B_mat, thetadot_max, k, dt)

    # # Export milestone 2 result to CSV
    # filepath_m2 = fr'C:\Users\jarmi\ME_449\Final_Project\milestone_two.csv'
    # np.savetxt(filepath_m2, trajectory_array, delimiter=',')
    # print(f'Exported CSV to {filepath_m2}\n')

    # Feedback controller gains (proportional and integral)
    kps = {'best':0.9 * np.eye(6), 'overshoot':2.2 * np.eye(6), 'newTask':0.9 * np.eye(6), 'test':0.9 * np.eye(6)}
    kis = {'best':0.01 * np.eye(6), 'overshoot':0.02 * np.eye(6), 'newTask':0.01 * np.eye(6), 'test':100.9 * np.eye(6) }
    kp = kps[task]
    ki = kis[task]
    X_etotal = np.zeros(6)

    # Simulation loop
    max_speed = 15
    config_list = [iconfig]
    errors_list = []
    config = iconfig
    previous_gripper_state = 0  # DEBUG
    logging.info('Simulating...\n')
    for i,traj in enumerate(trajectory_array):
        # Set gripper state
        config[-1] = traj[-1]

        # Get current and next desired configurations
        Rd, pd = np.array([traj[0:3], traj[3:6], traj[6:9]]), traj[9:12]
        Xd = mr.RpToTrans(Rd, pd)

        if i + 1 >= len(trajectory_array): break  # Done if no traj left
        else: traj_next = trajectory_array[i+1]
        Rd_next, pd_next = np.array([traj_next[0:3], traj_next[3:6], traj_next[6:9]]), traj_next[9:12]
        Xd_next = mr.RpToTrans(Rd_next, pd_next)

        # Get config of EE in relation to arm base with FK
        thetalist = config[3:8]
        T_0e = mr.FKinBody(M_0e, B_mat, thetalist)

        # Get commanded end effector twist from FeedbackControl
        phi, x, y = config[0], config[1], config[2]
        T_sb = np.array([[c(phi), -s(phi), 0,   x],
                            [s(phi),  c(phi), 0,   y],
                            [0,   0,       1, 0.0963],
                            [0,   0,       0,      1]])
        X = T_sb @ T_b0 @ T_0e

        # Test values
        # Xd = np.array([[0, 0, 1, 0.5], [0, 1, 0, 0], [-1, 0, 0, 0.5], [0, 0, 0, 1]])
        # Xd_next = np.array([[0, 0, 1, 0.6], [0, 1, 0, 0], [-1, 0, 0, 0.3], [0, 0, 0, 1]])

        # Get Ve and total error from FeedbackControl
        Ve, X_e, X_etotal = FeedbackControl(X, Xd, Xd_next, X_etotal, kp, ki, dt)
        errors_list.append(X_e.copy())

        # Calculate Jacobian pseudoinverse
        Je = calculate_jacobian(T_0e, T_b0, B_mat, H, thetalist)
        Je_psi = np.linalg.pinv(Je)

        # Convert commanded twist into vector of commanded joint speeds
        speeds = Je_psi @ Ve

        # Simulate next frame of trajectory
        config = NextState(config, speeds, dt, max_speed, H, T_sb)

        # Check if joint limits have been violated
        violation, Je_mod = TestJointLimits(config, Je)
        if violation:
            Je_psi_mod = np.linalg.pinv(Je_mod)
            speeds_mod = Je_psi_mod @ Ve
            config = NextState(config, speeds_mod, dt, max_speed, H, T_sb)

        config_list.append(config)

    # Compile motion and export to CSV
    final_motion = np.vstack(config_list)
    filepath_final = os.path.join(save_folder, f'{task}.csv')
    np.savetxt(filepath_final, final_motion, delimiter=',')
    logging.info(f'Exported CSV to {filepath_final}')

    # Create error figure
    errors = np.vstack(errors_list)
    filepath_errors = os.path.join(save_folder, f'{task}_errors.csv')
    np.savetxt(filepath_errors, errors, delimiter=',')
    logging.info(f'Exported errors CSV to {filepath_final}')

    plot_errors(errors, save_folder, task)

    return

def main():
    # Select which task to run
    task = 'test'   # Options: best | overshoot | newTask
    save_folder = fr'C:\Users\jarmi\ME_449\Final_Project\results'   # Replace saving filepath here
    save_folder = os.path.join(save_folder, task)
    try: 
        os.makedirs(save_folder, exist_ok=True)
    except FileExistsError:
        pass

    log_name = os.path.join(save_folder, f'{task}_log.txt')

    # Configure logging
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


    logging.info('>> python code/main.py')
    logging.info('STARTING\n')

    match task:
        case 'best' | 'test':
            # Object location/frames
            T_sc_i = np.array([[1, 0, 0, 1],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])
            T_sc_f = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, -1],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])
            iconfig = [1, 0, 0.2, 0, 0, -0.6, 0, 0,  # phi, x, y, th1, th2, th3, th4, th5
                       0, 0, 0, 0, 0]                # wh1, wh2, wh3, wh4, gripper_state
        case 'overshoot':
            # Object location/frames
            T_sc_i = np.array([[1, 0, 0, 1],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])
            T_sc_f = np.array([[0, 1, 0, 0],
                               [-1, 0, 0, -1],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])
            iconfig = [1, 0, 0.2, 0, 0, -0.6, 0, 0,  # phi, x, y, th1, th2, th3, th4, th5
                       0, 0, 0, 0, 0]                # wh1, wh2, wh3, wh4, gripper_state
        case 'newTask':
            # Object location/frames
            T_sc_i = np.array([[1, 0, 0, 2],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])
            T_sc_f = np.array([[1, 0, 0, 2],
                               [0, 1, 0, -1],
                               [0, 0, 1, 0.025],
                               [0, 0, 0, 1]])
            iconfig = [0.4, 0.75, -0.5, 0, 0, -0.8, 0, 0.6,  # phi, x, y, th1, th2, th3, th4, th5
                       0, 0, 0, 0, 0]                # wh1, wh2, wh3, wh4, gripper_state
            
    run_simulation(task, T_sc_i, T_sc_f, iconfig, save_folder)

    logging.info('DONE')
    return

if __name__ == '__main__':
    main()