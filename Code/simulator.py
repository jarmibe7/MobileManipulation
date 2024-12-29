# simulator.py
#
# Contain primary functions for the simulation of the youBot
# kinematics
# 
# Author: Jared Berry
# Date: 12/06/2024
#
import numpy as np
import modern_robotics as mr

def NextState(config, speeds, dt, max_speed, H, T_sb):
    """
    Complete a single step of a simple first-order Euler integration scheme
    to compute the configuration of the youBot after a time-step.

    :param config: A 12-vector representing the current configuration of the robot 
                  (3 variables for the chassis configuration, 5 variables for the arm configuration, 
                  and 4 variables for the wheel angles)
    :param speeds: A 9-vector of controls indicating the wheel speeds (u, 4 variables) and
                   the arm joint speeds (thetadot, 5 variables)
    :param dt: A single time-step
    :param max_speed: A positive real value indicating the max angular speed of the arm joints
                      and the wheels
    :param H: H(0) matrix for youBot

    :return: A 12-vector representing the configuration of the robot time dt later
    """
    # Constrain wheel/joint speeds to bounds
    for i, speed in enumerate(speeds):
        if speed > max_speed: speeds[i] = max_speed
        elif speed < -max_speed: speeds[i] = -max_speed

    # Separate out config and joint speeds into corresponding sections
    q = config[0:3]
    theta_arm = config[3:8]
    theta_wheels = config[8:12]
    gripper_state = config[-1]

    thetadot_wheels = speeds[0:4]
    thetadot_arm = speeds[4:]

    # Euler integrate for new arm joint and wheel angles
    theta_arm_next = theta_arm + (thetadot_arm * dt)
    theta_wheels_next = theta_wheels + (thetadot_wheels * dt)

    #
    # Obtain new chasis configuration with odometry
    #
    # 1.) Measure wheel displacements (deltatheta) since last timestep
    deltatheta_wheels = theta_wheels_next - theta_wheels

    # 2.) Find pseudoinverse of H
    H_psi = np.linalg.pinv(H, rcond=1e-4)

    # 3.) Find Vb = H_psi @ thetadot
    Vb = H_psi @ deltatheta_wheels

    # 4.) Convert planar twist Vb to Vb_6
    Vb_6 = np.array([0, 0, Vb[0], Vb[1], Vb[2], 0])

    # 5.) Integrate Vb_6 over dt, T_bk_bk+1 = e^[Vb_6]
    Vb_6_mat = mr.VecTose3(Vb_6)
    T_b_bnext = mr.MatrixExp6(Vb_6_mat)

    # 6.) Express new chasis frame relative to space frame
    T_sb_next = T_sb @ T_b_bnext
    
    # 7.) Extract coordinates q_k+1
    R_next, p_next = mr.TransToRp(T_sb_next)
    trig_funcs = (np.array([1, 0, 0]) @ R_next)[:2]
    # phi_next = np.arctan2(trig_funcs[1], trig_funcs[0])
    phi_next = np.arccos(trig_funcs[0])
    q_next = np.array([phi_next, p_next[0], p_next[1]])

    # Compile back into next conifg array
    config_next = np.hstack([q_next, theta_arm_next, theta_wheels_next, np.array([gripper_state])])

    return config_next