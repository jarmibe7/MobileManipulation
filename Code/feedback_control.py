# feedback_control.py
#
# This file contains a feedback control function for the mobile manipulation of a youBot.
#
# Author: Jared Berry
# Date: 12/09/2024
#
import numpy as np
import modern_robotics as mr

def FeedbackControl(X, Xd, Xd_next, X_etotal, kp, ki, dt):
    """
    This function calculates the kinematic task-space feedforward + feedback control law for
    a youBot.

    :param X: The current actual configuration of the end effector X (T_se)
    :param Xd: The current end-effector reference configuration (T_se,d)
    :param Xd_next: The end-effector reference configuration at the next timestep 
                    in the reference trajectory (T_se,d,next), at a time dt later
    :param X_etotal: The total compiled error for the integral term
    :param kp: The proportional gain matrix
    :param ki: The integral gain matrix
    :param dt: The timestep between reference trajectories

    :return: The commanded end-effector twist in the end-effector frame, and single step + total errors
    """

    #
    # Compute feedforward component
    #
    # Compute matrix adjoint of X_bd = X_inv @ Xd
    X_bd_Ad = mr.Adjoint(mr.TransInv(X) @ Xd)

    # Compute Vd -> [Vd] = (1/dt) * log(X_inv @ Xd)
    # The transformation from X_inv to Xd is the transformation in end-effector frame in
    # timestep dt. Dividing by dt scales it to be in unit time, converting it to a twist.
    Vd_mat = (1/dt) * mr.MatrixLog6(mr.TransInv(Xd) @ Xd_next)
    Vd = mr.se3ToVec(Vd_mat)
    feed_forward_term = X_bd_Ad @ Vd

    #
    # Compute feedback component (proportional)
    #
    # Compute X_e -> [X_e] = log(X_bd) and multiply by proportional gain
    # X_e is an error twist from the actual to desired frame at the current timestep
    X_e = mr.se3ToVec(mr.MatrixLog6(mr.TransInv(X) @ Xd))
    p_term = kp @ X_e

    # Compute feedback component (integral)
    # Euler integrate X_e over dt and add to X_etotal and multiply by integral gain
    X_etotal += X_e * dt
    i_term = ki @ X_etotal

    # Sum controller terms to determine final commanded twist
    Ve = feed_forward_term + p_term + i_term

    return Ve, X_e, X_etotal

def TestJointLimits(config, Je):
    """
    Keep the arm of a youBot from self-collision and singularities during mobile manipulation.

    :param config: Next planned configuration of the youBot
    :param Je: The manipulator jacobian for the next planned motion

    :return violation: True if the joint limits were violated
    :return Je: The updated manipulator jacobian that prevents joint limit violation.
    """
    Je_copy = Je.copy()
    thetalist = config[3:8]
    violation = False

    # Joint 1
    if abs(thetalist[0]) > 1.9: 
        violation = True
        Je_copy[:,5] = np.zeros(6)

    # Joint 2
    if thetalist[1] > 0.5 or thetalist[1] < -1.4: 
        violation = True
        Je_copy[:,5] = np.zeros(6)

    # Joint 3
    if (thetalist[1] > 1.1 and thetalist[2] > 1.5) or \
        (thetalist[1] < -1.05 and thetalist[2] < -1.6) or \
        thetalist[2] < -2.0 or abs(thetalist[2] > 1.8):
        violation = True
        Je_copy[:,6] = np.zeros(6)

    # Joint 4
    if (thetalist[1] < -1.05 and thetalist[2] < -1.5 and thetalist[3] < -0.1) or \
       (abs(thetalist[3]) > 1.78):
        violation = True
        Je_copy[:,7] = np.zeros(6)
    
    return violation, Je_copy
