# trajectory_generator.py
# 
# Generate trajectories for the arm of a youBot, in order
# to pick up a block.
#
# Author: Jared Berry
# Date: 11/20/2024
#
# Additional References:
#   Max joint speeds of arm: Benjamin Keiser, Torque Control of a KUKA youBot Arm, Master Thesis
#                            Robotics and Perception Group, University of Zurich
#   Angular displacement between 2 transformations: http://www.boris-belousov.net/2016/12/01/quat-dist/
#
import numpy as np
import modern_robotics as mr
mr.ScrewTrajectory

### GIVEN FRAMES ###
# {s}: Space frame
# {b}: Body frame of wheeled portion
# {0}: Base of robot arm
# {e}: End effector frame
# {c}: Cube frame

### TRAJECTORY SEGMENTS ###
# 1.) Move to grasp standoff: T_se_i -> T_se_stand_initial
# 2.) Move to grasp: T_se_stand_initial -> T_se_grasp_initial
# 3.) Grasp: Close end effector while holding rest of arm still
# 4.) Move back to grasp standoff: T_se_grasp_initial -> T_se_stand_initial
# 5.) Move to goal standoff: T_se_stand_intitial -> T_se_stand_final
# 6.) Move to goal cube: T_se_stand_final -> T_se_grasp_final
# 7.) Release block: Open end effector while holding rest of arm still
# 8.) Move back to goal standoff: T_se_grasp_final -> T_se_stand_final

### TRANSFORMS WE NEED ###
# T_se_stand_initial = T_sc_i * T_ce_stand
# T_se_grasp_initial = T_sc_i * T_ce_grasp
# T_se_stand_final = T_sc_f * T_ce_stand
# T_se_grasp_final = T_sc_f * T_ce_grasp

def TrajectoryGenerator(T_se_i, T_sc_i, T_sc_f, T_ce_grasp, T_ce_stand, JB, thetadot_max, k, dt):
    """
    Computes the trajectory of a youBot's robotic arm, in order to pick up a block
    at a specified location.

    :param T_se_i: The initial configuration of the end-effector in the reference trajectory
    :param T_sc_i: The cube's initial configuration
    :param T_sc_f: The cube's desired final configuration
    :param T_ce_grasp: The end-effector's configuration relative to the cube when it is grasping the cube
    :param T_ce_stand: The end-effector's standoff configuration above the cube, before and after grasping
    :param Blist: List of screw axes in the B_frame
    :param thetadot_max: List of max joint speeds for each joint
    :param k: The number of trajectory reference configurations per 0.01 seconds
    :param dt: The time-step

    :return: A representation of the N configurations of the end-effector along the entire concatenated eight
             segment reference trajectory. Each reference point represents a transformation T_se of the end-effector
             frame {e} relative to {s} at an instant in time, plus the gripper state (0 or 1)
    """
    # Calculate all needed transformations/matrices
    T_se_stand_i = T_sc_i @ T_ce_stand
    T_se_grasp_i = T_sc_i @ T_ce_grasp
    T_se_stand_f = T_sc_f @ T_ce_stand
    T_se_grasp_f = T_sc_f @ T_ce_grasp

    # Create dict matching trajectory "tags" to their corresponding starting and ending configurations
    trajectories = {'i_to_stand_i': (T_se_i, T_se_stand_i), 'stand_i_to_grasp_i': (T_se_stand_i, T_se_grasp_i), 
                    'close': (T_se_grasp_i, T_se_grasp_i), 'grasp_i_to_stand_i': (T_se_grasp_i, T_se_stand_i), 
                    'stand_i_to_stand_f': (T_se_stand_i, T_se_stand_f), 'stand_f_to_grasp_f': (T_se_stand_f, T_se_grasp_f), 
                    'open': (T_se_grasp_f, T_se_grasp_f), 'grasp_f_to_stand_f': (T_se_grasp_f, T_se_stand_f)}
    
    trajectories_list = []

    # Gripper starts open
    gripper_state = 0

    # Iterate over the 8 trajectories
    for tag, (start, end) in trajectories.items():
        # print(f"Creating trajectory for {tag}\n")

        # if "tag" signifies gripper opening or closing call generate_gripper_trajectory
        if (tag == 'close') or (tag == 'open'):
            # Switch gripper state
            # If gripper is 0: 1 - 0 = 1
            # If gripper is 1: 1 - 1 = 0
            gripper_state = 1 - gripper_state

            # Number of steps = 0.625 * (k / 0.01) -> rounded up to nearest integer
            N = int(0.625 * (k / dt))

            # Create gripper trajectory
            SE3_trajectory_list = [start] * N
        else:
            # Calculate linear and angular distances to target from initial trajectory staring point
            linear_dist = np.linalg.norm(end[0:3, 3] - start[0:3, 3])
            diff_rotation_matrix = start.T @ end
            angular_dist = np.arccos(np.clip((np.trace(diff_rotation_matrix) - 1) / 2, -1.0, 1.0))

            # Calculate max end effector twist, and extract max end effector linear and angular speeds (pi/2 rad/s)
            V_bmax = JB @ thetadot_max

            # Calculate time required for trajectory based off of max end effector speeds
            linear_time = linear_dist / np.linalg.norm(V_bmax[-3:])
            angular_time = angular_dist / np.linalg.norm(V_bmax[0:3])

            # Calculate time required for trajectory based off of max end effector speeds, and round to
            # nearest multiple of 0.01 (2 decimal places)
            traj_time_unrounded = max(linear_time, angular_time)
            traj_time = round(traj_time_unrounded, 2) + 0.5
            # traj_time = max(traj_time, 1)

            #  Calculate N for each trajectory segment based off of time of segment
            N = int(traj_time * (k / dt))

            # Create quintic screw trajectory
            SE3_trajectory_list = mr.CartesianTrajectory(start, end, traj_time, N, 5)

        # Format SE3 matrices in trajectory_list for CSV file, and add gripper state to end
        trajectory_list = [np.append(matrix[:3, :3].flatten(), np.append(matrix[:3, 3], gripper_state)) for matrix in SE3_trajectory_list]           

        # Concat lists of reference configurations into one array for that trajectory
        trajectory = np.array(trajectory_list)

        # Store trajectory into save list
        trajectories_list.append(trajectory)

    # Concat all trajectories in storage array into one large trajectory
    final_trajectory = np.vstack(trajectories_list)
    
    return final_trajectory

