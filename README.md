## Pick-and-Place + Mobile Manipulation of a KUKA youBot
**Author: Jared Berry**

This project was associated with Northwestern University ME 449: Robotic Manipulation (Fall 2024).

## Introduction
#### Objective
The goal of this project was to perform pick-and-place mobile manipulation tasks with a KUKA youBot, starting
from an initial condition with an arbitrary error relative to a desired trajectory.

#### Software Format
- main.py<br>
This file allows the user to run the code. It contains the simulation
loop, and functions to load the robot kinematics and system parameters. It also
has helper functions to calculate the manipulator jacobian and plot the errors.

- trajectory_generator.py<br>
This file contains the function to generate the reference trajectory. The time for each
trajectory segment is calculated dynamically using the distance of the trajectory and a 
joint speed parameter.

- feedback_control.py<br>
This file contains the feedforward + PI feedback control function. It also contains the function TestJointLimits, 
which tests for self-collisions and singularities, and modifies the manipulator Jacobian accordingly.

- simulator.py<br>
This file contains the NextState function, which simulates a single timestep of the robot's movement.

#### Process Outline
1. Create a reference trajectory based on the initial and goal configurations of a target object.
2. Use forward kinematics to get the current configuration of the end-effector for each reference frame.
3. Use feedforward + PI control to determine the current error from the expected reference frame.
4. Plan a corrective end-effector twist based on this error and the next frame in the trajectory.
5. Calculate the manipulator Jacobian and the necessary wheel and joint speeds to reach the next reference frame.
6. Integrate wheel and joint speeds to reach the desired reference frame.
7. Apply joint limits to prevent singularities and self-collisions.
8. Repeat steps 2-7 for entire reference trajectory.

## Results
### Feedforward + PI Control with minimal overshoot
The goal for this stage was to design an optimized controller that enables the youBot to perform the 
pick-and-place task with minimal overshoot. This controller was tested with multiple pick-and-place
tasks. Generally, the transient error converged quickly, and the youBot was able to perform all
tasks successfully. The figure below is the result from one pick-and-place task, demonstrating the convergence
of the transient error after an incorrect initial configuration.

![best_errors_fig.png](Figures/best_errors_fig.png)

Here is a video demonstration.

![best_demo.gif](Figures/best_demo.gif)


### Feedforward + PI Control with overshoot
The task for this stage was to design an optimized controller that allows the youBot to perform the 
pick-and-place task with minimal overshoot. This allowed me to prove the versatility of my controller,
and demonstrate my depth of knowledge. The figure below demonstrates an initial overshoot, followed by
a rapid convergence. 

![overshoot_errors_fig.png](Figures/overshoot_errors_fig.png)

Here is a video demonstration.

![overshoot_demo.gif](Figures/overshoot_demo.gif)


