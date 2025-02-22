# Optimal_Control_of_flexible_surface
Together with Luca Malferrari and Filippo Ugolini, we developed this project as part of the final examination for the Optimal Control course in the Automation Engineering program at the University of Bologna.

The goal of this project is to design an optimal control strategy for an actuated flexible surface,
which models the behavior of adaptive wing structures in aircraft or unmanned aerial vehicles
(UAVs). 

# Project Structure

- dynamics.py : Symbolic and numeric computation of the dynamic.
- cost.py : Defines the structure of the cost function.
- animation.py : Graphic tool to appreciate the motion of the design trajectory on the flexible surface.
- reference_trajectory.py`: Generation of a variety of reference trajectory (step and smooth).
- solver.py : Implementation of LQR and MPC algorithms.
- armijo.py : Computation of the Armijo stepsize.
- main.py : Entry point of the project program.

 NOTE: consideration inside the reference_trajectory.py:
- if 'step_reference' (inside the main) is True the function gen() is used to compute the reference trajectory between two equilibria,  
  otherwise the function gen_swing() is used to compute smooth transition between four equilibria.
- if 'step' is True (and also 'step_reference' is True) the function gen() returns a step reference trajectory between two equilibria, otherwise a smooth       
  trajectory between the same two equilibria is generated. The variable 'step' is inside the argument of the gen() arguments.


For further details, please refer to the attached PDF report.

Best regards,
Enrico Battistini.
