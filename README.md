# Towards abstraction-based control of a bipedal robot: modelling of a 5-link planar bipedal robot

This GitHub repository contains all the codes and data I've used during my Master Thesis.

## Repository structure

- data/: Contains all datasets used, divided into two folders: one for experiments conducted on the robot and one for simulations.
- analysis/: Contains folders for the analysis conducted in each chapter of this thesis.
- src/: This folder includes scripts for cleaning and preprocessing data, initial methodologies for parameter identification of the Dynamixels, and a folder with Python scripts for recurrent plots and analysis.
- param-id/: Contains the Julia scripts for parameter identification using the optimization-based approach.
- URDF/: Contains all the URDF files obtained with the Onshape to URDF workflow, and a folder with Julia scripts for visualization of the URDFs.
- single_pendulum/: Contains all the Julia scripts used for modeling the single- pendulum configuration of the robot.
- double_pendulum/: Contains the Julia scripts for modeling the double-pendulum system and a Python folder for the construction of the obstacle avoidance problem.
- utils/: Contains various utilities used across several Julia scripts, such as loading datasets, loading URDF files, and defining ModelingToolkit components.

Please note that this repo is the concatenation of multiple repo's I've created. So each independent code may not run without some path redefinition, except the most important ones of course (namely the Julia scripts for parameter identification and model validation).

## Recommendations

To find information about libraries, visit the GitHub and dive into the source code if the documentation is not enough. Can also have a look at https://discourse.julialang.org.
