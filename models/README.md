# Models

This project builds upon the methodology outlined in the ModelingToolkitStandardLibrary.jl. The model is constructed by developing separate components, which are then assembled to form the full system. These components can be found in the `model_components.jl` file.

## Single-Pendulum Configuration

The focus is on modeling one leg of the robot with the knee fixed as an example. The system for this configuration is developed in `single_pendulum_system.jl`.

Within the `single_pendulum` folder, you'll find other related models. Initially, I developed the model using the `DifferentialEquations.jl` library before transitioning to `ModelingToolkit.jl`. To manage non-linearities, such as friction modeled by the `sign()` function, I used the callback methodology described [here](https://klaff.github.io/friction_state_machine.jl.html).

### Improvements and Future Work

Reflecting on my approach, I now question whether defining separate components was the most efficient strategy. While isolating individual components made it easier to observe their behavior, for production purposes, I would recommend constructing the model as a single function. The mathematical model, when expressed in matrix form, is relatively simple. The main challenge will be transitioning between different configurations when modeling the full robot dynamics. However, two key components are still necessary to measure the evolution of outputs and to linearize around operating points using `ModelingToolkit.jl`.

### Summary

- Model the system using matrix formulations.
- Use components to measure inputs/outputs for linearization around operational points.

## Double-Pendulum Configuration

The double-pendulum system demonstrates how to reuse components to build more complex models. However, as the full robot model grows, it will become increasingly cumbersome to manage numerous components. This reinforces my belief that a matrix-based approach for the entire system would be more practical for modeling the robot's dynamics.
