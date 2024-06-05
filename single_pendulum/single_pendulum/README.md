The single_pendulum folder is structured as follows :

 - model_tanh.jl :
 Implementation of the friction model with tanh as sign() function

 - model_callback.jl :
 Implementation of the friction model with callbacks for sign() function

 - model_callback_2.jl :
 First attempt at implementing the model using ModelingToolkit components. Observation : callbacks doesn't work with components

 - single_pendulum_system.jl :
 The final implementation of the system, the components are defined in model_components.jl

 - single_pendulum_ode.jl :
 Dont know

 - single_pendulum_ode_DE.jl :
 Implementation of the model using DifferentialEquations.jl

 - single_pendulum_ode_simplify.jl :
 Dont know

