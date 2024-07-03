using RigidBodyDynamics

function get_mechanism(; wanted_mech = "double_pendulum")
    # Load the URDF file
    if wanted_mech == "double_pendulum"
        urdf_path = joinpath("double_pendulum_right", "robot.urdf")
    elseif wanted_mech == "single_pendulum_knee"
        urdf_path = joinpath("single_pendulum_knee", "robot.urdf")
    else
        urdf_path = joinpath("single_pendulum_hip", "robot.urdf")
    end
    mechanism = parse_urdf(Float64, urdf_path)
    remove_fixed_tree_joints!(mechanism)

    return mechanism
end
