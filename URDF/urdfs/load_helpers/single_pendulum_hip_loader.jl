using RigidBodyDynamics
using MechanismGeometries
import Base.Filesystem: mktempdir

function load_urdf_with_dynamic_paths(urdf_relative_path::String, base_mesh_directory::String)
    # Calculate the absolute path of the URDF file
    current_working_directory = pwd()
    urdf_absolute_path = joinpath(current_working_directory, urdf_relative_path)
    
    # Read the URDF content
    urdf_content = read(urdf_absolute_path, String)
    
    # Replace placeholder or relative path with the actual mesh directory path
    # Adjust this line based on your actual path placeholder or structure
    urdf_content_modified = replace(urdf_content, "package://" => base_mesh_directory)
    
    # Write the modified URDF content to a temporary file
    temp_dir = mktempdir()
    temp_urdf_path = joinpath(temp_dir, "temp_robot.urdf")
    write(temp_urdf_path, urdf_content_modified)
    
    # Load the URDF file from the temporary path
    mechanism = parse_urdf(Float64, temp_urdf_path)
    
    # Clean up the temporary directory and file if necessary
    # rm(temp_dir; recursive=true)
    
    return mechanism
end

# Example usage
base_mesh_directory = joinpath(pwd(), "utils", "urdfs", "single_pendulum")
urdf_relative_path = joinpath("utils", "urdfs", "single_pendulum", "robot.urdf")
mechanism = load_urdf_with_dynamic_paths(urdf_relative_path, base_mesh_directory)

# Now your `mechanism` is loaded with dynamically adjusted paths for mesh files
