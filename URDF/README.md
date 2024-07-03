# Workflow FreeCAD to URDF

The goal is to build a URDF file of the robot from FreeCAD, where it is originally assembled. A free Onshape account can be obtained with a student email. The current version of the robot on Onshape is available at this [link](https://cad.onshape.com/documents/033ca0e37c14982505ca7efe/w/03be525f447a6c65334b88cb/e/2a239dbb48d9757e23f71310?renderMode=0&uiState=66857c18c2e51236e0b2f171).

## Workflow Steps

1. **Export Components from FreeCAD to Onshape:**
   - Export each part individually from FreeCAD to ensure compatibility.

2. **Reassemble in Onshape:**
   - Reassemble the parts within Onshape to address compatibility issues and prevent bugs that occur with FreeCAD assemblies imported into Onshape.

3. **Define Rotational Joints in Onshape:**
   - Follow the guidelines in the [Onshape-to-Robot documentation](https://onshape-to-robot.readthedocs.io/en/latest/) to define the rotational joints.

4. **Convert Assembly to URDF:**
   - Use the Onshape-to-Robot library in Python to convert the assembly into a URDF file.

For a hands-on tutorial, refer to this [YouTube video](https://www.youtube.com/watch?v=C8oK4uUmbRw&t=979s&pp=ygUQb25zaGFwZSB0byByb2JvdA%3D%3D).

At the end of this process, the robot.urdf file is created with the desired assembly. One last step is to change each "package:///" occurences in the robot.urdf file by your chosen folder name.

## Other Possibilities

1. **FreeCROSS:**
   - [FreeCROSS](https://github.com/galou/freecad.cross) is an option, though currently under limited development. It requires ROS2, which is not easy to download and install, particularly on macOS.

2. **Using SolidWorks:**
   - In the project folder (`project>Models>GlobalSystem`), open the assembly `global_system.SLDASM` to view the entire project with the URDF configuration.
   - To ensure it works:
     - Do not let the plug-in automatically generate the axes and origins of the joints. Create and specify them manually.