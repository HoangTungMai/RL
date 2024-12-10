import pybullet as p
# Add a marker for the world frame
def add_world_frame():
    length = 0.1  # Length of the axis
    radius = 0.01  # Radius of the lines

    print("Frame")
    print("X-axis: red")
    print("Y-axis: green")
    print("Z-axis: blue")

    # X-axis (red)
    p.addUserDebugLine([0, 0, 0], [length, 0, 0], [1, 0, 0], lineWidth=2)

    # Y-axis (green)
    p.addUserDebugLine([0, 0, 0], [0, length, 0], [0, 1, 0], lineWidth=2)

    # Z-axis (blue)
    p.addUserDebugLine([0, 0, 0], [0, 0, length], [0, 0, 1], lineWidth=2)
