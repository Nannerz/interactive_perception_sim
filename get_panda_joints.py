import pybullet as p
import pybullet_data

# 1) connect and load the Panda URDF
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF("panda_with_sensor.urdf", useFixedBase=True)

# 2) find out how many joints
num_joints = p.getNumJoints(robot_id)
print(f"Panda has {num_joints} joints (including fixed):\n")

# 3) iterate and print name, type, and limits
joint_type_map = {
    p.JOINT_REVOLUTE:   "REVOLUTE",
    p.JOINT_PRISMATIC:  "PRISMATIC",
    p.JOINT_SPHERICAL:  "SPHERICAL",
    p.JOINT_PLANAR:     "PLANAR",
    p.JOINT_FIXED:      "FIXED",
}

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_index = info[0]
    joint_name  = info[1].decode('utf-8')
    joint_type  = joint_type_map[info[2]]
    lower_limit = info[8]
    upper_limit = info[9]
    max_force   = info[10]
    max_vel     = info[11]
    print(f"#{joint_index:2d}: {joint_name:15s} | {joint_type:9s} | "
          f"limits=({lower_limit:.3f}, {upper_limit:.3f}) "
          f"maxForce={max_force:.1f} maxVel={max_vel:.1f}")
