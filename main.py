import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import csv
import subprocess
import sys
import signal
import atexit

filename = "ft_data.csv"
ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
num_joints = 7
fieldnames = [f"{name}_{j}" for j in range(1, num_joints+1) for name in ft_names]

class InteractPerceive:
    def __init__(self, robot):
        self.robot = robot
        self.joint_indices = [i for i in range(7)]  # Joint indices of the robot
        self.p = p

    def get_joint_states(self):
        joint_states = self.p.getJointStates(self.robot, self.joint_indices)
        return joint_states

def main():    
    # -----------------------------------------------------------------------------------------------------------
    # Initialize plot file & start up plotter, register cleanup
    # -----------------------------------------------------------------------------------------------------------
    ft = {
        f"{name}_{j}": 0
        for j in range(1, num_joints+1)
        for name in ft_names
    }
    initialize_plot_file()
    
    plotter = subprocess.Popen(
        [sys.executable, "plotter.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # 2) define a cleanup that will kill the plotter if it's still alive
    def cleanup(signum=None, frame=None):
        if plotter.poll() is None:
            plotter.terminate()
            plotter.wait()
        # if this was a signal, exit now
        if signum is not None:
            sys.exit(0)

    # 3) register for Ctrl‑C and Ctrl‑Break
    signal.signal(signal.SIGINT, cleanup)   # Ctrl‑C
    signal.signal(signal.SIGBREAK, cleanup) # Ctrl‑Break

    # 4) register for normal program exit
    atexit.register(cleanup)
    
    # -----------------------------------------------------------------------------------------------------------
    # Load the simulation and robot
    # -----------------------------------------------------------------------------------------------------------
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                 cameraYaw=50,
                                 cameraPitch=-30,
                                 cameraTargetPosition=[0.5, 0, 0.2])

    # ground plane
    p.loadURDF("plane.urdf")

    robot = p.loadURDF("panda_with_sensor.urdf",
                       basePosition=[0, 0, 0],
                       baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                       useFixedBase=True)

    # create a red box at (0.5, 0, 0.05)
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05]*3)
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05]*3,
                                  rgbaColor=[1, 0, 0, 1])
    p.createMultiBody(baseMass=1,
                      baseCollisionShapeIndex=box_col,
                      baseVisualShapeIndex=box_vis,
                      basePosition=[0.5, 0, 0.05])

    # -----------------------------------------------------------------------------------------------------------
    # Indices on Franka Panda that we care about
    # -----------------------------------------------------------------------------------------------------------
    wrist_joint    = 7     # joint before the gripper
    finger_joints  = [9, 10]
    ee_link_index  = 11    # Panda “tool” link for IK

    # fixed “downward” orientation so the gripper faces straight down
    down_ori = p.getQuaternionFromEuler([math.pi, 0, 0])

    # -----------------------------------------------------------------------------------------------------------
    # Initial position
    # -----------------------------------------------------------------------------------------------------------
    above_pos = [0.5, 0, 0.3]
    ik_vals = p.calculateInverseKinematics(robot,
                                           ee_link_index,
                                           above_pos,
                                           down_ori,
                                           maxNumIterations=1000,
                                           residualThreshold=1e-5)
    # snap to that pose before we start moving
    for i in range(7):
        p.resetJointState(robot, i, ik_vals[i])
        p.enableJointForceTorqueSensor(robot, i, 1)
    time.sleep(1)  # let things settle

    print("Lowering into the cube…")

    # ———————————
    # 4) LOWER INTO THE CUBE
    # ———————————
    for z in [0.15, 0.1, 0.05]:  # descending heights
        target = [0.5, 0, z]
        ik_vals = p.calculateInverseKinematics(robot,
                                               ee_link_index,
                                               target,
                                               down_ori)
        # command the arm joints
        for i in range(7):
            p.setJointMotorControl2(robot, i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=ik_vals[i],
                                    force=200)
        # keep gripper half‑closed
        for j in finger_joints:
            p.setJointMotorControl2(robot, j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.04,
                                    force=50)
        time.sleep(2)  # give time for collision

    print("Contact made! Wrist readings follow:")
    
    # -----------------------------------------------------------------------------------------------------------
    # Stream force readings to csv file for plotting
    # -----------------------------------------------------------------------------------------------------------
    try:
        while True:
            for joint in range(0, num_joints):
                _, _, forces, _ = p.getJointState(robot, joint)
                # fx, fy, fz, mx, my, mz = forces
                for name, val in zip(ft_names, forces):
                    ft[f"{name}_{joint + 1}"] = val
            
            write_to_csv(data_dict=ft)
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        plotter.terminate()
        plotter.wait()

# Initialize CSV file with headers
# Overwrites/deletes existing file with the same name
def initialize_plot_file(filename: str = filename, fieldnames: list = fieldnames):
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Write joint index & force/torque readings to CSV file
def write_to_csv(data_dict: dict, filename: str = filename, fieldnames: list = fieldnames):
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data_dict)
        
if __name__ == "__main__":
    main()
