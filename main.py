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
import os
import json
import threading
from collections import deque
import pandas as pd
from plotter import Plotter
from position_gui import Position_GUI, Position_GUI_Thread

# -----------------------------------------------------------------------------------------------------------
class InteractPerceive:
    def __init__(self) -> None:
        super().__init__()
        self.robot = None
        
        self.wrist_joint = 7 # joint before the gripper
        self.finger_joints = [9, 10]
        self.ee_link_index = 11 # Panda “tool” link for IK
        self.num_joints = 7
        
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.ft_types = [f"{name}_{j}" for j in range(1, self.num_joints+1) for name in self.ft_names]
        # self.maxlen = 1000
        self.ft = {
            # f"{name}_{j}": deque(maxlen=self.maxlen)
            f"{name}_{j}": 0
            for j in range(1, self.num_joints+1)
            for name in self.ft_names
        }
        
        self.data_path = os.path.dirname(os.path.abspath(__file__))
        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")

        
    def init_sim(self) -> None:
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                          cameraYaw=50,
                                          cameraPitch=-30,
                                          cameraTargetPosition=[0.5, 0, 0.2])
        p.loadURDF("plane.urdf") # ground plane

        urdf_dir = os.path.join(self.data_path, "panda_with_sensor.urdf")
        print(f"Loading URDF from: {urdf_dir}")
        self.robot = p.loadURDF(urdf_dir,
                                basePosition=[0, 0, 0],
                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)
        for i in range(7):
            p.enableJointForceTorqueSensor(self.robot, i, 1)
        self.create_object()
        
    def create_object(self) -> None:
        # create a red box at (0.5, 0, 0.05)
        box_half_extents = [0.05]*3
        box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_half_extents)
        box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=box_half_extents,
                                            rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=box_col,
                                baseVisualShapeIndex=box_vis,
                                basePosition=[0.5, 0, 0.05])

    # Initialize CSV file with headers, overwrites/deletes existing file with the same name
    def initialize_plot_file(self) -> None:
        with open(self.ft_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ft_types)
            writer.writeheader()

    # Get force/torque readings from each joint
    def get_forces(self) -> dict:
        for joint in range(0, self.num_joints):
            _, _, forces, _ = p.getJointState(self.robot, joint)
            # fx, fy, fz, mx, my, mz = forces
            for name, val in zip(self.ft_names, forces):
                self.ft[f"{name}_{joint + 1}"] = val
        
        return self.ft

    # Write joint index & force/torque readings to CSV file
    def write_forces(self, data_dict: dict) -> None:
        with open(self.ft_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ft_types)
            writer.writerow(data_dict)

    def get_joint_positions(self) -> list:
        joint_positions = []
        for joint in range(0, self.num_joints):
            joint_state = p.getJointState(self.robot, joint)
            joint_positions.append(joint_state[0])
        return joint_positions
    
    # Write current world frame positions of end effector to pos_file
    def write_wf_position(self) -> None:
        link_state = p.getLinkState(self.robot, self.ee_link_index)
        ee_position_wf = link_state[4]  # world frame position of the end effector
        ee_orientation_wf = p.getEulerFromQuaternion(link_state[5])  # world frame orientation of the end effector
        
        pos = {
            "x": ee_position_wf[0],
            "y": ee_position_wf[1],
            "z": ee_position_wf[2],
            "roll": ee_orientation_wf[0],
            "pitch": ee_orientation_wf[1],
            "yaw": ee_orientation_wf[2]
        }
        
        with open(self.pos_file, "w") as f:
            json.dump(pos, f)

    def reset_pose(self, position, orientation) -> None:
        ik_vals = p.calculateInverseKinematics(self.robot,
                                            self.ee_link_index,
                                            position,
                                            orientation,
                                            maxNumIterations=1000,
                                            residualThreshold=1e-5)
        
        for i in range(7):
            p.resetJointState(self.robot, i, ik_vals[i])

# -----------------------------------------------------------------------------------------------------------

'''
start_plotter: Starts the plotter.py script that plots forces on each joint
'''
def start_plotter():
    plotter_path = os.path.join(os.path.dirname(__file__), "plotter.py")
    if not os.path.exists(plotter_path):
        print(f"Plotter script not found at {plotter_path}.")
        sys.exit(1)
        
    plotter = subprocess.Popen(
        [sys.executable, plotter_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return plotter

def start_pos_gui():
    position_path = os.path.join(os.path.dirname(__file__), "position_gui.py")
    if not os.path.exists(position_path):
        print(f"Position script not found at {position_path}.")
        sys.exit(1)
        
    position_gui = subprocess.Popen(
        [sys.executable, position_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    return position_gui

# -----------------------------------------------------------------------------------------------------------

'''
cleanup: Kills subprocesses if they arte still active and disconnects from PyBullet
'''
def cleanup(processes, signum=None, frame=None):
    for proc in processes:
        # only terminate if still running
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    
    try:
        p.disconnect()
    except: # already disconnected, ignore
        pass
        
    # if this was a signal, exit now
    if signum is not None:
        sys.exit(0)

# -----------------------------------------------------------------------------------------------------------

'''
register_cleanup: Registers the "cleanup" function for Ctrl-C, Ctrl-Break, and normal program exit
'''
def register_cleanup(processes):
    # 3) register for Ctrl‑C and Ctrl‑Break
    signal.signal(signal.SIGINT, lambda s, f: cleanup(processes, signum=s, frame=f))   # Ctrl‑C
    signal.signal(signal.SIGBREAK, lambda s, f: cleanup(processes, signum=s, frame=f)) # Ctrl‑Break

    # 4) register for normal program exit
    atexit.register(cleanup, processes)
    
# -----------------------------------------------------------------------------------------------------------

def main():    
    # Initialize the simulation
    ip = InteractPerceive()
    ip.init_sim()
    ip.initialize_plot_file()
    
    subprocesses = []
    # processes.append(start_plotter())
    subprocesses.append(start_pos_gui())
    register_cleanup(subprocesses)
    
    plotter = Plotter().start()
    # gui = Position_GUI_Thread().start()

    up_position = [0.5, 0, 0.3]
    down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
    ip.reset_pose(up_position, down_orientation)
    time.sleep(1)  # let things settle

    print("Lowering into the cube…")

    # ———————————
    # 4) LOWER INTO THE CUBE
    # ———————————
    for z in [0.15, 0.1, 0.05]:  # descending heights
        target = [0.5, 0, z]
        ik_vals = p.calculateInverseKinematics(ip.robot,
                                               ip.ee_link_index,
                                               target,
                                               down_orientation)
        # command the arm joints
        for i in range(7):
            p.setJointMotorControl2(ip.robot, i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=ik_vals[i],
                                    force=200)
        # keep gripper half‑closed
        for j in ip.finger_joints:
            p.setJointMotorControl2(ip.robot, j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.04,
                                    force=50)
        ip.write_wf_position()
        ip.write_forces(ip.get_forces())
        time.sleep(2)  # give time for collision

    print("Contact made! Wrist readings follow:")
    
    # -----------------------------------------------------------------------------------------------------------
    # Stream force readings to csv file for plotting
    # -----------------------------------------------------------------------------------------------------------
    try:
        while True:
            ip.write_wf_position()
            ip.write_forces(ip.get_forces())
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
        
if __name__ == "__main__":
    main()
