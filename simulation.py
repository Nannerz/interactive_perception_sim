import pybullet as p
import pybullet_data
import time, math, csv, sys, os, subprocess, threading, signal, atexit, json
import numpy as np
from collections import deque
import pandas as pd

# -----------------------------------------------------------------------------------------------------------
class Simulation(threading.Thread):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.daemon = True
        self.interval = 0.01
        
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
                                basePosition=[0.75, 0, 0.1])

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

    def get_joint_positions(self) -> list[float]:
        joint_positions = []
        for joint in range(0, self.num_joints):
            joint_state = p.getJointState(self.robot, joint)
            joint_positions.append(joint_state[0])
        return joint_positions
    
    def get_ee_position(self) -> list[float]:
        link_state = p.getLinkState(self.robot, self.ee_link_index)
        ee_position_wf = link_state[4]  # world frame position of the end effector
        ee_orientation_wf = p.getEulerFromQuaternion(link_state[5])  # world frame orientation of the end effector
        
        return ee_position_wf, ee_orientation_wf
        
    # Write current world frame positions of end effector to pos_file
    def write_wf_position(self) -> None:
        ee_position_wf, ee_orientation_wf = self.get_ee_position()
        
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