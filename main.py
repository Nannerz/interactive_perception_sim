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

# -----------------------------------------------------------------------------------------------------------
class InteractPerceive:
    def __init__(self):
        super().__init__()
        self.robot = None
        self.filename = "ft_data.csv"
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.num_joints = 7
        self.fieldnames = [f"{name}_{j}" for j in range(1, self.num_joints+1) for name in self.ft_names]
        self.ft = {
            f"{name}_{j}": 0
            for j in range(1, self.num_joints+1)
            for name in self.ft_names
        }
        
    def init_sim(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(1)
        p.resetDebugVisualizerCamera(cameraDistance=1.2,
                                          cameraYaw=50,
                                          cameraPitch=-30,
                                          cameraTargetPosition=[0.5, 0, 0.2])
        p.loadURDF("plane.urdf") # ground plane

        urdf_dir = os.path.join(self.path, "panda_with_sensor.urdf")
        print(f"Loading URDF from: {urdf_dir}")
        self.robot = p.loadURDF(urdf_dir,
                                basePosition=[0, 0, 0],
                                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                                useFixedBase=True)
        self.create_object()
        
    def create_object(self):
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
    def initialize_plot_file(self, filename: str = None, fieldnames: list = None):
        if filename is None:
            filename = self.filename
        if fieldnames is None:
            fieldnames = self.fieldnames
            
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Write joint index & force/torque readings to CSV file
    def write_to_csv(self, data_dict: dict, filename: str = None, fieldnames: list = None):
        if filename is None:
            filename = self.filename
        if fieldnames is None:
            fieldnames = self.fieldnames
            
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(data_dict)
            
    def get_forces(self):
        for joint in range(0, self.num_joints):
            _, _, forces, _ = p.getJointState(self.robot, joint)
            # fx, fy, fz, mx, my, mz = forces
            for name, val in zip(self.ft_names, forces):
                self.ft[f"{name}_{joint + 1}"] = val
        
        return self.ft

# -----------------------------------------------------------------------------------------------------------

'''
start_plotter

Starts the plotter.py script that plots forces on each joint
Registers cleanup function on exit to kill the plot process
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

def main():    
    # Initialize the simulation
    ip = InteractPerceive()
    ip.init_sim()
    ip.initialize_plot_file()
    start_plotter()

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
    ik_vals = p.calculateInverseKinematics(ip.robot,
                                           ee_link_index,
                                           above_pos,
                                           down_ori,
                                           maxNumIterations=1000,
                                           residualThreshold=1e-5)
    # snap to that pose before we start moving
    for i in range(7):
        p.resetJointState(ip.robot, i, ik_vals[i])
        p.enableJointForceTorqueSensor(ip.robot, i, 1)
    time.sleep(1)  # let things settle

    print("Lowering into the cube…")

    # ———————————
    # 4) LOWER INTO THE CUBE
    # ———————————
    for z in [0.15, 0.1, 0.05]:  # descending heights
        target = [0.5, 0, z]
        ik_vals = p.calculateInverseKinematics(ip.robot,
                                               ee_link_index,
                                               target,
                                               down_ori)
        # command the arm joints
        for i in range(7):
            p.setJointMotorControl2(ip.robot, i,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=ik_vals[i],
                                    force=200)
        # keep gripper half‑closed
        for j in finger_joints:
            p.setJointMotorControl2(ip.robot, j,
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
            ft = ip.get_forces()
            ip.write_to_csv(data_dict=ft)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        
if __name__ == "__main__":
    main()
