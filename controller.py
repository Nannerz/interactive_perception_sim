import threading, json, os, csv, time, math, sys
import pybullet as p
import numpy as np
from simulation import Simulation
# -----------------------------------------------------------------------------------------------------------

class Controller(threading.Thread):
    _instance = None
# -----------------------------------------------------------------------------------------------------------
    def __new__(cls, *args, **kwargs) -> 'Controller':
        if cls._instance is None:
            cls._instance = super(Controller, cls).__new__(cls)
        return cls._instance
# -----------------------------------------------------------------------------------------------------------
    def __init__(self, sim: Simulation, data_path, **kwargs):
        super().__init__(**kwargs)
        self.daemon = True
        self.interval = 0.01
        self.sim = sim
        self.robot = sim.robot
        self.sim_lock = sim.sim_lock
        
        self.wrist_joint = 7 # joint before the gripper
        self.finger_joints = [9, 10]
        self.ee_link_index = 11 # Panda “tool” link for IK
        self.num_joints = 8
        
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.ft_types = [f"{name}_{j}" for j in range(1, self.num_joints+1) for name in self.ft_names]
        # self.maxlen = 1000
        self.ft = {
            # f"{name}_{j}": deque(maxlen=self.maxlen)
            f"{name}_{j}": 0
            for j in range(1, self.num_joints+1)
            for name in self.ft_names
        }
        
        self.data_path = data_path
        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.initialize_plot_file()
        
        # State variables
        self.ik_vals = [0] * self.num_joints
        self.startup_done = False
# -----------------------------------------------------------------------------------------------------------
    # Initialize CSV file with headers, overwrites/deletes existing file with the same name
    def initialize_plot_file(self) -> None:
        with open(self.ft_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ft_types)
            writer.writeheader()
# -----------------------------------------------------------------------------------------------------------
    # Get force/torque readings from each joint
    def get_forces(self) -> dict:
        with self.sim_lock:
            for joint in range(0, self.num_joints):
                _, _, forces, _ = p.getJointState(self.robot, joint)
                # fx, fy, fz, mx, my, mz = forces
                for name, val in zip(self.ft_names, forces):
                    self.ft[f"{name}_{joint + 1}"] = val
        
        return self.ft
# -----------------------------------------------------------------------------------------------------------
    # Write joint index & force/torque readings to CSV file
    def write_forces(self, data_dict: dict) -> None:
        with open(self.ft_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ft_types)
            writer.writerow(data_dict)
# -----------------------------------------------------------------------------------------------------------
    def get_joint_positions(self) -> list[float]:
        with self.sim_lock:
            joint_positions = []
            for joint in range(0, self.num_joints):
                joint_state = p.getJointState(self.robot, joint)
                joint_positions.append(joint_state[0])
        return joint_positions
# -----------------------------------------------------------------------------------------------------------
    def get_ee_position(self) -> list[float]:
        with self.sim_lock:
            link_state = p.getLinkState(self.robot, self.ee_link_index)
            ee_position_wf = link_state[4]  # world frame position of the end effector
            ee_orientation_wf = p.getEulerFromQuaternion(link_state[5])  # world frame orientation of the end effector
        
        return ee_position_wf, ee_orientation_wf
# -----------------------------------------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------------------------------------
    def reset_pose(self, position, orientation) -> None:
        with self.sim_lock:
            ik_vals = p.calculateInverseKinematics(self.robot,
                                                self.ee_link_index,
                                                position,
                                                orientation,
                                                maxNumIterations=1000,
                                                residualThreshold=1e-5)
            
            for i in range(7):
                p.resetJointState(self.robot, i, ik_vals[i])
# -----------------------------------------------------------------------------------------------------------
    def set_desired_position(self, ik_vals) -> None:
        self.ik_vals = ik_vals
# -----------------------------------------------------------------------------------------------------------
    def do_startup(self) -> None:
        initial_x = 0.7
        initial_y = 0
        initial_z = 0.4
        up_position = [initial_x, initial_y, initial_z]
        down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.reset_pose(up_position, down_orientation)
        time.sleep(1)  # let things settle

        print("Lowering into the cube…")

        # ———————————
        # 4) LOWER INTO THE CUBE
        # ———————————
        for z in np.arange(initial_z, 0.15, -0.05):  # descending heights
            with self.sim_lock:
                target = [initial_x, initial_y, z]
                ik_vals = p.calculateInverseKinematics(self.robot,
                                                    self.ee_link_index,
                                                    target,
                                                    down_orientation)
                # command the arm joints
                for i in range(7):
                    p.setJointMotorControl2(self.robot, i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=ik_vals[i],
                                            force=5)
                # keep gripper half‑closed
                for j in self.finger_joints:
                    p.setJointMotorControl2(self.robot, j,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=0.04,
                                            force=50)
            print("Position: ", self.get_ee_position())
            self.write_wf_position()
            self.write_forces(self.get_forces())
            time.sleep(2)  # give time for collision

        print("Contact made! Wrist readings follow:")
        
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        if not self.startup_done:
            self.do_startup()
            self.startup_done = True
            
        # -----------------------------------------------------------------------------------------------------------
        # Stream force readings to csv file for plotting
        # -----------------------------------------------------------------------------------------------------------
        try:
            self.write_wf_position()
            self.write_forces(self.get_forces())
        except KeyboardInterrupt:
            pass

        
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)