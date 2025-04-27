import threading, json, os, csv, time, math, sys
import pybullet as p
import numpy as np
from simulation import Simulation
from fsm import FSM
# -----------------------------------------------------------------------------------------------------------

class Controller(threading.Thread):
    _instance = None
# -----------------------------------------------------------------------------------------------------------
    def __new__(cls, *args, **kwargs) -> 'Controller':
        if cls._instance is None:
            cls._instance = super(Controller, cls).__new__(cls)
        return cls._instance
# -----------------------------------------------------------------------------------------------------------
    def __init__(self, sim: Simulation, data_path, shutdown_event: threading.Event, **kwargs) -> None:
        super().__init__(**kwargs)
        self.daemon = True
        self.interval = 0.001
        self.sim = sim
        self.robot = sim.robot
        self.sim_lock = sim.sim_lock
        self.shutdown_event = shutdown_event
        self.fsm = FSM(controller=self)
        
        # self.wrist_joint = 7 # joint before the gripper
        self.finger_joints = [9, 10]

        self.joint_idx = []
        self.movable_joint_idx = []
        with self.sim_lock:
            for i in range(p.getNumJoints(self.robot)):
                info = p.getJointInfo(self.robot, i)
                if info[2] == p.JOINT_REVOLUTE:
                    self.joint_idx.append(i)
                if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    self.movable_joint_idx.append(i)
                if info[1].decode('utf-8') == "panda_grasptarget_hand":
                    self.ee_link_index = i
        
        if not self.ee_link_index:
            print("Could not find end effector link index, exiting")
            sys.exit(0)

        self.num_joints = len(self.joint_idx)
        self.num_movable_joints = len(self.movable_joint_idx)
        
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.ft_types = [f"{name}_{j}" for j in range(1, self.num_joints+1) for name in self.ft_names]

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
        self.go_pos = True
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
        if os.path.isfile(self.ft_file):
            with open(self.ft_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.ft_types)
                writer.writerow(data_dict)
        else:
            print(f"Could not find file {self.ft_file}, exiting")
            sys.exit(0)
# -----------------------------------------------------------------------------------------------------------
    def get_joint_positions(self) -> list[float]:
        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, self.joint_idx)
            joint_positions = [joint_state[0] for joint_state in joint_states]
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
        
        try:
            with open(self.pos_file, "w") as f:
                json.dump(pos, f)
        except:
            print(f"Could not find file {self.pos_file}, exiting")
            sys.exit(0)     
# -----------------------------------------------------------------------------------------------------------
    def reset_pose(self, position, orientation) -> None:
        with self.sim_lock:
            ik_vals = p.calculateInverseKinematics(self.robot,
                                                self.ee_link_index,
                                                position,
                                                orientation,
                                                maxNumIterations=1000,
                                                residualThreshold=1e-5)
            
            for i in self.joint_idx:
                p.resetJointState(self.robot, i, ik_vals[i])
# -----------------------------------------------------------------------------------------------------------
    def set_desired_position(self, ik_vals) -> None:
        self.ik_vals = ik_vals
# -----------------------------------------------------------------------------------------------------------
    def keep_finger_position(self) -> None:
        for j in self.finger_joints:
            p.setJointMotorControl2(self.robot, j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.04,
                                    force=50)
# -----------------------------------------------------------------------------------------------------------
    def go_to_desired_position(self, force=100) -> None:
        with self.sim_lock:
            for i in range(self.num_joints):
                p.setJointMotorControl2(self.robot, i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.ik_vals[i],
                                        force=force,
                                        maxVelocity=0.5)
# -----------------------------------------------------------------------------------------------------------
    def compensate_gravity(self) -> None:
        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, [i for i in self.movable_joint_idx])
            current_positions = [joint_state[0] for joint_state in joint_states]
            current_velocities = [joint_state[1] for joint_state in joint_states]

        with self.sim_lock:
            # 3) compute gravity‐only torques (zero vel & accel → pure gravity term)
            tau_full = p.calculateInverseDynamics(self.robot,
                                                  current_positions,
                                                  [0.0]*self.num_movable_joints,
                                                  [0.0]*self.num_movable_joints)
            tau_g = [tau_full[i] for i in range(self.num_movable_joints)]

            # 4) apply those torques as external torques
            for idx, j in enumerate(self.movable_joint_idx):
                axis = p.getJointInfo(self.robot, j)[13]   # joint axis in link frame
                torque = [axis[0]*tau_g[idx],
                        axis[1]*tau_g[idx],
                        axis[2]*tau_g[idx]]
                p.applyExternalTorque(self.robot,
                                    j,
                                    torque,
                                    flags=p.LINK_FRAME)
# -----------------------------------------------------------------------------------------------------------
    def initial_pos(self) -> None:
        initial_x = 0.7
        initial_y = 0
        initial_z = 0.4
        up_position = [initial_x, initial_y, initial_z]
        with self.sim_lock:
            down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.reset_pose(up_position, down_orientation)
# -----------------------------------------------------------------------------------------------------------
    def do_startup(self, next_z) -> None:
        initial_x = 0.7
        initial_y = 0

        with self.sim_lock:
            down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        
        # lower to object
        print("Lowering into the cube…")
        with self.sim_lock:
            target = [initial_x, initial_y, next_z]
            ik_vals = p.calculateInverseKinematics(self.robot,
                                                    self.ee_link_index,
                                                    target,
                                                    down_orientation)
        self.set_desired_position(ik_vals)
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        while not self.shutdown_event.is_set():
            time.sleep(self.interval)
            
            self.fsm.next_state()
            
            if self.go_pos:
                print("Moving to desired position...")
                self.keep_finger_position()
                self.go_to_desired_position()
            self.compensate_gravity()
            # print("Position: ", self.get_ee_position())
            # print("Desired Position: ", self.ik_vals)

            self.write_wf_position()
            self.write_forces(self.get_forces())
            
        print("Exiting controller thread...")
        sys.exit(0)
            
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)