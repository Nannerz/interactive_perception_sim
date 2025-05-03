import threading, json, os, csv, time, math, sys
from collections import defaultdict
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

        self.revolute_joint_idx = []
        self.movable_joint_idxs = []
        self.all_joint_idx = []
        self.ee_link_index = None
        self.sensor_idx = None
        self.wrist_idx = None
        self.parent_map = defaultdict(list)
        self.get_idxs()

        self.num_revolute_joints = len(self.revolute_joint_idx)
        self.num_movable_joints = len(self.movable_joint_idxs)
        
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.ft_types = ["fx_raw", "fy_raw", "fz_raw", "tx_raw", "ty_raw", "tz_raw",
                         "fx_comp", "fy_comp", "fz_comp", "tx_comp", "ty_comp", "tz_comp"]
                        #  "fx_wf", "fy_wf", "fz_wf", "tx_wf", "ty_wf", "tz_wf"]
        self.ft = {ft_name: 0 for ft_name in self.ft_types}

        self.data_path = data_path
        
        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.initialize_plot_file()
        
        # State variables
        self.ik_vals = [0] * self.num_revolute_joints
        self.go_pos = True
# -----------------------------------------------------------------------------------------------------------
    def get_idxs(self) -> None:
        with self.sim_lock:
            self.total_num_joints = p.getNumJoints(self.robot)
            for i in range(self.total_num_joints):
                info = p.getJointInfo(self.robot, i)
                self.all_joint_idx.append(i)
                
                parent_link = info[16]
                self.parent_map[parent_link].append(i)
                
                if info[2] == p.JOINT_REVOLUTE:
                    self.revolute_joint_idx.append(i)
                if info[2] != p.JOINT_FIXED:
                    self.movable_joint_idxs.append(i)
                if info[1].decode('utf-8') == "panda_grasptarget_hand":
                    self.ee_link_index = i
                if info[1].decode('utf-8') == "panda_joint8":
                    self.sensor_idx = i
                if info[1].decode('utf-8') == "panda_joint7":
                    self.wrist_idx = i
                    
        if not self.ee_link_index:
            print("Could not find end effector link index, setting to last joint index")
            self.ee_link_index = self.revolute_joint_idx[-1]
            
        if not self.sensor_idx:
            print("Could not find sensor joint index, setting to last joint index")
            self.sensor_idx = self.revolute_joint_idx[-1]    
            
        if not self.wrist_idx:
            print("Could not find wrist joint index, setting to last joint index")
            self.wrist_idx = self.revolute_joint_idx[-1] 
# -----------------------------------------------------------------------------------------------------------
    def get_downstream(self, link_idx) -> list:
        to_visit = [link_idx]
        descendants = []
        while to_visit:
            link = to_visit.pop()
            descendants.append(link)
            to_visit.extend(self.parent_map[link])
        return descendants
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
            raw_ft = p.getJointState(self.robot, self.sensor_idx)[2]
        for name, val in zip(self.ft_names, raw_ft):
            self.ft[f"{name}_raw"] = val
        
        ft_comp = self.get_ft_minus_mass()
        for name, val in zip(self.ft_names, ft_comp):
            self.ft[f"{name}_comp"] = val
            
        # ft_wf = self.get_sensor_ft_wf()
        # for name, val in zip(self.ft_names, ft_wf):
        #     self.ft[f"{name}_wf"] = val
        
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
            joint_states = p.getJointStates(self.robot, self.revolute_joint_idx)
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
            "roll": ee_orientation_wf[0]*180/math.pi,
            "pitch": ee_orientation_wf[1]*180/math.pi,
            "yaw": ee_orientation_wf[2]*180/math.pi
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
            
            for i in self.revolute_joint_idx:
                p.resetJointState(self.robot, i, ik_vals[i])
# -----------------------------------------------------------------------------------------------------------
    def set_desired_position(self, ik_vals) -> None:
        self.ik_vals = ik_vals
# -----------------------------------------------------------------------------------------------------------
    def go_to_desired_position(self, force=100) -> None:
        with self.sim_lock:
            for i in range(self.num_revolute_joints):
                p.setJointMotorControl2(self.robot, i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.ik_vals[i],
                                        force=force,
                                        maxVelocity=0.5)
# -----------------------------------------------------------------------------------------------------------
    def get_sensor_ft_wf(self) -> list:
        with self.sim_lock:
            sensor_state = p.getLinkState(self.robot, self.sensor_idx)
            sensor_pos_wf, sensor_ori_wf = sensor_state[0], sensor_state[1]
            
            raw_ft = p.getJointState(self.robot, self.sensor_idx)[2]
            
            # Convert joint frame to world frame
            raw_f_wf = p.rotateVector(sensor_ori_wf, raw_ft[0:3])
            raw_t_wf = p.rotateVector(sensor_ori_wf, raw_ft[3:6])
        
        print(f"f_wf: {raw_f_wf}, t_wf: {raw_t_wf}")
        return raw_f_wf + raw_t_wf
# -----------------------------------------------------------------------------------------------------------
    def get_ft_minus_mass(self) -> list:
        with self.sim_lock:
            sensor_state = p.getLinkState(self.robot, self.sensor_idx)
            sensor_pos_wf, sensor_ori_wf = sensor_state[0], sensor_state[1]
            
            raw_ft = p.getJointState(self.robot, self.sensor_idx)[2]
            
            # Convert joint frame to world frame
            raw_f_wf = p.rotateVector(sensor_ori_wf, raw_ft[0:3])
            raw_t_wf = p.rotateVector(sensor_ori_wf, raw_ft[3:6])
            
            mass_f_wf = [-0.0021, -0.0009, 9.9064]
            mass_t_wf = [0.0002, -0.4716, 0.0000]
            
            comp_f_wf = [raw_f_wf[i] - mass_f_wf[i] for i in range(3)]
            comp_t_wf = [raw_t_wf[i] - mass_t_wf[i] for i in range(3)]
            
            _, inv_ori_s = p.invertTransform([0,0,0], sensor_ori_wf)
            
            comp_f_s = p.rotateVector(inv_ori_s, comp_f_wf)
            comp_t_s = p.rotateVector(inv_ori_s, comp_t_wf)
            
        return comp_f_s + comp_t_s
# -----------------------------------------------------------------------------------------------------------
    def get_jacobian(self) -> list:
        with self.sim_lock:
            # Get current positions and velocities of all joints
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
            q = [joint_state[0] for joint_state in joint_states]
            qdot = [joint_state[1] for joint_state in joint_states]
            n = self.num_movable_joints
            
            ee_pos_wf, ee_ori_wf = p.getLinkState(self.robot, self.ee_link_index)[:2]
            wrist_pos_local = p.getJointState(self.robot, self.wrist_idx)[0]
            
            # Full inverse‐dynamics torques (G + C(q,qdot)⋅qdot; M⋅0 = 0)
            # G = gravity, C = coriolis/centrifugal, M/i = inertia which we want to be 0
            tau_icg = p.calculateInverseDynamics(self.robot, q, qdot, [0.0]*n)
            
            movable_joint_map = {joint:idx for idx, joint in enumerate(self.movable_joint_idxs)}
            
            # Joint damping & Coulomb friction torques
            tau_df = []
            for j in self.movable_joint_idxs:
                info = p.getJointInfo(self.robot, j)
                damping = info[6]
                friction = info[7]
                vel = qdot[movable_joint_map[j]]
                tau_df.append( -damping*vel + ( -friction*math.copysign(1,vel) if abs(vel)>1e-8 else 0.0 ) )
            
            # Total predicted joint torque
            tau_total = np.array(tau_icg) + np.array(tau_df) # shape (n,)
            
            Jp, Jo = p.calculateJacobian(
                self.robot, 
                self.wrist_idx,
                localPosition = wrist_pos_local,
                objPositions = q,
                objVelocities = qdot,
                objAccelerations = [0.0]*n
            )
            J = np.vstack((Jp, Jo))   # shape (6, n)

            # Map joint torques to equivalent world‐frame ft f_pred so that J^T f_pred = tau_total
            # f_pred = inv(J J^T) J * tau_total
            # JJt = J.dot(J.T)
            # f_pred_wf = np.linalg.inv(JJt + 1e-3*np.eye(6)).dot(J.dot(tau_total))
            # f_pred_wf = np.linalg.pinv(J.T).dot(tau_total)
# -----------------------------------------------------------------------------------------------------------
    def initial_pos(self) -> None:
        initial_x = 0.7
        initial_y = 0
        initial_z = 0.4
        up_position = [initial_x, initial_y, initial_z]
        with self.sim_lock:
            down_orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
        self.reset_pose(up_position, down_orientation)
# -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self,
                  direction,
                  body_id = None,
                  ee_link_idx = None,
                  wrist_link_idx = None,
                  max_force=500) -> None:
        
        body_id = body_id or self.robot
        ee_link_idx = ee_link_idx or self.ee_link_index
        wrist_link_idx = wrist_link_idx or self.wrist_idx
        
        w_p, w_o = p.getLinkState(body_id, wrist_link_idx,   computeForwardKinematics=True)[:2]
        g_p, g_o = p.getLinkState(body_id, ee_link_idx, computeForwardKinematics=True)[:2]
        
        inv_p, inv_o = p.invertTransform(g_p, g_o)
        local_pos, _ = p.multiplyTransforms(inv_p, inv_o, w_p, w_o)
        
        omega_local = np.array([2 * direction, 0, 0])
        
        # 2) rotate local ω to world frame
        _, orn = p.getLinkState(body_id, ee_link_idx)[:2]
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
        omega_world = R.dot(omega_local)
        
        # 3) build full 6-vector twist (v = [0; ω_world])
        v = np.hstack((np.zeros(3), omega_world))
        
        # 4) compute current joint positions & zero velocities/accels
        q = []
        for j in self.movable_joint_idxs:
            q.append(p.getJointState(body_id, j)[0])
            
        zero = [0.0]*self.num_movable_joints
        
        # 5) get Jacobians (world frame)
        J_lin, J_ang = p.calculateJacobian(body_id,
                                        ee_link_idx,
                                        local_pos,
                                        q, zero, zero)
        # stack into 6×n
        J = np.vstack((np.array(J_lin), np.array(J_ang)))  # shape (6,n)
        
        # 6) solve for joint velocities dq = J⁺·v
        dq = np.linalg.pinv(J).dot(v)  # shape (n,)
        
        # 7) send velocity command to each moving joint
        for idx, joint_idx in enumerate(self.movable_joint_idxs):
            p.setJointMotorControl2(
                bodyIndex=body_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=float(dq[idx]),
                force=max_force
            )
        
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
                # print("Moving to desired position...")
                # self.keep_finger_position()
                self.go_to_desired_position()
            # self.maintain_robot_pose()
            # print("Position: ", self.get_ee_position())
            # print("Desired Position: ", self.ik_vals)

            self.write_wf_position()
            self.write_forces(self.get_forces())
            
        print("Exiting controller thread...")
        sys.exit(0)
            
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)