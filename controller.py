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
        self.parent_map = defaultdict(list)
        self.get_idxs()

        self.num_revolute_joints = len(self.revolute_joint_idx)
        self.num_movable_joints = len(self.movable_joint_idxs)
        
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.ft_types = ["fx_raw", "fy_raw", "fz_raw", "tx_raw", "ty_raw", "tz_raw",
                         "fx_comp", "fy_comp", "fz_comp", "tx_comp", "ty_comp", "tz_comp",
                         "fx_wf", "fy_wf", "fz_wf", "tx_wf", "ty_wf", "tz_wf"]
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
                if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    self.movable_joint_idxs.append(i)
                if info[1].decode('utf-8') == "panda_grasptarget_hand":
                    self.ee_link_index = i
                if info[1].decode('utf-8') == "panda_joint8":
                    self.sensor_idx = i
                    
        if not self.ee_link_index:
            print("Could not find end effector link index, setting to last joint index")
            self.ee_link_index = self.revolute_joint_idx[-1]
            
        if not self.sensor_idx:
            print("Could not find sensor joint index, setting to last joint index")
            self.sensor_idx = self.revolute_joint_idx[-1]     
# -----------------------------------------------------------------------------------------------------------
    def get_descendants(self, link_idx) -> list:
        to_visit = [link_idx]
        descendants = []
        while to_visit:
            link = to_visit.pop()
            descendants.append(link)
            # add *all* child links (regardless of joint type)
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
        
        # icg_ft = self.get_downstream_icg_ft()
        # ft_comp = [raw_ft[i] - icg_ft[i] - gripper_mass_ft[i] for i in range(len(raw_ft))]
        # gripper_mass_ft = self.get_gripper_mass_ft()
        # ft_comp = [raw_ft[i] - gripper_mass_ft[i] for i in range(len(raw_ft))]
        ft_comp = self.get_ft_minus_mass()
        for name, val in zip(self.ft_names, ft_comp):
            self.ft[f"{name}_comp"] = val
            
        ft_wf = self.get_sensor_ft_wf()
        for name, val in zip(self.ft_names, ft_wf):
            self.ft[f"{name}_wf"] = val
        
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
    def keep_finger_position(self) -> None:
        for j in self.finger_joints:
            p.setJointMotorControl2(self.robot, j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=0.04,
                                    force=50)
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
    def maintain_robot_pose(self) -> None:
        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, [i for i in self.movable_joint_idxs])
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
            for idx, j in enumerate(self.movable_joint_idxs):
                axis = p.getJointInfo(self.robot, j)[13]   # joint axis in link frame
                torque = [axis[0]*tau_g[idx],
                        axis[1]*tau_g[idx],
                        axis[2]*tau_g[idx]]
                p.applyExternalTorque(self.robot,
                                    j,
                                    torque,
                                    flags=p.LINK_FRAME)
# -----------------------------------------------------------------------------------------------------------
    def get_downstream_icg_ft(self) -> list:
        with self.sim_lock:
            # Get current positions and velocities of all joints
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
            q = [joint_state[0] for joint_state in joint_states]
            qdot = [joint_state[1] for joint_state in joint_states]
            
            sensor_info = p.getJointInfo(self.robot, self.sensor_idx)
            joint_frame_pos, joint_frame_ori = sensor_info[14], sensor_info[15]
            
            # Convert joint frame to world frame
            raw_ft_joint = p.getJointState(self.robot, self.sensor_idx)[2]
            raw_f_wf = p.rotateVector(joint_frame_ori, raw_ft_joint[0:3])
            raw_t_wf = p.rotateVector(joint_frame_ori, raw_ft_joint[3:6])
            
            # Convert from world frame to adjusted sensor frame
            sensor_pos_wf, sensor_ori_wf = p.getLinkState(self.robot, self.sensor_idx)[:2]
            inv_s = p.invertTransform([0,0,0], sensor_ori_wf)[1]
            raw_f_s = p.rotateVector(inv_s, raw_f_wf)
            raw_t_s = p.rotateVector(inv_s, raw_t_wf)
            
            downstream_links = self.get_descendants(self.sensor_idx)
            downstream_actuated = [j for j in self.movable_joint_idxs if j in downstream_links]
            m = len(downstream_actuated)
            movable_joint_map = {joint:idx for idx, joint in enumerate(self.movable_joint_idxs)}
            movable_joint_list = [movable_joint_map[j] for j in downstream_actuated]
            
            # Full inverse‐dynamics torques (G + C(q,qdot)⋅qdot; M⋅0 = 0)
            # G = gravity, C = coriolis/centrifugal, M/i = inertia which we want to be 0
            tau_icg_all = p.calculateInverseDynamics(self.robot, q, qdot, [0.0]*self.num_movable_joints)
            tau_icg = [tau_icg_all[movable_joint_map[j]] for j in downstream_actuated]
            print(f"tau_icg_all: {tau_icg_all}")
            print(f"tau_icg: {tau_icg}")
            
            # Joint damping & Coulomb friction torques
            tau_df = []
            for j in downstream_actuated:
                info = p.getJointInfo(self.robot, j)
                damping = info[6]
                friction = info[7]
                vel = qdot[movable_joint_map[j]]
                tau_df.append( -damping*vel + ( -friction*math.copysign(1,vel) if abs(vel)>1e-8 else 0.0 ) )
                print(f"joint {j}, friction: {friction}, vel: {vel}, damping: {damping}")
                print(f"tau_df: {tau_df}")
            
            # Total predicted joint torque
            tau_total = np.array(tau_icg) + np.array(tau_df)     # shape (m,)
            
            Jp_all, Jo_all = p.calculateJacobian(
                self.robot, 
                self.sensor_idx,
                localPosition = joint_frame_pos,
                objPositions = q,
                objVelocities = qdot,
                objAccelerations = [0.0]*self.num_movable_joints
            )
            J_full = np.vstack((Jp_all, Jo_all))   # shape (6, m)
            Jp_all = np.array(Jp_all)
            Jo_all = np.array(Jo_all)
            
            Jp = Jp_all[:, movable_joint_list]
            Jo = Jo_all[:, movable_joint_list]
            J = np.vstack((Jp, Jo))   # shape (6, m)

            # Map joint torques to equivalent world‐frame ft f_pred so that J^T f_pred = tau_total
            # f_pred = inv(J J^T) J * tau_total
            # JJt = J.dot(J.T)
            # f_pred_wf = np.linalg.inv(JJt + 1e-3*np.eye(6)).dot(J.dot(tau_total))
            f_pred_wf = np.linalg.pinv(J.T).dot(tau_total)
            print(f"f_pred_wf: {f_pred_wf}")

            # Rotate from world frame to sensor frame
            inv_ori = p.invertTransform([0,0,0], sensor_ori_wf)[1]
            f_pred_s = p.rotateVector(inv_ori, f_pred_wf[0:3])
            t_pred_s = p.rotateVector(inv_ori, f_pred_wf[3:6])

        return f_pred_s + t_pred_s
# -----------------------------------------------------------------------------------------------------------
    def get_sensor_ft_wf(self) -> list:
        with self.sim_lock:
            sensor_state = p.getLinkState(self.robot, self.sensor_idx)
            sensor_pos_wf, sensor_ori_wf = sensor_state[0], sensor_state[1]
            
            raw_ft = p.getJointState(self.robot, self.sensor_idx)[2]
            
            # Convert joint frame to world frame
            raw_f_wf = p.rotateVector(sensor_ori_wf, raw_ft[0:3])
            raw_t_wf = p.rotateVector(sensor_ori_wf, raw_ft[3:6])
            
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
            
            mass_f_wf = [0.0, 0.0, 9.91]
            mass_t_wf = [0.0, 0.0, -0.47]
            
            comp_f_wf = [raw_f_wf[i] - mass_f_wf[i] for i in range(3)]
            comp_t_wf = [raw_t_wf[i] - mass_t_wf[i] for i in range(3)]
            
            _, inv_ori_s = p.invertTransform([0,0,0], sensor_ori_wf)
            
            comp_f_s = p.rotateVector(inv_ori_s, comp_f_wf)
            comp_t_s = p.rotateVector(inv_ori_s, comp_t_wf)
            
        return comp_f_s + comp_t_s
    
        #     raw_ft_s = p.getJointState(self.robot, self.sensor_idx)[2]
        #     joint_ori_wf = p.getJointInfo(self.robot, self.sensor_idx)[15] 
            
        #     # Convert joint frame to world frame
        #     raw_f_wf = p.rotateVector(joint_ori_wf, raw_ft_s[0:3])
        #     raw_t_wf = p.rotateVector(joint_ori_wf, raw_ft_s[3:6])
            
        #     mass_f_wf = [0.0, 0.0, 9.91]
        #     mass_t_wf = [0.0, 0.0, -0.47]
            
        #     comp_f_wf = [raw_f_wf[i] - mass_f_wf[i] for i in range(3)]
        #     comp_t_wf = [raw_t_wf[i] - mass_t_wf[i] for i in range(3)]
            
        #     _, inv_ori_s = p.invertTransform([0,0,0], joint_ori_wf)
            
        #     comp_f_s = p.rotateVector(inv_ori_s, comp_f_wf)
        #     comp_t_s = p.rotateVector(inv_ori_s, comp_t_wf)
            
        # return comp_f_s + comp_t_s
# -----------------------------------------------------------------------------------------------------------
    def get_gripper_mass_ft(self) -> list:
        robot = self.robot
        sensor_idx = self.sensor_idx
        downstream_links = self.get_descendants(sensor_idx)
        
        with self.sim_lock:
            sensor_state = p.getLinkState(robot, sensor_idx)
            sensor_pos_wf, sensor_ori_wf = sensor_state[0], sensor_state[1]
            
            # A -> B transform
            # invPosB, invOrnB = p.invertTransform(posB_wf, ornB_wf)
            # posA_inB, ornA_inB = p.multiplyTransforms(
            #     invPosB, invOrnB,
            #     posA_wf,  ornA_wf
            # )
            
            inv_pos_s, inv_ori_s = p.invertTransform(sensor_pos_wf, sensor_ori_wf)
            
            F_s = [0.0, 0.0, 0.0]
            T_s = [0.0, 0.0, 0.0]
            
            for link in downstream_links:
                # Get mass and center of mass in local frame
                dyn = p.getDynamicsInfo(robot, link)
                mass, com_pos_local, com_ori_local = dyn[0], dyn[3], dyn[4]
                
                # Get position and orientation of the link in world frame
                link_state = p.getLinkState(robot, link, computeForwardKinematics=True)
                link_pos_wf, link_ori_wf = link_state[0], link_state[1]
                
                # Center of mass
                com_pos_wf, com_ori_wf = p.multiplyTransforms(link_pos_wf, link_ori_wf, com_pos_local, [0, 0, 0, 1])
                com_pos_s, com_ori_s = p.multiplyTransforms(inv_pos_s, inv_ori_s, com_pos_wf, [0, 0, 0, 1])
                r = com_pos_s
                
                # gravity force in world frame
                # negative because we want to oppose the gravity force
                F_link_wf = [0.0, 0.0, -mass*self.sim.gravity]
                F_link_s = p.rotateVector(inv_ori_s, F_link_wf)
                
                # torque in sensor frame
                T_link_s = [
                    r[1]*F_link_s[2] - r[2]*F_link_s[1],
                    r[2]*F_link_s[0] - r[0]*F_link_s[2],
                    r[0]*F_link_s[1] - r[1]*F_link_s[0]
                ]
                
                for i in range(3):
                    F_s[i] += F_link_s[i]
                    T_s[i] += T_link_s[i]
                print(f"link {link}, ft: {list(F_link_s) + T_link_s}")
                
            print(f"total mass ft: {F_s + T_s}")

        return F_s + T_s
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
    def do_wiggle(self) -> None:
        with self.sim_lock:
            for i in range(self.num_revolute_joints):
                p.setJointMotorControl2(self.robot, i,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetPosition=self.ik_vals[i],
                                        force=100,
                                        maxVelocity=0.5)
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
            
            # if self.go_pos:
                # print("Moving to desired position...")
                # self.keep_finger_position()
                # self.go_to_desired_position()
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