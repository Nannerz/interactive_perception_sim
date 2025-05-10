import threading, json, os, csv, time, math, sys
from collections import defaultdict
import pybullet as p
import numpy as np
from simulation import Simulation
from fsm import FSM
# -----------------------------------------------------------------------------------------------------------
class Controller(threading.Thread):
    def __init__(self, 
                 sim: Simulation, 
                 shutdown_event: threading.Event, 
                 **kwargs) -> None:
        
        super().__init__(**kwargs)
        self.daemon = True
        self.interval = 0.001
        self.sim = sim
        self.robot = sim.robot
        self.sim_lock = sim.sim_lock
        self.shutdown_event = shutdown_event
        self.initial_x = 0.8
        self.initial_y = 0
        self.initial_z = 0.6
        self.fsm = FSM(controller=self,
                       initial_x=self.initial_x,
                       initial_y=self.initial_y,
                       initial_z=self.initial_z)
        
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
        self.ft_types = ["raw", "comp"]  # , "wf"]
        self.ft_keys = [f"{name}_{type}" for type in self.ft_types for name in self.ft_names]
        self.ft = {ft_name: 0 for ft_name in self.ft_keys}
        
        self.joint_vels = {joint: 0 for joint in self.revolute_joint_idx}

        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.vel_file = os.path.join(self.data_path, "vel_data.csv")
        self.initialize_plot_files()
        
        # State variables
        self.mode = 'position'
        self.pause_controls = False
        self.next_vel = [0.0] * self.num_movable_joints
        self.next_pos = [0.0] * self.num_movable_joints # ik_values
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
    ''' Initialize CSV file with headers, overwrites/deletes existing file with the same name '''
    def initialize_plot_files(self) -> None:
        with open(self.ft_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.ft.keys())
            writer.writeheader()
            
        with open(self.vel_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.joint_vels.keys())
            writer.writeheader()
# -----------------------------------------------------------------------------------------------------------
    ''' Get force/torque readings from each joint '''
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
    def get_joint_velocities(self) -> dict:
        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, self.revolute_joint_idx)
            joint_velocities = [joint_state[1] for joint_state in joint_states]
        
        for i, joint in enumerate(self.revolute_joint_idx):
            self.joint_vels[joint] = joint_velocities[i]
        
        return self.joint_vels
# -----------------------------------------------------------------------------------------------------------
    ''' Write joint force/torque readings and joint velocities to CSV files '''
    def write_data_files(self) -> None:
        ft_data = self.get_forces()
        if os.path.isfile(self.ft_file):
            with open(self.ft_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=ft_data.keys())
                writer.writerow(ft_data)
        else:
            print(f"Could not find file {self.ft_file}, exiting")
            sys.exit(0)
        
        joint_vels = self.get_joint_velocities()
        if os.path.isfile(self.vel_file):
            with open(self.vel_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=joint_vels.keys())
                writer.writerow(joint_vels)
        else:
            print(f"Could not find file {self.vel_file}, exiting")
            sys.exit(0)      
# -----------------------------------------------------------------------------------------------------------
    ''' Write current world frame positions of end effector to pos_file '''
    def write_wf_position(self) -> None:        
        with self.sim_lock:
            link_state = p.getLinkState(self.robot, self.ee_link_index)
            ee_position_wf = link_state[4]  # world frame position of the end effector
            ee_orientation_wf = p.getEulerFromQuaternion(link_state[5])  # world frame orientation of the end effector
        
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
    def go_to_desired_position(self, max_speed = 0.4, force=200) -> None:
        with self.sim_lock:
            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.setJointMotorControl2(self.robot, 
                                        joint_idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=self.next_pos[idx],
                                        force=force,
                                        maxVelocity=max_speed)
                                        # positionGain=0.2,
                                        # velocityGain=0.2)
# -----------------------------------------------------------------------------------------------------------                
    def get_joint_errors(self) -> list:
        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
        q = [joint_state[0] for joint_state in joint_states]
        qd = [joint_state[1] for joint_state in joint_states]
        
        print("Desired pos: ", self.next_pos)
        print("Current pos: ", q)
        pos_error = [self.next_pos[i] - q[i] for i in range(self.num_movable_joints)]
        vel_error = [self.next_vel[i] - qd[i] for i in range(self.num_movable_joints)]
        # print(f"pos_error: {pos_error}")
        return pos_error, vel_error
# -----------------------------------------------------------------------------------------------------------
    def stop_movement(self) -> None:
        with self.sim_lock:
            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.setJointMotorControl2(
                    bodyIndex=self.robot,
                    jointIndex=joint_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=0
                )
# -----------------------------------------------------------------------------------------------------------
    def apply_speed(self) -> None:
        with self.sim_lock:
            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.setJointMotorControl2(
                    bodyIndex=self.robot,
                    jointIndex=joint_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=float(self.next_vel[idx]),
                    force=200
                )

        print(*(f"J{i}: {vel: 8.5f}," for i, vel in zip(self.movable_joint_idxs, self.next_vel)))
# -----------------------------------------------------------------------------------------------------------
    def initial_reset(self) -> None:
        pos = [self.initial_x, self.initial_y, self.initial_z]
        with self.sim_lock:
            orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
            ik_vals = p.calculateInverseKinematics(self.robot,
                                                   self.ee_link_index,
                                                   pos,
                                                   orn,
                                                   maxNumIterations=1000,
                                                   residualThreshold=1e-5)
            
            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.resetJointState(self.robot, joint_idx, ik_vals[idx])
# -----------------------------------------------------------------------------------------------------------
    def do_startup(self, next_z) -> None:
        with self.sim_lock:
            orn = p.getQuaternionFromEuler([0, math.pi/2, 0])
            target = [self.initial_x, self.initial_y, next_z]
            ik_vals = p.calculateInverseKinematics(self.robot,
                                                   self.ee_link_index,
                                                   target,
                                                   orn)
        self.next_pos = ik_vals
# -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self, direction) -> None:
        with self.sim_lock:
            # wf = world frame
            wrist_pos_wf, wrist_orn_wf = p.getLinkState(self.robot, 
                                                        self.wrist_idx, 
                                                        computeForwardKinematics=True)[:2]
            ee_pos_wf, ee_orn_wf = p.getLinkState(self.robot, 
                                                  self.ee_link_index, 
                                                  computeForwardKinematics=True)[:2]
            
            inv_wrist_pos, inv_wrist_orn = p.invertTransform(wrist_pos_wf, wrist_orn_wf)
            ee_pos_wrist_frame, _ = p.multiplyTransforms(inv_wrist_pos, inv_wrist_orn, ee_pos_wf, ee_orn_wf)
            
            omega_wrist_frame = np.array([0.2 * direction, 0, 0])
            
            # rotate local w to world frame
            _, orn = p.getLinkState(self.robot, self.ee_link_index)[:2]
            R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3,3)
            omega_wf = R.dot(omega_wrist_frame)
            
            # build full 6-vector twist (v = [0; w_world])
            v = np.hstack((np.zeros(3), omega_wf))
            
            # Get current joint states & velocities
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
            q = [joint_state[0] for joint_state in joint_states]
            qd = [joint_state[1] for joint_state in joint_states] 
            qdd = [0.0]*self.num_movable_joints # acceleration is zero
            
            # Jacobian, world frame
            J_lin, J_ang = p.calculateJacobian(self.robot,
                                            self.wrist_idx,
                                            ee_pos_wrist_frame,
                                            q, 
                                            qd, 
                                            qdd)
            # stack into shape (6,n)
            J = np.vstack((np.array(J_lin), np.array(J_ang)))
            
            # solve for joint velocities dq = J+ . v
            dq = np.linalg.pinv(J).dot(v)  # shape (n,)
            
            self.next_vel = [dq[i] for i in range(self.num_movable_joints)]
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        while not self.shutdown_event.is_set():
            time.sleep(self.interval)
            # self.next_vel = [0.0] * self.num_movable_joints
            # self.next_pos = [0.0] * self.num_movable_joints
            
            with self.sim_lock:
                keys = p.getKeyboardEvents()
            if ord('q') in keys:
                self.pause_controls = True
                self.stop_movement()
            if ord('e') in keys:
                self.pause_controls = False
                
            if self.pause_controls == False:
                self.fsm.next_state()
                
                match self.mode:
                    case 'position':
                        self.go_to_desired_position()
                    case 'velocity':
                        self.apply_speed()
            # else:
            #     self.stop_movement()
            
            self.write_wf_position()
            self.write_data_files()
            
        print("Exiting controller thread...")
        sys.exit(0)
            
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)