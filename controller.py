import threading, json, os, csv, time, math, sys
from collections import defaultdict
import pybullet as p
import numpy as np
from simulation import Simulation
from fsm import FSM
from cvxopt import matrix, solvers
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
        self.initial_x = 0.7
        self.initial_y = 0.0
        self.initial_z = 0.15
        self.fsm = FSM(controller=self,
                       initial_x=self.initial_x,
                       initial_y=self.initial_y,
                       initial_z=self.initial_z)
        
        # self.wrist_joint = 7 # joint before the gripper
        self.finger_joints = [9, 10]

        self.revolute_joint_idx = []
        self.movable_joint_idxs = []
        self.all_joint_idx = []
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_names = []
        self.max_joint_velocities = []
        self.ee_link_index = None
        self.wrist_idx = None
        self.parent_map = defaultdict(list)
        self.get_idxs()

        self.num_revolute_joints = len(self.revolute_joint_idx)
        self.num_movable_joints = len(self.movable_joint_idxs)
        
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        # self.ft_types = ["raw", "comp", "contact"]
        self.ft_types = ["contact_ft"]
        self.ft_keys = [f"{name}_{type}" for type in self.ft_types for name in self.ft_names]
        self.ft = {ft_name: 0 for ft_name in self.ft_keys}
        self.ft_contact = [0.0] * 6
        
        self.joint_vels = {joint: 0 for joint in self.revolute_joint_idx}

        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.vel_file = os.path.join(self.data_path, "vel_data.csv")
        
        # State variables
        self.mode = 'velocity'
        self.pause_controls = False
        self.next_vel = [0.0] * self.num_movable_joints
        self.next_pos = [0.0] * self.num_movable_joints # ik_values
        self.next_pos_ee_xyz = [0.0, 0.0, 0.0]
        self.next_orn_ee = [0.0, 0.0, 0.0]
        self.vel_err = [0.0] * self.num_movable_joints
        self.speed_wf = [0.0] * 6
        self.speed_names = ["vx", "vy", "vz", "wx", "wy", "wz"]
        self.speed_types = ["v_wf"]
        self.speed_keys = [f"{name}_{type}" for type in self.speed_types for name in self.speed_names]
        self.speed = {name: 0.0 for name in self.speed_keys}
        
        self.joint_types = ["jointspeed"]
        self.joint_keys = [f"{name}_{type}" for type in self.joint_types for name in self.movable_joint_idxs]
        self.joint_speed = {name: 0.0 for name in self.speed_keys}
        
        self.initialize_plot_files()
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
                    self.joint_names.append(info[1].decode('utf-8'))
                    self.joint_lower_limits.append(info[8])
                    self.joint_upper_limits.append(info[9])
                    self.max_joint_velocities.append(info[11])
                if info[1].decode('utf-8') == "panda_grasptarget_hand":
                    self.ee_link_index = i
                # if info[1].decode('utf-8') == "panda_joint7":
                if info[1].decode('utf-8') == "panda_hand_joint":
                    self.wrist_idx = i
        
        if not self.ee_link_index:
            print("Could not find end effector link index")
            sys.exit(0)
            
        if not self.wrist_idx:
            print("Could not find wrist joint index")
            sys.exit(0)
# -----------------------------------------------------------------------------------------------------------
    ''' Initialize CSV file with headers, overwrites/deletes existing file with the same name '''
    def initialize_plot_files(self) -> None:
        # keys = self.ft_keys + self.speed_keys + self.joint_keys
        keys = self.ft_keys + self.speed_keys
        with open(self.ft_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            
        with open(self.vel_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.joint_vels.keys())
            writer.writeheader()
# -----------------------------------------------------------------------------------------------------------
    ''' Write joint force/torque readings and joint velocities to CSV files '''
    def write_data_files(self) -> None:
        for i, name in enumerate(self.speed_names):
            self.speed[f"{name}_v_wf"] = self.speed_wf[i]
            
        # for i, name in enumerate(self.movable_joint_idxs):
        #     self.joint_speed[f"{name}_jointspeed"] = self.next_vel[i]
            
        for i, name in enumerate(self.ft_names):
            self.ft[f"{name}_contact_ft"] = self.ft_contact[i]
            
        combined = self.ft | self.speed | self.joint_speed
        
        if os.path.isfile(self.ft_file):
            with open(self.ft_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=combined.keys())
                writer.writerow(combined)
        else:
            print(f"Could not find file {self.ft_file}, exiting")
            sys.exit(0)     
# -----------------------------------------------------------------------------------------------------------
    ''' Write current world frame positions of end effector to pos_file '''
    def write_wf_position(self) -> None:        
        with self.sim_lock:
            link_state = p.getLinkState(self.robot, self.ee_link_index)
            # ee_position_wf = link_state[4]
            # ee_orientation_wf = p.getEulerFromQuaternion(link_state[5])
            ee_position_wf = link_state[0]
            ee_orientation_wf = p.getEulerFromQuaternion(link_state[1])
            
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
    def get_contact_ft(self) -> None:
        with self.sim_lock:
            wrist_pos, wrist_ori = p.getLinkState(self.robot, self.wrist_idx)[:2]
            wrist_pos = np.array(wrist_pos)

            rot_wrist_world = np.array(p.getMatrixFromQuaternion(wrist_ori)).reshape(3, 3)
            rot_world_to_wrist = rot_wrist_world.T

            total_force_world = np.zeros(3)
            total_torque_world = np.zeros(3)
            
            contact_pts = p.getContactPoints(self.robot, self.sim.obj)
        for pt in contact_pts:
            link = pt[3]
            if link <= self.wrist_idx:
                print(f"Skipping contact point {link} < wrist_idx {self.wrist_idx}")
                continue

            contact_pos = np.array(pt[5])
            normal = np.array(pt[7])
            fn = pt[9]

            lateral_dir1 = np.array(pt[11])
            f_lat1 = pt[10]
            lateral_dir2 = np.array(pt[13])
            f_lat2 = pt[12]
            
            contact_force = fn * normal + f_lat1 * lateral_dir1 + f_lat2 * lateral_dir2
            
            r = contact_pos - wrist_pos
            torque = np.cross(r, contact_force)
            
            total_force_world += contact_force
            total_torque_world += torque
                
        total_force_local = rot_world_to_wrist @ total_force_world
        total_torque_local = rot_world_to_wrist @ total_torque_world

        self.ft_contact = list(total_force_local) + list(total_torque_local)
# -----------------------------------------------------------------------------------------------------------
    def stop_movement(self) -> None:
        with self.sim_lock:
            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.setJointMotorControl2(
                    bodyIndex = self.robot,
                    jointIndex = joint_idx,
                    controlMode = p.VELOCITY_CONTROL,
                    targetVelocity = 0.0)
# -----------------------------------------------------------------------------------------------------------
    def apply_speed(self) -> None:
        limit_hit = False
        with self.sim_lock:
            # # Check joint limits
            # for i, joint_idx in enumerate(self.movable_joint_idxs):
            #     pos = p.getJointState(self.robot, joint_idx)[0]
            #     margin = 0.02
                
            #     if pos < self.joint_lower_limits[i] + margin and self.next_vel[i] < 0:
            #         limit_hit = True
            #         self.next_vel[i] = 0.0
            #         print(f"Joint {joint_idx} {self.joint_names[joint_idx]} limit {self.joint_lower_limits[i]} hit at {pos}")
            #     elif pos > self.joint_upper_limits[i] - margin and self.next_vel[i] > 0:
            #         limit_hit = True
            #         self.next_vel[i] = 0.0
            #         print(f"Joint {joint_idx} {self.joint_names[joint_idx]} limit {self.joint_upper_limits[i]} hit at {pos}")
                
            #     if limit_hit:
            #         # v_cmd = [0.0] * self.num_movable_joints
            #         break
                    
            p.setJointMotorControlArray(
                bodyIndex = self.robot,
                jointIndices = self.movable_joint_idxs,
                controlMode = p.VELOCITY_CONTROL,
                targetVelocities = self.next_vel,
                velocityGains = [1.0] * self.num_movable_joints,
                # forces = [100] * self.num_movable_joints,
            )
        
        # print(f"vel_err: {self.vel_err}")
        # print(*(f"J{i}: {vel: 8.5f}," for i, vel in zip(self.movable_joint_idxs, self.next_vel)))
# -----------------------------------------------------------------------------------------------------------
    def reset_home_pos(self) -> None:
        pos = [self.initial_x, self.initial_y, self.initial_z]
        with self.sim_lock:            
            ik_vals = [(self.joint_upper_limits[i] + self.joint_lower_limits[i])/2 
                       for i in range(self.num_movable_joints)]
            
            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.resetJointState(self.robot, joint_idx, ik_vals[idx])
            
        self.next_pos = ik_vals
# -----------------------------------------------------------------------------------------------------------
    def do_move_pos(self, pos: list = [0, 0, 0], orn: list = [0, 0, 0], max_speed = 1.0) -> None:
        with self.sim_lock:
            Kp = 2.1
            
            ls = p.getLinkState(self.robot, self.ee_link_index)
            # pos_wf = np.array(ls[4])
            # orn_wf = np.array(ls[5])
            pos_wf = np.array(ls[0])
            orn_wf = np.array(ls[1])
            
            orn_des = p.getQuaternionFromEuler(orn)
            orn_err = p.getDifferenceQuaternion(orn_wf, orn_des)
            axis, angle = p.getAxisAngleFromQuaternion(orn_err)
            pd_ang = np.array(axis) * angle * Kp
            # pd_ang = np.array(p.getEulerFromQuaternion(orn_err)) * Kp
            pd_lin = (pos - pos_wf) * Kp
                        
        self.next_pos_ee_xyz = pos
        self.do_move_velocity(v_des=pd_lin.tolist(), w_des=pd_ang.tolist(), link='ee', wf=True)
# -----------------------------------------------------------------------------------------------------------        
    def get_pos_error(self, desired_pos = None, desired_orn = None):
        if desired_pos is None:
            desired_pos = self.next_pos_ee_xyz
        if desired_orn is None:
            desired_orn = self.next_orn_ee
        with self.sim_lock:
            ls = p.getLinkState(self.robot, self.ee_link_index)
            ee_pos = ls[0]
            pos_error = [desired_pos[i] - ee_pos[i] for i in range(3)]
            current_quat = ls[1]
            desired_quat = p.getQuaternionFromEuler(desired_orn)
            # quat_error = p.getDifferenceQuaternion(current_quat, desired_quat)
            # euler_error = p.getEulerFromQuaternion(quat_error)
            # euler_error = [angle * 180 / math.pi for angle in euler_error]
            orn_err = p.getDifferenceQuaternion(current_quat, desired_quat)
            axis, angle = p.getAxisAngleFromQuaternion(orn_err)
            euler_error = np.array(axis) * angle * 180 / math.pi
        
        return pos_error, euler_error
# -----------------------------------------------------------------------------------------------------------
    def do_move_velocity(self, v_des:list = [0, 0, 0], w_des: list = [0, 0, 0], link = 'wrist', wf = False) -> None:
        if link == 'wrist':
            link = self.wrist_idx
        else: # link == 'ee':
            link = self.ee_link_index
        
        with self.sim_lock:
            # wf = world frame
            ee_pos_wf, ee_orn_wf = p.getLinkState(self.robot, 
                                                  self.ee_link_index, 
                                                  computeForwardKinematics=True)[:2]
            
            # direction is just + or - 1
            # convert speed to world frame
            if wf:
                v_wf = np.array(v_des)
                w_wf = np.array(w_des)
            else:
                R_wrist2world = np.array(p.getMatrixFromQuaternion(ee_orn_wf)).reshape(3,3)
                v_wf = R_wrist2world.dot(np.array(v_des))
                w_wf = R_wrist2world.dot(np.array(w_des))                
            
            # debug direction
            start_pos = np.array(ee_pos_wf)
            end_pos = start_pos + v_wf * 0.5
            p.addUserDebugLine(start_pos,
                               end_pos,
                               lineColorRGB=[1, 0, 0],
                               lineWidth=3,
                               lifeTime=0.05
            )
            end_pos = start_pos + w_wf * 0.5
            p.addUserDebugLine(start_pos,
                               end_pos,
                               lineColorRGB=[0, 1, 0],
                               lineWidth=3,
                               lifeTime=0.05
            )
            
            speed_wf = np.hstack((v_wf, w_wf))
            self.speed_wf = speed_wf
            # print(f"speed_wf: {self.speed_wf}")
            
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
            q = [joint_state[0] for joint_state in joint_states]
            qd = [joint_state[1] for joint_state in joint_states]
            qdd = [qd[i] / (1.0/240.0) for i in range(self.num_movable_joints)]
            # zeros = [0.0] * self.num_movable_joints
            
            # Jacobian, world frame
            J_lin, J_ang = p.calculateJacobian(
                bodyUniqueId = self.robot,
                linkIndex = self.wrist_idx,
                # linkIndex = self.ee_link_index,
                localPosition = [0, 0, 0],
                objPositions = q, 
                objVelocities = qd,
                objAccelerations = qdd
            )

            J = np.vstack((np.array(J_lin), np.array(J_ang)))
            n = J.shape[1]
            
            alpha = 1e-2
            dt = 1.0 / 240.0
            y = 1e-2
            q = np.array(q)
            q_min = np.array(self.joint_lower_limits)
            q_max = np.array(self.joint_upper_limits)
            limits = np.array(self.max_joint_velocities)
            W = np.diag((1.0/limits)**2)
            
            # Quadratic cost: ½ dqᵀ H dq + gᵀ dq
            # H = J.T.dot(J) + alpha * np.eye(n)
            H = J.T.dot(J) + y * W 
            g = -J.T.dot(speed_wf)
            
            # Joint velocity bounds from position limits:
            #  q + dq·dt ≥ q_min  ⇒ dq ≥ (q_min - q)/dt
            #  q + dq·dt ≤ q_max  ⇒ dq ≤ (q_max - q)/dt
            lb = (q_min - q) / dt
            ub = (q_max - q) / dt
            
            # Inequality G dq ≤ h encapsulating both bounds:
            #  dq ≤ ub   →  I dq ≤ ub
            #  -dq ≤ -lb → -I dq ≤ -lb
            G = np.vstack(( np.eye(n), -np.eye(n) ))
            h = np.hstack(( ub, -lb ))
            
            # Convert to cvxopt matrices
            P = matrix(H)
            q_cvx = matrix(g)
            G_cvx = matrix(G)
            h_cvx = matrix(h)
            
            # Solve QP
            sol = solvers.qp(P, q_cvx, G_cvx, h_cvx, options={'show_progress': False})
            dq = np.array(sol['x']).flatten()
            
            self.next_vel = dq
            
            # limits = np.array(self.max_joint_velocities)
            # # Weighted pseudoinverse, bias movement towards joints with more freedom
            # W_inv = np.diag(1.0/limits**2)

            # # Damped pseudo inverse, avoid singularities
            # y = 1e-2
            # JWJt = J.dot(W_inv).dot(J.T) + y * np.eye(6)
            # J_wpinv = W_inv.dot(J.T).dot(np.linalg.inv(JWJt))
            # dq_primary = J_wpinv.dot(speed_wf)
            # q = np.array(q)
            # q_min = np.array(self.joint_lower_limits)
            # q_max = np.array(self.joint_upper_limits)
            
            # # Compute gradient to avoid joint limits
            # # qc   = (q_min + q_max)/2
            # # grad = (q - qc) / (q_max - q_min)**2
            # epsilon = 1e-3
            # d_high = np.maximum(q - q_max, epsilon)
            # d_low  = np.maximum(q_min - q, epsilon)
            # grad = -2.0/d_low**3 + 2.0/d_high**3
            # k = 0.4
            # dq_null = -k * grad
            
            # # Project in null space
            # P = np.eye(n) - J_wpinv.dot(J) 
            
            # dq = dq_primary + P.dot(dq_null)
            # over = np.abs(dq) > limits
            # if np.any(over):
            #     s = np.min(limits[over] / np.abs(dq[over]))
            #     dq *= s
                
            # self.next_vel = dq
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        while not self.shutdown_event.is_set():
            with self.sim_lock:
                keys = p.getKeyboardEvents()
            if ord('q') in keys:
                self.pause_controls = True
                self.stop_movement()
            if ord('e') in keys:
                self.pause_controls = False
            
            if self.pause_controls == False:
                self.get_contact_ft()
                self.fsm.next_state()
                
                match self.mode:
                    # case 'position':
                        # self.go_to_desired_position()
                    case 'velocity':
                        # self.do_move_velocity(type='linear', speed=[0.0, 0.0, 0.01])
                        self.apply_speed()
                    case _:
                        # print("Unknown mode, stopping movement...")
                        # self.stop_movement()
                        pass
            else:
                pass
            #     self.stop_movement()
            
            self.write_wf_position()
            self.write_data_files()
            p.stepSimulation()
            time.sleep(1.0/240.0)
            
        print("Exiting controller thread...")
        sys.exit(0)
            
if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)