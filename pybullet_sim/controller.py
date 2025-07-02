import threading, json, os, csv, time, math, sys
from typing import Tuple, List, Literal
from collections import defaultdict
import pybullet as p
import numpy as np
from simulation import Simulation
from fsm import FSM
from data_writer import DataWriter
# from cvxopt import matrix, solvers
import scipy.sparse as sp
import osqp
import queue


# -----------------------------------------------------------------------------------------------------------
class Controller(threading.Thread):
    def __init__(
        self,
        sim: Simulation,
        shutdown_event: threading.Event,
        draw_debug: bool = False,
        do_timers: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.daemon = True
        
        self.sim = sim
        self.robot = sim.robot
        self.sim_lock = sim.sim_lock
        self.shutdown_event = shutdown_event
        self.draw_debug = draw_debug
        self.do_timers = do_timers
        # self.debug_lines = []
        # self.prev_debug_lines = []
        self.interval = 0.01 # 10 ms
        self.thread_cntr_max = 50
        self.initial_x = 0.7
        self.initial_y = 0.0
        self.initial_z = 0.12
        self.fsm = FSM(
            controller=self,
            initial_x=self.initial_x,
            initial_y=self.initial_y,
            initial_z=self.initial_z,
        )

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
        self.right_idx = None
        self.left_idx = None
        self.parent_map = defaultdict(list)
        self.get_idxs()

        self.num_revolute_joints = len(self.revolute_joint_idx)
        self.num_movable_joints = len(self.movable_joint_idxs)

        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        # self.ft_types = ["raw", "comp", "contact"]
        self.ft_types = ["contact_ft", "ema_ft", "feeling_ft"]
        self.ft_keys = [f"{name}_{type}" for type in self.ft_types for name in self.ft_names]
        self.ft = {ft_name: 0.0 for ft_name in self.ft_keys}
        self.ft_contact_wrist = [0.0] * 6
        self.ft_ema = [0.0] * 6
        self.ft_feeling = [0.0] * 6

        # self.joint_vels = {joint: 0 for joint in self.revolute_joint_idx}

        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_path, exist_ok=True)

        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.vel_file = os.path.join(self.data_path, "vel_data.csv")

        # State variables
        self.mode = "velocity"
        self.pause_sim = False
        self.next_vel = [0.0] * self.num_movable_joints
        self.next_pos = [0.0] * self.num_movable_joints  # ik_values
        self.next_pos_ee_xyz = [0.0, 0.0, 0.0]
        self.next_orn_ee = [0.0, 0.0, 0.0]
        self.vel_err = [0.0] * self.num_movable_joints
        self.speed_wf = [0.0] * 6
        self.speed_wrist = [0.0] * 6
        self.speed_names = ["vx", "vy", "vz", "wx", "wy", "wz"]
        self.speed_types = ["v_wf", "v_wrist"]
        self.speed_keys = [f"{name}_{type}" for type in self.speed_types for name in self.speed_names]
        self.speed = {name: 0.0 for name in self.speed_keys}

        self.joint_types = ["jointspeed"]
        self.joint_keys = [
            f"{name}_{type}"
            for type in self.joint_types
            for name in self.movable_joint_idxs
        ]
        self.joint_speed = {name: 0.0 for name in self.speed_keys}

        self.initialize_plot_files()
        self.data_queue = queue.Queue()
        csv_keys = self.ft_keys + self.speed_keys
        self.data_writer = DataWriter(self.data_queue, self.shutdown_event, self.ft_file, self.pos_file, csv_keys)

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
                    self.joint_names.append(info[1].decode("utf-8"))
                    self.joint_lower_limits.append(info[8])
                    self.joint_upper_limits.append(info[9])
                    self.max_joint_velocities.append(info[11])
                if info[1].decode("utf-8") == "panda_grasptarget_hand":
                    self.ee_link_index = i
                if info[1].decode("utf-8") == "panda_hand_joint":
                    self.wrist_idx = i
                if info[1].decode("utf-8") == "panda_finger_joint1":
                    self.right_idx = i
                if info[1].decode("utf-8") == "panda_finger_joint2":
                    self.left_idx = i

        get_idxs = (self.right_idx, self.left_idx, self.wrist_idx, self.ee_link_index)

        if not self.ee_link_index:
            print("Could not find end effector link index")
            sys.exit(0)

        if not self.wrist_idx or not self.right_idx or not self.left_idx:
            print("Could not find wrist or finger index")
            sys.exit(0)

    # -----------------------------------------------------------------------------------------------------------
    """ Initialize CSV file with headers, overwrites/deletes existing file with the same name """

    def initialize_plot_files(self) -> None:
        keys = self.ft_keys + self.speed_keys
        with open(self.ft_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

    # -----------------------------------------------------------------------------------------------------------
    """ Write joint force/torque readings and joint velocities to CSV files """

    def write_data_files(self) -> None:
        for i, name in enumerate(self.speed_names):
            self.speed[f"{name}_v_wf"] = self.speed_wf[i]
            self.speed[f"{name}_v_wrist"] = self.speed_wrist[i]

        for i, name in enumerate(self.ft_names):
            self.ft[f"{name}_contact_ft"] = self.ft_contact_wrist[i]
            self.ft[f"{name}_ema_ft"] = self.ft_ema[i]
            self.ft[f"{name}_feeling_ft"] = self.ft_feeling[i]

        combined = self.ft | self.speed
        
        self.data_queue.put({"type": "csv", "data": combined})

    # -----------------------------------------------------------------------------------------------------------
    """ Write current world frame positions of end effector to pos_file """

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
            "roll": ee_orientation_wf[0] * 180.0 / math.pi,
            "pitch": ee_orientation_wf[1] * 180.0 / math.pi,
            "yaw": ee_orientation_wf[2] * 180.0 / math.pi,
        }

        self.data_queue.put({"type": "json", "data": pos})

    # -----------------------------------------------------------------------------------------------------------
    def get_contact_ft(self) -> None:
        with self.sim_lock:
            link_state = p.getLinkState(
                self.robot, self.wrist_idx, computeForwardKinematics=True
            )
            # wrist_pos, wrist_quat = p.getLinkState(self.robot, self.wrist_idx)[:2]
            wrist_pos, wrist_quat = link_state[4], link_state[5]
            wrist_pos = np.array(wrist_pos)

            rot_wrist_world = np.array(p.getMatrixFromQuaternion(wrist_quat)).reshape(3, 3)
            rot_world_to_wrist = rot_wrist_world.T

            total_force_world = np.zeros(3)
            total_torque_world = np.zeros(3)

            contact_pts = p.getContactPoints(self.robot, self.sim.obj)

        cntr = 0
        if len(contact_pts) <= 0:
            self.ft_contact_wrist = [0.0] * 6
            return

        contact_pos = np.zeros(3)
        for pt in contact_pts:
            cntr += 1
            link = pt[3]
            contact_pos = np.array(pt[5])
            normal = np.array(pt[7])
            fn = pt[9]

            lateral_dir1 = np.array(pt[11])
            f_lat1 = pt[10]
            lateral_dir2 = np.array(pt[13])
            f_lat2 = pt[12]

            if link <= self.wrist_idx:
                continue

            contact_force = fn * normal + f_lat1 * lateral_dir1 + f_lat2 * lateral_dir2

            r = contact_pos - wrist_pos
            torque = np.cross(r, contact_force)

            total_force_world += contact_force
            total_torque_world += torque

        total_force_local = rot_world_to_wrist @ total_force_world
        total_torque_local = rot_world_to_wrist @ total_torque_world

        self.ft_contact_wrist = list(total_force_local) + list(total_torque_local)

        # Draw debug line
        if self.draw_debug:
            # debug direction
            start_pos = contact_pos

            for i in range(3):
                # force_world = np.zeros(3)
                # force_world[i] = total_force_world[i]
                # end_pos_force = start_pos + force_world

                force_local = np.zeros(3)
                force_local[i] = total_force_local[i]
                force_world = rot_wrist_world @ force_local
                end_pos_force = start_pos + force_world

                linecolor = [0.0, 0.0, 0.0]
                linecolor[i] = 0.75

                with self.sim_lock:
                    line = p.addUserDebugLine(
                        start_pos,
                        end_pos_force,
                        lineColorRGB=linecolor,
                        lineWidth=5,
                        lifeTime=0,
                    )
                # self.debug_lines.append(line)

                # torque_world = np.zeros(3)
                # torque_world[i] = total_torque_world[i]
                # end_pos_torque = start_pos + torque_world

                torque_local = np.zeros(3)
                torque_local[i] = total_torque_local[i]
                torque_world = rot_wrist_world @ torque_local
                end_pos_torque = start_pos + torque_world

                linecolor[i] = 0.25

                with self.sim_lock:
                    line = p.addUserDebugLine(
                        start_pos,
                        end_pos_torque,
                        lineColorRGB=linecolor,
                        lineWidth=8,
                        lifeTime=0,
                    )
                # self.debug_lines.append(line)

    # -----------------------------------------------------------------------------------------------------------
    def stop_movement(self) -> None:
        with self.sim_lock:
            p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.movable_joint_idxs,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[0.0] * self.num_movable_joints,
            )

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
                bodyIndex=self.robot,
                jointIndices=self.movable_joint_idxs,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=self.next_vel,
                velocityGains=[1.0] * self.num_movable_joints,
                # forces = [100] * self.num_movable_joints,
            )

        # print(f"vel_err: {self.vel_err}")
        # print(*(f"J{i}: {vel: 8.5f}," for i, vel in zip(self.movable_joint_idxs, self.next_vel)))

    # -----------------------------------------------------------------------------------------------------------
    def reset_home_pos(self) -> None:
        pos = [self.initial_x, self.initial_y, self.initial_z]
        with self.sim_lock:
            ik_vals = [
                (self.joint_upper_limits[i] + self.joint_lower_limits[i]) / 2
                for i in range(self.num_movable_joints)
            ]

            for idx, joint_idx in enumerate(self.movable_joint_idxs):
                p.resetJointState(self.robot, joint_idx, ik_vals[idx])

        self.next_pos = ik_vals

    # -----------------------------------------------------------------------------------------------------------
    def open_gripper(self) -> None:
        with self.sim_lock:
            p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.finger_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0.04] * len(self.finger_joints),
            )

    # -----------------------------------------------------------------------------------------------------------
    def check_gripper_pos(self) -> Literal[True, False]:
        with self.sim_lock:
            gripper_pos = [
                p.getJointState(self.robot, joint_idx)[0]
                for joint_idx in self.finger_joints
            ]
            if all(pos >= 0.039 for pos in gripper_pos):
                return True
            else:
                return False

    # -----------------------------------------------------------------------------------------------------------
    def do_move_pos(
        self, pos: list = [0, 0, 0], orn: list = [0, 0, 0], max_speed=1.0
    ) -> None:
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
        self.do_move_velocity(v_des=pd_lin.tolist(), w_des=pd_ang.tolist(), link="ee", wf=True)

    # -----------------------------------------------------------------------------------------------------------
    def get_pos_error(self, desired_pos=None, desired_orn=None):
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
    def get_fingertip_pos(self):
        finger_length = 0.055
        finger_side_offset = 0.008
        right_tip_local = [0, finger_side_offset, finger_length]
        left_tip_local = [0, -finger_side_offset, finger_length]

        with self.sim_lock:
            link_state = p.getLinkState(self.robot, self.left_idx, computeForwardKinematics=True)

            left_base_pos_wf, left_base_orn_wf = link_state[4], link_state[5]
            left_tip_pos_wf, _ = p.multiplyTransforms(
                left_base_pos_wf, left_base_orn_wf, left_tip_local, [0, 0, 0, 1]
            )

            link_state = p.getLinkState(self.robot, self.right_idx, computeForwardKinematics=True)
            right_base_pos_wf, right_base_orn_wf = link_state[4], link_state[5]
            right_tip_pos_wf, _ = p.multiplyTransforms(
                right_base_pos_wf, right_base_orn_wf, right_tip_local, [0, 0, 0, 1]
            )
            # if self.draw_debug:
            #     link_state = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)
            #     wrist_pos_wf, wrist_orn_wf = link_state[4], link_state[5]

            #     p.addUserDebugLine(wrist_pos_wf, left_tip_pos_wf,  [1, 0, 0], 5, lifeTime=0.1)
            #     p.addUserDebugLine(wrist_pos_wf, right_tip_pos_wf, [0, 1, 0], 5, lifeTime=0.1)

        return right_tip_pos_wf, left_tip_pos_wf

    # -----------------------------------------------------------------------------------------------------------
    def get_fingertip_pos_wrist_frame(self):
        right_tip_pos_wf, left_tip_pos_wf = self.get_fingertip_pos()
        with self.sim_lock:
            link_state = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)
            wrist_pos_wf, wrist_orn_wf = link_state[4], link_state[5]

            inv_pos, inv_orn = p.invertTransform(wrist_pos_wf, wrist_orn_wf)

            right_tip_pos_wrist_frame, _ = p.multiplyTransforms(
                inv_pos, inv_orn, right_tip_pos_wf, [0, 0, 0, 1]
            )

            left_tip_pos_wrist_frame, _ = p.multiplyTransforms(
                inv_pos, inv_orn, left_tip_pos_wf, [0, 0, 0, 1]
            )

        return right_tip_pos_wrist_frame, left_tip_pos_wrist_frame

    # -----------------------------------------------------------------------------------------------------------
    # def calc_spin_around(self, w_des, right_finger = True):
    #     w_des_local = np.array(w_des)

    #     right_tip_pos_wrist, left_tip_pos_wrist = self.get_fingertip_pos_wrist_frame()
    #     r = right_tip_pos_wrist if right_finger else left_tip_pos_wrist
    #     r = np.array(r)

    #     v_des_local = -np.cross(w_des_local, r)
    #     print(f"r: {r}")
    #     print(f"spin around v_des: {v_des_local}, w_des: {w_des_local}")

    #     self.do_move_velocity(v_des=v_des_local.tolist(), w_des=w_des_local.tolist(), link='wrist', wf=False)
    # -----------------------------------------------------------------------------------------------------------
    def do_move_velocity(self, v_des, w_des, link="wrist", wf=False) -> None:

        if link == "wrist":
            link = self.wrist_idx
        else:  # link == 'ee':
            link = self.ee_link_index

        # wf = world frame
        with self.sim_lock:
            ee_pos_wf, ee_orn_wf = p.getLinkState(self.robot, self.ee_link_index, computeForwardKinematics=True)[:2]

        # direction is just + or - 1
        # convert speed to world frame
        self.speed_wrist = [0.0] * 6
        if wf:
            v_wf = np.array(v_des)
            w_wf = np.array(w_des)
        else:
            self.speed_wrist[0:3] = v_des
            self.speed_wrist[3:6] = w_des
            with self.sim_lock:
                R_wrist2world = np.array(p.getMatrixFromQuaternion(ee_orn_wf)).reshape(3, 3)
            v_wf = R_wrist2world.dot(np.array(v_des))
            w_wf = R_wrist2world.dot(np.array(w_des))

        # debug direction
        with self.sim_lock:
            if self.draw_debug:
                start_pos = np.array(ee_pos_wf)
                end_pos = start_pos + v_wf * 0.5
                line = p.addUserDebugLine(
                    start_pos, end_pos, lineColorRGB=[1, 0, 0], lineWidth=3, lifeTime=0
                )
                # self.debug_lines.append(line)

                end_pos = start_pos + w_wf * 0.5
                line = p.addUserDebugLine(
                    start_pos, end_pos, lineColorRGB=[0, 1, 0], lineWidth=3, lifeTime=0
                )
                # self.debug_lines.append(line)

        speed_wf = np.hstack((v_wf, w_wf))
        self.speed_wf = speed_wf
        # print(f"speed_wf: {self.speed_wf}")

        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
            q = [joint_state[0] for joint_state in joint_states]
            qd = [joint_state[1] for joint_state in joint_states]
            qdd = [qd[i] / (1.0 / 240.0) for i in range(self.num_movable_joints)]
            # zeros = [0.0] * self.num_movable_joints

            # Jacobian, world frame
            J_lin, J_ang = p.calculateJacobian(
                bodyUniqueId=self.robot,
                linkIndex=self.wrist_idx,
                # linkIndex = self.ee_link_index,
                localPosition=[0, 0, 0],
                objPositions=q,
                objVelocities=qd,
                objAccelerations=qdd,
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
        W = np.diag((1.0 / limits) ** 2)

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
        G = np.vstack((np.eye(n), -np.eye(n)))
        h = np.hstack((ub, -lb))

        # Convert to cvxopt matrices
        # P = matrix(H)
        # q_cvx = matrix(g)
        # G_cvx = matrix(G)
        # h_cvx = matrix(h)

        # Solve QP
        # Use OSQP for faster QP solving if available, else fallback to cvxopt
        # Convert numpy arrays to sparse matrices for OSQP
        P_osqp = sp.csc_matrix((H + H.T) / 2)  # Ensure symmetry
        q_osqp = g
        G_osqp = sp.csc_matrix(G)
        l_osqp = -np.inf * np.ones(h.shape)
        u_osqp = h

        prob = osqp.OSQP()
        prob.setup(P=P_osqp, q=q_osqp, A=G_osqp, l=l_osqp, u=u_osqp, verbose=False)
        res = prob.solve()
        dq = res.x

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
        next_thread_time = time.perf_counter()
        prev_sleep_time = next_thread_time
        interval = self.interval
        thread_cntr = 0
        thread_times = []
        
        while not self.shutdown_event.is_set():
            with self.sim_lock:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            timediff = []
            timenames = []
            thread_start_time = time.perf_counter()
                
            with self.sim_lock: 
                qKey = ord("q")
                keys = p.getKeyboardEvents()
                if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                    self.pause_sim = not self.pause_sim
                    if self.pause_sim:
                        p.setRealTimeSimulation(0)
                    else:
                        p.setRealTimeSimulation(1)
                    print(f"Got input to {'pause' if self.pause_sim else 'unpause'} simulation")
            
            if self.do_timers:
                timediff.append(time.perf_counter() - thread_start_time)
                timenames.append("input_check")

            if self.pause_sim == False:
                with self.sim_lock:
                    p.removeAllUserDebugItems()
                    
                self.get_contact_ft()
                if self.do_timers:
                    timediff.append(time.perf_counter() - timediff[-1] - thread_start_time)
                    timenames.append("get_contact_ft")

                self.fsm.next_state()
                if self.do_timers:
                    timediff.append(time.perf_counter() - timediff[-1] - thread_start_time)
                    timenames.append("fsm_next_state")

                match self.mode:
                    case "position":
                        pass
                        # self.go_to_desired_position()
                    case "velocity":
                        # self.do_move_velocity(type='linear', speed=[0.0, 0.0, 0.01])
                        self.apply_speed()
                    case _:
                        # print("Unknown mode, stopping movement...")
                        # self.stop_movement()
                        pass

                if self.do_timers:
                    timediff.append(time.perf_counter() - timediff[-1] - thread_start_time)
                    timenames.append("apply_speed")
                
                self.write_wf_position()
                self.write_data_files()
                
                if self.do_timers:
                    timediff.append(time.perf_counter() - timediff[-1] - thread_start_time)
                    timenames.append("write_data_files")
            
            else:
                pass
            
            with self.sim_lock:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            
            current_time = time.perf_counter()
            elapsed_time = current_time - thread_start_time
            if elapsed_time > interval:
                print(f"Controller thread is running slow! Thread time: {elapsed_time:.6f}s") 
            if self.do_timers:
                thread_cntr += 1
                timediff.append(elapsed_time)
                timenames.append("thread_elapsed_time")
                thread_times.append(timediff)
            
                if len(thread_times) > self.thread_cntr_max:
                    thread_cntr = 0
                    avgs = [ sum(i) / len(i) for i in zip(*thread_times) ]
                    print("Average thread times:", [f"{name}: {v:.6f}s" for name, v in zip(timenames, avgs)])
                    thread_times = []
            
            next_thread_time += interval
            sleep_time = next_thread_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Controller thread is running behind schedule by {-sleep_time:.6f}s")
                if len(thread_times) > 0:
                    print(f"Thread times: ", [f"{name}: {t:.6f}s" for name, t in zip(timenames, thread_times[-1])])
                next_thread_time = time.perf_counter()

        print("Exiting controller thread...")
        sys.exit(0)


if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)
