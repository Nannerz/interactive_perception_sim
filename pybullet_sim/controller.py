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
        self.debug_lines = {}
        # self.prev_debug_lines = []
        self.interval = 0.01 # 10 ms
        self.thread_cntr_max = 500
        self.initial_x = 0.7
        self.initial_y = 0.0
        self.initial_z = 0.1
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
        self.get_idxs()

        self.num_revolute_joints = len(self.revolute_joint_idx)
        self.num_movable_joints = len(self.movable_joint_idxs)

        # ===============================================================================
        # Variables to write data/plot files
        self.ft_names = ["fx", "fy", "fz", "tx", "ty", "tz"]
        # self.ft_types = ["raw", "comp", "contact"]
        self.ft_types = ["contact_ft", "ema_ft", "feeling_ft"]
        self.ft_keys = [f"{name}_{type}" for type in self.ft_types for name in self.ft_names]
        self.ft = {ft_name: 0.0 for ft_name in self.ft_keys}
        self.ft_contact_wrist = [0.0] * 6
        self.ft_ema = [0.0] * 6
        self.ft_feeling = [0.0] * 6
        
        self.speed_world_frame = [0.0] * 6
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

        # ===============================================================================
        # Data files for plotting & writer thread
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_path, exist_ok=True)

        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.vel_file = os.path.join(self.data_path, "vel_data.csv")
        
        self.initialize_plot_files()
        self.data_queue = queue.Queue()
        csv_keys = self.ft_keys + self.speed_keys
        self.data_writer = DataWriter(self.data_queue, self.shutdown_event, self.ft_file, self.pos_file, csv_keys)
        
        # ===============================================================================
        # State variables
        self.mode = "velocity"
        self.prev_mode = "velocity"
        self.pause_sim = False
        self.next_dq = [0.0] * self.num_movable_joints
        self.prev_dq = [0.0] * self.num_movable_joints
        self.next_pos = [0.0] * self.num_movable_joints  # ik_values
        self.desired_pos_ee = np.zeros(3)
        self.desired_pos_wrist = np.zeros(3)
        self.desired_quat_ee = np.zeros(4)
        self.desired_quat_wrist = np.zeros(4)
        self.velocity_solver = osqp.OSQP()

    # -----------------------------------------------------------------------------------------------------------
    def get_idxs(self) -> None:
        with self.sim_lock:
            self.total_num_joints = p.getNumJoints(self.robot)
            for i in range(self.total_num_joints):
                info = p.getJointInfo(self.robot, i)
                self.all_joint_idx.append(i)

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
            self.speed[f"{name}_v_wf"] = self.speed_world_frame[i]
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
            ee_pos, ee_quat = p.getLinkState(self.robot, self.ee_link_index)[:2]
            roll, pitch, yaw = np.rad2deg(p.getEulerFromQuaternion(ee_quat))

        pos = {
            "x": ee_pos[0],
            "y": ee_pos[1],
            "z": ee_pos[2],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }

        self.data_queue.put({"type": "json", "data": pos})

    # -----------------------------------------------------------------------------------------------------------
    def get_contact_ft(self) -> None:
        with self.sim_lock:
            link_state = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)
            wrist_pos, wrist_quat = link_state[4], link_state[5]
            wrist_pos = np.array(wrist_pos)

            rot_wrist_world = np.array(p.getMatrixFromQuaternion(wrist_quat)).reshape(3, 3)
            rot_world_to_wrist = rot_wrist_world.T

            total_force_world = np.zeros(3)
            total_torque_world = np.zeros(3)

            contact_pts = p.getContactPoints(self.robot, self.sim.obj)

        if len(contact_pts) <= 0:
            self.ft_contact_wrist = [0.0] * 6
            return

        contact_pos = np.zeros(3)
        for pt in contact_pts:
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

                kwargs = {"lineColorRGB": linecolor, "lineWidth": 5, "lifeTime": 0}
                self.update_debug_line("force_wrist", start_pos, end_pos_force, kwargs)

                # torque_world = np.zeros(3)
                # torque_world[i] = total_torque_world[i]
                # end_pos_torque = start_pos + torque_world

                torque_local = np.zeros(3)
                torque_local[i] = total_torque_local[i]
                torque_world = rot_wrist_world @ torque_local
                end_pos_torque = start_pos + torque_world

                linecolor[i] = 0.25

                kwargs = {"lineColorRGB": linecolor, "lineWidth": 8, "lifeTime": 0}
                self.update_debug_line("torque_wrist", start_pos, end_pos_force, kwargs)

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
        with self.sim_lock:
            p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.movable_joint_idxs,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=self.next_dq,
                velocityGains=[1.0] * self.num_movable_joints,
                # forces = [10] * self.num_movable_joints,
            )

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
    def do_move_pos(self, pos: list = [0, 0, 0], euler_rad: list = [0, 0, 0]) -> None:
        Kp_lin = 0.7
        Kp_ang = 0.95
        
        pos_err, ang_err = self.get_pos_error(desired_pos=pos, desired_euler_radians=euler_rad)
        rad_err = np.deg2rad(ang_err)
                
        pd_ang = np.array(rad_err) * Kp_ang
        pd_lin = np.array(pos_err) * Kp_lin

        self.desired_pos_ee = np.array(pos)
        self.desired_quat_ee = np.array(p.getQuaternionFromEuler(euler_rad))
        self.do_move_velocity(v_des=pd_lin.tolist(), w_des=pd_ang.tolist(), link="ee", world_frame=True)

    # -----------------------------------------------------------------------------------------------------------
    def get_pos_error(self, desired_pos, desired_euler_radians):
        with self.sim_lock:
            ee_pos, ee_quat = p.getLinkState(self.robot, self.ee_link_index)[:2]
            
            desired_quat = p.getQuaternionFromEuler(desired_euler_radians)
            quat_error = p.getDifferenceQuaternion(ee_quat, desired_quat)
            axis, angle = p.getAxisAngleFromQuaternion(quat_error)
            
        ang_error = np.rad2deg(np.array(axis) * angle)
        pos_error = [desired_pos[i] - ee_pos[i] for i in range(3)]

        return pos_error, ang_error

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
        #     with self.sim_lock:
        #         link_state = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)
        #     wrist_pos_wf, wrist_orn_wf = link_state[4], link_state[5]
            
        #     start_pos = wrist_pos_wf
            
        #     end_pos1 = left_tip_pos_wf
        #     linecolor1 = [1, 0, 0]
        #     kwargs1 = {"lineColorRGB": linecolor1, "lineWidth": 5, "lifeTime": 0}
        #     self.update_debug_line("left_tip", start_pos, end_pos1, kwargs1)
            
        #     end_pos2 = right_tip_pos_wf
        #     linecolor2 = [0, 1, 0]
        #     kwargs2 = {"lineColorRGB": linecolor2, "lineWidth": 5, "lifeTime": 0}
        #     self.update_debug_line("right_tip", start_pos, end_pos2, kwargs2)

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


    def get_wrist_pos(self):
        with self.sim_lock:
            wrist_pos, wrist_quat = p.getLinkState(self.robot, self.wrist_idx)[:2]
            wrist_pos = np.array(wrist_pos)
            wrist_quat = np.array(wrist_quat)

        return wrist_pos, wrist_quat
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
    def do_move_velocity(self, v_des, w_des, link="wrist", world_frame=False) -> None:
        if link == "wrist":
            link = self.wrist_idx
        else:  # link == 'ee':
            link = self.ee_link_index

        # wf = world frame
        with self.sim_lock:
            # cur_pos_world_frame, cur_quat_world_frame = p.getLinkState(self.robot, self.ee_link_index, computeForwardKinematics=True)[:2]
            cur_pos_world_frame, cur_quat_world_frame = p.getLinkState(self.robot, link, computeForwardKinematics=True)[:2]
            # wrist_pos_world_frame, wrist_orn_world_frame = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)[:2]
            # ee_pos_world_frame, ee_quat_world_frame = p.getLinkState(self.robot, self.ee_link_index, computeForwardKinematics=True)[:2]

        # convert speed to world frame
        if world_frame:
            v_world_frame = np.array(v_des)
            w_world_frame = np.array(w_des)
        else:
            with self.sim_lock:
                R_wrist2world = np.array(p.getMatrixFromQuaternion(cur_quat_world_frame)).reshape(3, 3)
            v_world_frame = R_wrist2world.dot(np.array(v_des))
            w_world_frame = R_wrist2world.dot(np.array(w_des))
        
        # # Position feedback
        # pos_err = np.zeros(3)
        # if link == 'wrist':
        #     pos_err = self.desired_pos_wrist - np.array(cur_pos_world_frame)
        #     self.desired_pos_wrist = self.desired_pos_wrist + np.array(v_world_frame) * self.interval
        # else:
        #     pos_err = self.desired_pos_ee - np.array(cur_pos_world_frame)
        #     self.desired_pos_ee = self.desired_pos_ee + np.array(v_world_frame) * self.interval

        # Kp_lin = 0.12
        # v_world_frame = v_world_frame + Kp_lin * pos_err

        # # Rotational feedback
        # if link == 'wrist':
        #     prev_quat = self.desired_quat_wrist
        # else:
        #     prev_quat = self.desired_quat_ee

        # if np.linalg.norm(w_world_frame) > 1e-6:
        #     axis_w = w_world_frame / np.linalg.norm(w_world_frame)
        #     angle_w = np.linalg.norm(w_world_frame) * self.interval
        #     delta_quat = p.getQuaternionFromAxisAngle(axis_w.tolist(), angle_w)
        #     desired_quat = p.multiplyTransforms([0,0,0], delta_quat, [0,0,0], prev_quat)[1]
        # else:
        #     desired_quat = prev_quat

        # # normalize and store desired quaternion
        # desired_quat = np.array(desired_quat)
        # desired_quat /= np.linalg.norm(desired_quat)
        # if link == 'wrist':
        #     self.desired_quat_wrist = desired_quat.tolist()
        # else:
        #     self.desired_quat_ee = desired_quat.tolist()

        # # compute error quaternion and corrective angular velocity
        # quat_error = p.getDifferenceQuaternion(cur_quat_world_frame, desired_quat)
        # axis_err, angle_err = p.getAxisAngleFromQuaternion(quat_error)
        # if angle_err > np.pi:
        #     angle_err -= 2 * np.pi
        # elif angle_err < -np.pi:
        #     angle_err += 2 * np.pi

        # Kp_ang = 0.2
        # if abs(angle_err) < 1e-6 or np.linalg.norm(axis_err) < 1e-6:
        #     w_correction = np.zeros(3)
        # else:
        #     axis_n = np.array(axis_err) / np.linalg.norm(axis_err)
        #     w_correction = axis_n * (angle_err * Kp_ang)

        # # total commanded angular velocity
        # w_world_frame = w_world_frame + w_correction
                
        self.speed_world_frame = np.hstack((v_world_frame, w_world_frame))
        
        with self.sim_lock:
            R_wrist2world = np.array(p.getMatrixFromQuaternion(cur_quat_world_frame)).reshape(3, 3)
        v_wrist = R_wrist2world.T.dot(v_world_frame)
        w_wrist = R_wrist2world.T.dot(w_world_frame)
        self.speed_wrist[:3] = v_wrist
        self.speed_wrist[3:] = w_wrist

        # Draw debug lines showing the desired velocities
        if self.draw_debug:
            start_pos = np.array(cur_pos_world_frame)
            end_pos = start_pos + v_world_frame * 0.5
            kwargs = {"lineColorRGB": [0, 0, 1], "lineWidth": 3, "lifeTime": 0}
            self.update_debug_line("v_wf", start_pos, end_pos, kwargs)

            end_pos = start_pos + w_world_frame * 0.5
            kwargs = {"lineColorRGB": [1, 0, 0], "lineWidth": 3, "lifeTime": 0}
            self.update_debug_line("w_wf", start_pos, end_pos, kwargs)

        # Get current jacobian
        with self.sim_lock:
            joint_states = p.getJointStates(self.robot, self.movable_joint_idxs)
            q = [joint_state[0] for joint_state in joint_states]
            qd = [joint_state[1] for joint_state in joint_states]
            qdd = [qd[i] / self.interval for i in range(self.num_movable_joints)]

            # Jacobian, world frame
            J_lin, J_ang = p.calculateJacobian(
                bodyUniqueId=self.robot,
                linkIndex=link,
                localPosition=[0, 0, 0],
                objPositions=q,
                objVelocities=qd,
                objAccelerations=qdd,
            )

        J = np.vstack((np.array(J_lin), np.array(J_ang)))
        n = J.shape[1]

        dt = self.interval
        y = 1e-2
        q = np.array(q)
        q_min = np.array(self.joint_lower_limits)
        q_max = np.array(self.joint_upper_limits)
        dq_limits = np.array(self.max_joint_velocities)
        W = np.diag((1.0 / dq_limits) ** 2)

        # Quadratic cost: 0.5 * dq.T H dq + g.T dq
        H = J.T.dot(J) + y * W
        g = -J.T.dot(self.speed_world_frame)

        # Calculate joint velocity limit based on current position and distance to limits
        dq_q_lower = np.where(q > q_min, (q_min - q) / dt, 0)
        dq_q_upper = np.where(q < q_max, (q_max - q) / dt, 0)

        dq_upper_bound = np.minimum(dq_q_upper, dq_limits)
        dq_lower_bound = np.maximum(dq_q_lower, -dq_limits)

        # G dq <= h
        #  dq <= ub     ->    I dq <= ub
        # -dq <= -lb   ->   -I dq <= -lb
        G = np.vstack((np.eye(n), -np.eye(n)))
        h = np.hstack((dq_upper_bound, -dq_lower_bound))

        # use osqp, need to convert numpy arrays to sparse matrices
        P_osqp = sp.csc_matrix((H + H.T) / 2)  # Ensure symmetry
        q_osqp = g
        G_osqp = sp.csc_matrix(G)
        l_osqp = -np.inf * np.ones(h.shape)
        u_osqp = h

        self.velocity_solver.setup(P=P_osqp, q=q_osqp, A=G_osqp, l=l_osqp, u=u_osqp, verbose=False)
        res = self.velocity_solver.solve()
        dq = res.x

        # Enforce max joint velocities
        over = np.abs(dq) > dq_limits
        if np.any(over):
            s = np.min(dq_limits[over] / np.abs(dq[over]))
            dq *= s
        
        # Keep fingers open
        dq[-2:] = self.max_joint_velocities[-2:]
        
        self.next_dq = dq

    # -----------------------------------------------------------------------------------------------------------
    def update_debug_line(self, line_name: str, start_pos: List[float], end_pos: List[float], kwargs: dict) -> None:
        if line_name in self.debug_lines:
            kwargs["replaceItemUniqueId"] = self.debug_lines[line_name]
        
        with self.sim_lock:
            line = p.addUserDebugLine(start_pos, end_pos, **kwargs)
        
        if line_name not in self.debug_lines:
            self.debug_lines[line_name] = line

    # -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        next_thread_time = time.perf_counter()
        interval = self.interval
        thread_cntr = 0
        thread_times = []
        
        while not self.shutdown_event.is_set():
            thread_start_time = time.perf_counter()
            prev_time = thread_start_time
            
            timediff = []
            timenames = []
            
            if self.do_timers:
                cur_time = time.perf_counter()
                timediff.append(cur_time - prev_time)
                prev_time = cur_time
                timenames.append("vis_off")
            
            # Press Q to pause/unpause simulation at runtime
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
                cur_time = time.perf_counter()
                timediff.append(cur_time - prev_time)
                prev_time = cur_time
                timenames.append("input_check")

            if self.pause_sim == False:
                self.get_contact_ft()
                if self.do_timers:
                    cur_time = time.perf_counter()
                    timediff.append(cur_time - prev_time)
                    prev_time = cur_time
                    timenames.append("get_contact_ft")

                self.fsm.next_state()
                if self.do_timers:
                    cur_time = time.perf_counter()
                    timediff.append(cur_time - prev_time)
                    prev_time = cur_time
                    timenames.append("fsm_next_state")

                match self.mode:
                    case "position":
                        pass
                        # self.go_to_desired_position()
                    case "velocity":
                        if self.prev_mode != "velocity":
                            self.desired_pos_ee, self.desired_quat_ee = p.getLinkState(self.robot, self.ee_link_index, computeForwardKinematics=True)[:2]
                            self.desired_pos_wrist, self.desired_quat_wrist = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)[:2]
                        self.apply_speed()
                    case _:
                        # print("Unknown mode, stopping movement...")
                        # self.stop_movement()
                        pass
                
                self.prev_mode = self.mode

                if self.do_timers:
                    cur_time = time.perf_counter()
                    timediff.append(cur_time - prev_time)
                    prev_time = cur_time
                    timenames.append("apply_speed")
                
                self.write_wf_position()
                self.write_data_files()
                
                if self.do_timers:
                    cur_time = time.perf_counter()
                    timediff.append(cur_time - prev_time)
                    prev_time = cur_time
                    timenames.append("write_data_files")
            
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
