import threading, os, csv, time, math, sys
import pybullet as p
import numpy as np
from numpy.typing import NDArray
from simulation import Simulation
from fsm import FSM
from data_writer import DataWriter
import scipy.sparse as sp # type: ignore
import osqp # type: ignore
import queue
from typing import Any
p: Any = p
osqp: Any = osqp


# -----------------------------------------------------------------------------------------------------------
class Controller(threading.Thread):
    def __init__(
        self,
        sim: Simulation,
        shutdown_event: threading.Event,
        draw_debug: bool = False,
        do_timers: bool = False,
        initial_robot_conf: dict[str, list[float]] = {"pos": [0.7, 0, 0.1], "orn": [0, 90.0 * math.pi / 180.0, 0]},
        **kwargs: Any,
    ) -> None:

        super().__init__(**kwargs)
        self.daemon = True
        
        self.sim = sim
        self.robot = sim.robot
        self.sim_lock = sim.sim_lock
        self.shutdown_event = shutdown_event
        self.draw_debug = draw_debug
        self.do_timers = do_timers
        self.debug_lines: dict[str, Any] = {}
        # self.prev_debug_lines = []
        self.interval = 0.005 # 5ms
        self.thread_cntr_max = 5/self.interval
        self.initial_pos = np.array(initial_robot_conf["pos"])
        self.fsm = FSM(
            controller=self,
            initial_pos=self.initial_pos,
            initial_orn=np.array(initial_robot_conf["orn"])
        )

        # self.wrist_joint = 7 # joint before the gripper
        self.finger_joints = [9, 10]

        self.revolute_joint_idx: list[int] = []
        self.movable_joint_idxs: list[int] = []
        self.all_joint_idx: list[int] = []
        self.joint_lower_limits: list[float] = []
        self.joint_upper_limits: list[float] = []
        self.joint_names: list[str] = []
        self.max_joint_velocities: list[float] = []
        self.ee_link_index: int | None = None
        self.wrist_idx: int | None = None
        self.right_idx: int | None = None
        self.left_idx: int | None = None
        self.get_idxs()

        self.num_revolute_joints: int = len(self.revolute_joint_idx)
        self.num_movable_joints: int = len(self.movable_joint_idxs)

        # ===============================================================================
        # Variables to write data/plot files
        self.ft_names: list[str] = ["fx", "fy", "fz", "tx", "ty", "tz"]
        # self.ft_types = ["raw", "comp", "contact"]
        self.ft_types: list[str] = ["contact_ft", "ema_ft", "feeling_ft"]
        self.ft_keys: list[str] = [f"{name}_{type}" for type in self.ft_types for name in self.ft_names]
        self.ft: dict[str, float] = {ft_name: 0.0 for ft_name in self.ft_keys}
        self.ft_contact_wrist: list[float] = [0.0] * 6
        self.ft_ema: list[float] = [0.0] * 6
        self.ft_feeling: list[float] = [0.0] * 6

        self.speed_world_frame: NDArray[np.float64] = np.zeros(6)
        self.speed_wrist: NDArray[np.float64] = np.zeros(6)
        self.speed_names: list[str] = ["vx", "vy", "vz", "wx", "wy", "wz"]
        self.speed_types: list[str] = ["v_wf", "v_wrist"]
        self.speed_keys: list[str] = [f"{name}_{type}" for type in self.speed_types for name in self.speed_names]
        self.speed: dict[str, float] = {name: 0.0 for name in self.speed_keys}

        self.joint_types: list[str] = ["jointspeed"]
        self.joint_keys: list[str] = [f"{name}_{type}"
                                      for type in self.joint_types
                                      for name in self.movable_joint_idxs]
        self.joint_speed: dict[str, float] = {name: 0.0 for name in self.speed_keys}

        # ===============================================================================
        # Data files for plotting & writer thread
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_path, exist_ok=True)

        self.ft_file = os.path.join(self.data_path, "ft_data.csv")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.vel_file = os.path.join(self.data_path, "vel_data.csv")
        
        self.initialize_plot_files()
        self.data_queue: queue.Queue[Any] = queue.Queue()
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
            self.total_num_joints: int = p.getNumJoints(self.robot)
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
            # using center of mass for force stuff
            wrist_pos, wrist_quat = link_state[:2]
            wrist_pos = np.array(wrist_pos)

            rot_wrist_world = np.array(p.getMatrixFromQuaternion(wrist_quat)).reshape(3, 3)
            rot_world_to_wrist = rot_wrist_world.T

            contact_pts = p.getContactPoints(self.robot, self.sim.obj)

        if len(contact_pts) <= 0:
            self.ft_contact_wrist = [0.0] * 6
            return
        
        total_force_world = np.zeros(3)
        total_torque_world = np.zeros(3)
        contact_pos = np.zeros(3)
        
        for pt in contact_pts:
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

            # link = pt[3]
            # print(f"DEBUG contact: link: {link}, cntr: {cntr}, "
            #       f"norm: {[f"{x:.6f}" for x in rot_world_to_wrist @ (fn * normal)]}, "
            #       f"ff1: {[f"{x:.6f}" for x in rot_world_to_wrist @ (f_lat1 * lateral_dir1)]}, "
            #       f"ff2: {[f"{x:.6f}" for x in rot_world_to_wrist @ (f_lat2 * lateral_dir2)]}, "
            #       f"total: {[f"{x:.6f}" for x in rot_world_to_wrist @ total_force_world]}")

        total_force_local = rot_world_to_wrist @ total_force_world
        total_torque_local = rot_world_to_wrist @ total_torque_world

        self.ft_contact_wrist = list(total_force_local) + list(total_torque_local)

        # Draw debug line
        if self.draw_debug:
            # debug direction
            start_pos = contact_pos

            for i in range(3):
                force_local = np.zeros(3)
                force_local[i] = total_force_local[i]
                force_world = rot_wrist_world @ force_local
                end_pos_force = start_pos + force_world

                linecolor = [0.0, 0.0, 0.0]
                linecolor[i] = 0.75

                kwargs: dict[str, Any] = {"lineColorRGB": linecolor, "lineWidth": 5, "lifeTime": 0}
                self.update_debug_line("force_wrist", start_pos, end_pos_force, kwargs)

                torque_local = np.zeros(3)
                torque_local[i] = total_torque_local[i]
                torque_world = rot_wrist_world @ torque_local
                end_pos_torque = start_pos + torque_world

                linecolor[i] = 0.25

                kwargs = {"lineColorRGB": linecolor, "lineWidth": 8, "lifeTime": 0}
                self.update_debug_line("torque_wrist", start_pos, end_pos_torque, kwargs)

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
                forces = [100] * self.num_movable_joints,
            )

    # -----------------------------------------------------------------------------------------------------------
    def reset_robot_pos(self) -> None:
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
    def is_gripper_open(self) -> bool:
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
    def do_move_pos(self, pos: NDArray[np.float64] = np.zeros(3), euler_rad: NDArray[np.float64] = np.zeros(3)) -> bool:
        Kp_lin = 0.7
        Kp_ang = 0.95
        
        pos_err, ang_err = self.get_pos_error(desired_pos=pos, desired_euler_orn=euler_rad)
        total_pos_err = np.linalg.norm(pos_err)
        total_ang_err = np.linalg.norm(ang_err)

        # if any(abs(x) > 0.03 for x in pos_err) or any(abs(x) > 0.2 for x in ang_err):
        if abs(total_pos_err) >= 0.06 or abs(total_ang_err) >= 1.8:
            # print(f"Moving to desired pose, position err: {[f"{x:.4f}" for x in pos_err]}, angular err: {[f"{x:.4f}" for x in ang_err]}, total pos err: {total_pos_err:.4f}, total ang err: {total_ang_err:.4f}")
            rad_err = np.deg2rad(ang_err)

            pd_ang = np.array(rad_err) * Kp_ang
            pd_lin = np.array(pos_err) * Kp_lin

            self.desired_pos_ee = np.array(pos)
            self.desired_quat_ee = np.array(p.getQuaternionFromEuler(euler_rad))
            self.do_move_velocity(v=pd_lin, w=pd_ang, link_name="ee", world_frame=True)
            return False
        else:
            print(f"Reached desired pose position err: {[f"{x:.4f}" for x in pos_err]}, angular err: {[f"{x:.4f}" for x in ang_err]}, total pos err: {total_pos_err:.4f}, total ang err: {total_ang_err:.4f}")
            return True

    # -----------------------------------------------------------------------------------------------------------
    def get_pos_error(self, 
                      desired_pos: NDArray[np.float64], 
                      desired_euler_orn: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with self.sim_lock:
            ls = p.getLinkState(self.robot, self.ee_link_index)
            ee_pos, ee_quat = ls[4], ls[5]
            
            desired_quat = p.getQuaternionFromEuler(desired_euler_orn)
            quat_error = p.getDifferenceQuaternion(ee_quat, desired_quat)
            axis, angle = p.getAxisAngleFromQuaternion(quat_error)
            
        ang_error = np.rad2deg(np.array(axis) * angle)
        pos_error = desired_pos - ee_pos

        return pos_error, ang_error

    # -----------------------------------------------------------------------------------------------------------
    def get_fingertip_pos(self):
        finger_length = 0.055
        finger_side_offset = 0.007
        right_tip_local: list[float] = [0, finger_side_offset, finger_length]
        left_tip_local: list[float] = [0, -finger_side_offset, finger_length]

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

    # -----------------------------------------------------------------------------------------------------------
    def get_wrist_pos(self):
        with self.sim_lock:
            ls = p.getLinkState(self.robot, self.wrist_idx)
            wrist_pos, wrist_quat = ls[4], ls[5]
            wrist_pos = np.array(wrist_pos)
            wrist_quat = np.array(wrist_quat)

        return wrist_pos, wrist_quat
    
    # -----------------------------------------------------------------------------------------------------------
    def get_relative_pos(self, pos_wf: NDArray[np.float64], quat_wf: NDArray[np.float64]):
        cur_pos, cur_quat = self.get_wrist_pos()
        inv_pos, inv_quat = p.invertTransform(pos_wf, quat_wf)
        relative_pos, relative_quat = p.multiplyTransforms(inv_pos, inv_quat, cur_pos, cur_quat)
        return relative_pos, relative_quat
    
    # -----------------------------------------------------------------------------------------------------------
    def do_move_velocity(self, 
                         v: NDArray[np.float64], 
                         w: NDArray[np.float64], 
                         link_name: str = "wrist", 
                         world_frame: bool = False
    ) -> None:
        match(link_name):
            case "wrist":
                link = self.wrist_idx
            case "ee":
                link = self.ee_link_index
            case _:
                print("ERROR: Unknown link for determining movement. Exiting ...")
                sys.exit(1)

        # wf = world frame
        with self.sim_lock:
            # use link positions for calculating movement
            ls = p.getLinkState(self.robot, link, computeLinkVelocity=True, computeForwardKinematics=True)
            link_pos_wf, link_quat_wf = ls[4], ls[5]
            R_link2world = np.array(p.getMatrixFromQuaternion(link_quat_wf)).reshape(3, 3)
            
        # convert speed to world frame
        if world_frame:
            v_wf = v
            w_wf = w
            v_link = R_link2world.T.dot(v_wf)
            w_link = R_link2world.T.dot(w_wf)
        else:
            v_link = v
            w_link = w
            v_wf = R_link2world.dot(v_link)
            w_wf = R_link2world.dot(w_link)

        self.speed_world_frame = np.hstack((v_wf, w_wf))
        self.speed_wrist[:3] = v_link
        self.speed_wrist[3:] = w_link

        # print(f"DEBUG do_move_velocity: v: {[f'{x:.6f}' for x in v]}, w: {[f'{x:.6f}' for x in w]}, v_wrist: {[f'{x:.6f}' for x in v_wrist]}, w_wrist: {[f'{x:.6f}' for x in w_wrist]}")

        # Draw debug lines showing the desired velocities
        if self.draw_debug:
            start_pos = np.array(link_pos_wf)
            speed_lines: list[tuple[str, NDArray[np.float64], list[float]]] = [
                ("v_wf", v_wf, [0, 0, 1]),  # v = blue
                ("w_wf", w_wf, [1, 0, 0])   # w = red
            ]

            for name, speed, color in speed_lines:
                speed_wf = np.where(np.abs(speed) < 1e-6, 1e-6, speed)
                speed_hat = speed_wf / np.linalg.norm(speed_wf)
                end_pos = start_pos + (speed_hat * 0.3)
                kwargs: dict[str, Any] = {"lineColorRGB": color, "lineWidth": 3, "lifeTime": 0}
                self.update_debug_line(name, start_pos, end_pos, kwargs)

            with self.sim_lock:
                ls = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)
                ee_pos_wf, ee_quat_wf = ls[4], ls[5]
                R_ee2world = np.array(p.getMatrixFromQuaternion(ee_quat_wf)).reshape(3, 3)

            start_pos = np.array(ee_pos_wf)
            line_len = 0.2
            ee_x_line_wf = R_ee2world.dot(np.array([line_len, 0, 0]))
            ee_y_line_wf = R_ee2world.dot(np.array([0, line_len, 0]))
            ee_z_line_wf = R_ee2world.dot(np.array([0, 0, line_len]))
            ee_orn_lines: list[tuple[str, NDArray[np.float64], list[float]]] = [
                ("wrist_orn_x", ee_x_line_wf, [1, 0, 0]), # red
                ("wrist_orn_y", ee_y_line_wf, [0, 1, 0]), # green
                ("wrist_orn_z", ee_z_line_wf, [0, 0, 1])  # blue
            ]

            for name, line, color in ee_orn_lines:
                kwargs = {"lineColorRGB": color, "lineWidth": 1, "lifeTime": 0}
                end_pos = start_pos + line
                self.update_debug_line(name, start_pos, end_pos, kwargs)

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
    def update_debug_line(self, 
                          line_name: str, 
                          start_pos: NDArray[np.float64], 
                          end_pos: NDArray[np.float64], 
                          kwargs: dict[str, Any]
    ) -> None:
        if line_name in self.debug_lines:
            kwargs["replaceItemUniqueId"] = self.debug_lines[line_name]
        with self.sim_lock:
            self.debug_lines[line_name] = p.addUserDebugLine(start_pos, end_pos, **kwargs)

    # -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        next_thread_time = time.perf_counter()
        interval = self.interval
        thread_cntr = 0
        thread_times: list[list[float]] = []

        while not self.shutdown_event.is_set():
            thread_start_time = time.perf_counter()
            prev_time = thread_start_time

            timediff: list[float] = []
            timenames: list[str] = []

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

            if not self.pause_sim:
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
                            ls = p.getLinkState(self.robot, self.ee_link_index, computeForwardKinematics=True)
                            self.desired_pos_ee, self.desired_quat_ee = ls[4], ls[5]
                            ls = p.getLinkState(self.robot, self.wrist_idx, computeForwardKinematics=True)
                            self.desired_pos_wrist, self.desired_quat_wrist = ls[4], ls[5]
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
                if not self.pause_sim:
                    print(f"Controller thread is running behind schedule by {-sleep_time:.6f}s")
                    if len(thread_times) > 0:
                        print(f"Thread times: ", [f"{name}: {t:.6f}s" for name, t in zip(timenames, thread_times[-1])])
                next_thread_time = time.perf_counter()

        print("Exiting controller thread...")
        sys.exit(0)


if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)
