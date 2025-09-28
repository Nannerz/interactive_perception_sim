import threading, os, csv, time, sys
from collections import defaultdict
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
        initial_robot_conf: dict[str, list[float]] = {"pos": [0.7, 0, 0.1], "orn": [ 0, 0.7071068, 0, 0.7071068 ]},
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
        self.interval = 0.005 # 5ms
        self.thread_cntr = 0
        self.thread_cntr_max = 5/self.interval
        self.timers: dict[str, float] = defaultdict(float)
        self.fsm_timers: dict[str, float] = {}
        self.loop_timers: dict[str, float] = {}
        self.initial_pos = np.array(initial_robot_conf["pos"])
        self.fsm = FSM(
            controller=self,
            initial_pos=self.initial_pos,
            initial_orn=np.array(initial_robot_conf["orn"])
        )

        self.finger_joints = [9, 10]

        self.joint_lower_limits: list[float] = []
        self.joint_upper_limits: list[float] = []
        self.joint_names: list[str] = []
        self.max_joint_velocities: list[float] = []
        self.movable_joint_idxs: list[int] = []
        self.arm_joint_idxs: list[int] = []
        self.idxs: dict[str, int] = {}
        self.get_idxs()

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

        self.speed_world: NDArray[np.float64] = np.zeros(6)
        self.speed_wrist: NDArray[np.float64] = np.zeros(6)
        self.speed_names: list[str] = ["vx", "vy", "vz", "wx", "wy", "wz"]
        self.speed_types: list[str] = ["v_world", "v_wrist"]
        self.speed_keys: list[str] = [f"{name}_{type}" for type in self.speed_types for name in self.speed_names]
        self.speed: dict[str, float] = {name: 0.0 for name in self.speed_keys}

        # ===============================================================================
        # Data files for plotting & writer thread
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
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
        self.pause_sim = False
        self.next_dq = [0.0] * self.num_movable_joints
        self.next_pos = [0.0] * self.num_movable_joints  # ik_values
        self.velocity_solver = osqp.OSQP()
        self.link_states: dict[str, Any] = {}
        self.movable_joint_states: dict[int, Any] = {}

    # -----------------------------------------------------------------------------------------------------------
    def get_idxs(self) -> None:
        """ Get indices of important joints and store them """
        
        self.total_num_joints: int = p.getNumJoints(self.robot)
        for i in range(self.total_num_joints):
            info = p.getJointInfo(self.robot, i)
            name: str = info[1].decode("utf-8")

            if info[2] != p.JOINT_FIXED:
                self.movable_joint_idxs.append(i)
                self.joint_names.append(name)
                self.joint_lower_limits.append(info[8])
                self.joint_upper_limits.append(info[9])
                self.max_joint_velocities.append(info[11])
            match name:
                case "panda_grasptarget_hand":
                    self.idxs["ee"] = i
                case "panda_hand_joint":
                    self.idxs["wrist"] = i
                case "panda_finger_joint1":
                    self.idxs["right"] = i
                case "panda_finger_joint2":
                    self.idxs["left"] = i
                case _:
                    pass
            if not "finger" in name and not "hand" in name:
                self.arm_joint_idxs.append(i)

        if not self.idxs.get("wrist") or not self.idxs.get("right") or not self.idxs.get("left") or not self.idxs.get("ee"):
            print("Could not find all indices, exiting ...")
            sys.exit(1)

    # -----------------------------------------------------------------------------------------------------------
    def initialize_plot_files(self) -> None:
        """ Initialize CSV file with headers for plotting, overwrites/deletes existing file with the same name """
        
        keys = self.ft_keys + self.speed_keys
        with open(self.ft_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

    # -----------------------------------------------------------------------------------------------------------
    def write_data_files(self) -> None:
        """ Write joint force/torque readings and joint velocities to CSV files for plotting """
        
        for i, name in enumerate(self.speed_names):
            self.speed[f"{name}_v_world"] = self.speed_world[i]
            self.speed[f"{name}_v_wrist"] = self.speed_wrist[i]

        for i, name in enumerate(self.ft_names):
            self.ft[f"{name}_contact_ft"] = self.ft_contact_wrist[i]
            self.ft[f"{name}_ema_ft"] = self.ft_ema[i]
            self.ft[f"{name}_feeling_ft"] = self.ft_feeling[i]

        combined = self.ft | self.speed
        
        self.data_queue.put({"type": "csv", "data": combined})

    # -----------------------------------------------------------------------------------------------------------
    def write_world_position(self) -> None:
        """ Write current world frame positions of end effector to pos_file """
        
        ls = self.link_states["wrist"]
        ee_pos, ee_quat = ls[4], ls[5]
        roll, pitch, yaw = np.rad2deg(p.getEulerFromQuaternion(ee_quat))
        R_ee2world = np.array(p.getMatrixFromQuaternion(ee_quat)).reshape(3, 3)

        pos: dict[str, Any] = {
            "x": ee_pos[0],
            "y": ee_pos[1],
            "z": ee_pos[2],
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "rx1": R_ee2world[0, 0],
            "rx2": R_ee2world[0, 1],
            "rx3": R_ee2world[0, 2],
            "ry1": R_ee2world[1, 0],
            "ry2": R_ee2world[1, 1],
            "ry3": R_ee2world[1, 2],
            "rz1": R_ee2world[2, 0],
            "rz2": R_ee2world[2, 1],
            "rz3": R_ee2world[2, 2],
        }

        self.data_queue.put({"type": "json", "data": pos}, timeout=0.001)

    # -----------------------------------------------------------------------------------------------------------
    def get_contact_ft(self) -> None:
        """ Get current contact points and sum forces/torques on wrist's center of mass """
        
        contact_pts = p.getContactPoints(self.robot, self.sim.obj)
        if len(contact_pts) <= 0:
            self.ft_contact_wrist = [0.0] * 6
            return
        
        ls = self.link_states["wrist"]
        # using center of mass link pos/quat for force stuff
        wrist_pos, wrist_quat = ls[:2]
        wrist_pos = np.array(wrist_pos)

        rot_wrist_world = np.array(p.getMatrixFromQuaternion(wrist_quat)).reshape(3, 3)
        rot_world_to_wrist = rot_wrist_world.T
        
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

        # # Draw debug line
        # if self.draw_debug:
        #     # debug direction
        #     start_pos = contact_pos
            
        #     for i in range(6):
        #         line_local = np.zeros(3)
        #         line_local[i % 3] = self.ft_contact_wrist[i]
        #         line_world = rot_wrist_world @ line_local
        #         end_pos = start_pos + line_world
                
        #         linecolor = [0.8, 0.0, 0.0] if i < 3 else [0.0, 0.0, 0.8]
        #         name = f"force_wrist_{i}" if i < 3 else f"torque_wrist_{i % 3}"

        #         kwargs: dict[str, Any] = {"lineColorRGB": linecolor, "lineWidth": 8, "lifeTime": 0}
        #         self.update_debug_line(name, start_pos, end_pos, kwargs)

    # -----------------------------------------------------------------------------------------------------------
    def stop_movement(self) -> None:
        """ Set joint velocities to 0 """
        
        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.movable_joint_idxs,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * self.num_movable_joints,
        )

    # -----------------------------------------------------------------------------------------------------------
    def apply_next_dq(self) -> None:
        """ Apply next_dq velocities to robot. High force (100) is used to prevent slipping """
        
        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.movable_joint_idxs,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=self.next_dq,
            velocityGains=[1.0] * self.num_movable_joints,
            forces = [100] * self.num_movable_joints,
        )
        
        self.next_dq = [0.0] * self.num_movable_joints

    # -----------------------------------------------------------------------------------------------------------
    def apply_next_pos(self) -> None:
        """ Apply next_pos positions to robot. """

        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.movable_joint_idxs,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.next_pos,
            positionGains=[0.7] * self.num_movable_joints,
            forces = [100] * self.num_movable_joints,
        )
        self.draw_dq_debug(link_name="wrist")

    # -----------------------------------------------------------------------------------------------------------
    def reset_robot_pos(self) -> None:
        """ Reset robot joints to center of joint limits """
        
        ik_vals = [
            (self.joint_upper_limits[i] + self.joint_lower_limits[i]) / 2
            for i in range(self.num_movable_joints)
        ]

        for idx, joint_idx in enumerate(self.movable_joint_idxs):
            p.resetJointState(self.robot, joint_idx, ik_vals[idx])

        self.next_pos = ik_vals

    # -----------------------------------------------------------------------------------------------------------
    def open_gripper(self) -> None:
        """ Open the gripper by setting finger joints to their max positions """

        self.next_dq[-1] = 0.2
        self.next_dq[-2] = 0.2

    # -----------------------------------------------------------------------------------------------------------
    def is_gripper_open(self) -> bool:
        """ Check if gripper is open (both fingers past 0.039) """
        
        gripper_pos = [
            p.getJointState(self.robot, joint_idx)[0]
            for joint_idx in self.finger_joints
        ]
        if all(pos >= 0.039 for pos in gripper_pos):
            return True
        else:
            return False

    # -----------------------------------------------------------------------------------------------------------
    def do_move_pos(self, 
                    pos_world: NDArray[np.float64], 
                    quat_world: NDArray[np.float64]
    ) -> bool:
        """
        Move wrist to desired world position and orientation using velocity control
        Max position error threshold: 6cm
        Max angular error threshold: 1.8 degrees

        Parameters:
            pos_world: Desired position in world frame [x, y, z]
            quat_world: Desired quaternion orientation in world frame [x, y, z, w]

        Returns:
            bool: True if reached desired position within threshold, False otherwise
        """
        
        Kp_lin = 0.9
        Kp_ang = 0.9
        link_name = "wrist"

        pos_err, quat_err = self.get_pos_error(desired_pos=pos_world, desired_quat=quat_world, link_name=link_name)
        total_pos_err = np.linalg.norm(pos_err)

        # print(f"Moving to desired pose, position err: {[f"{x:.4f}" for x in pos_err]}, angular err: {[f"{x:.4f}" for x in ang_err]}, total pos err: {total_pos_err:.4f}, total ang err: {total_ang_err:.4f}")

        pd_lin = np.array(pos_err) * Kp_lin
        # Use shortest arc
        if quat_err[3] < 0:
            quat_err = [-quat_err[0], -quat_err[1], -quat_err[2], -quat_err[3]]
        quat_xyz = np.array(quat_err[:3])
        axis = quat_err[3]
        quat_norm = np.linalg.norm(quat_xyz)

        # Theta is smallest angle between current & desired orientation
        theta = 2.0 * np.arctan2(quat_norm, axis)
        if quat_norm < 1e-12:
            pd_ang = np.zeros(3)
        else:
            pd_ang = theta * (quat_xyz / quat_norm) * Kp_ang
        
        if abs(total_pos_err) >= 0.02 or abs(theta) >= 0.04:
            # print(f"DEBUG total pos err: {total_pos_err}, pos diffs: {[f'{x:.4f}' for x in pos_err]}, pd_lin: {[f'{x:.4f}' for x in pd_lin]} theta err: {theta}, quat err: {[f'{x:.4f}' for x in quat_err]}, pd_ang: {[f'{x:.4f}' for x in pd_ang]}")
            self.set_next_dq(v=pd_lin, w=pd_ang, world_frame=True)
            return False
        else:
            print(f"Reached desired pose position err: {[f"{x:.4f}" for x in pos_err]}, angular err: {[f"{x:.4f}" for x in quat_err]}, total pos err: {total_pos_err:.4f}, total ang err: {theta:.4f}")
            return True

    # -----------------------------------------------------------------------------------------------------------
    def get_pos_error(self, 
                      desired_pos: NDArray[np.float64], 
                      desired_quat: NDArray[np.float64],
                      link_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ 
        Get a link's position and orientation errors in world frame from desired world frame position and orientation

        Parameters:
            desired_pos: Desired position in world frame [x, y, z]
            desired_quat: Desired quaternion orientation in world frame [x, y, z, w]
            link_name: Name of the link to get error for ("wrist", "ee")
        
        Returns:
            pos_error: Position error in world frame [ex, ey, ez]
            quat_error: Quaternion error in world frame [ex, ey, ez, ew]
        """

        ls = self.link_states[link_name]
        cur_pos, cur_quat = ls[4], ls[5]
        
        quat_error = p.getDifferenceQuaternion(cur_quat, desired_quat)
        pos_error = desired_pos - cur_pos

        return pos_error, quat_error

    # -----------------------------------------------------------------------------------------------------------
    def get_fingertip_pos_world(self) -> tuple[list[float], list[float]]:
        """ Get current left & right fingertip positions in world frame """

        ls_wrist = self.link_states["wrist"]
        pos_wrist_world = np.array(ls_wrist[4])

        right_tip_in_wrist, left_tip_in_wrist = self.get_fingertip_pos_wrist()
        right_tip_in_wrist = np.array(right_tip_in_wrist)
        left_tip_in_wrist = np.array(left_tip_in_wrist)
        
        R_wrist2world = np.array(p.getMatrixFromQuaternion(ls_wrist[5])).reshape(3, 3)
        right_tip_pos_world = (pos_wrist_world + R_wrist2world @ right_tip_in_wrist).tolist()
        left_tip_pos_world = (pos_wrist_world + R_wrist2world @ left_tip_in_wrist).tolist()

        return right_tip_pos_world, left_tip_pos_world

    # -----------------------------------------------------------------------------------------------------------
    def get_fingertip_pos_wrist(self) -> tuple[list[float], list[float]]:
        """ Get current left & right fingertip positions in wrist frame """
        
        finger_height_offset = 0.0
        finger_side_offset = 0.00729
        finger_length_offset = 0.05840 + 0.05385
        
        finger_right_offset = -1 * (finger_side_offset + self.movable_joint_states[self.idxs["right"]][0])
        finger_left_offset = (finger_side_offset + self.movable_joint_states[self.idxs["left"]][0])

        right_tip_in_wrist: list[float] = [finger_height_offset, finger_right_offset, finger_length_offset]
        left_tip_in_wrist: list[float] = [finger_height_offset, finger_left_offset, finger_length_offset]

        return right_tip_in_wrist, left_tip_in_wrist

    # -----------------------------------------------------------------------------------------------------------
    def draw_fingertip_debug(self) -> None:
        """ Draw debug lines from wrist to fingertips """
        if self.draw_debug:
            right_tip_pos_world, left_tip_pos_world = self.get_fingertip_pos_world()
            pos_wrist_world, _ = self.get_pos("wrist")

            linecolor1 = [1, 0, 0]
            kwargs1: dict[str, Any] = {"lineColorRGB": linecolor1, "lineWidth": 5, "lifeTime": 0}
            self.update_debug_line("left_tip", pos_wrist_world, np.array(left_tip_pos_world), kwargs1)

            linecolor2 = [0, 1, 0]
            kwargs2: dict[str, Any] = {"lineColorRGB": linecolor2, "lineWidth": 5, "lifeTime": 0}
            self.update_debug_line("right_tip", pos_wrist_world, np.array(right_tip_pos_world), kwargs2)

    # -----------------------------------------------------------------------------------------------------------
    def get_pos(self, link_name: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """ Get a link's current position and quaternion in world frame """

        ls = self.link_states[link_name]
            
        wrist_pos, wrist_quat = ls[4], ls[5]
        wrist_pos = np.array(wrist_pos)
        wrist_quat = np.array(wrist_quat)

        return wrist_pos, wrist_quat
    
    # -----------------------------------------------------------------------------------------------------------
    def get_relative_pos(self, 
                         pos_world: NDArray[np.float64], 
                         quat_world: NDArray[np.float64], 
                         link_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get a link's distance in the link's frame from the given world frame position and quaternion

        Parameters:
            pos_world: Desired position in world frame [x, y, z]
            quat_world: Desired orientation in world frame [x, y, z, w]
            link_name: Name of the link to get relative position for ("wrist", "ee", etc.)

        Returns:
            relative_pos: Position of link relative to desired position [x, y, z]
            relative_angle: Euler angles of link relative to desired orientation [roll, pitch, yaw] in degrees
        """

        cur_pos, cur_quat = self.get_pos(link_name=link_name)
        inv_pos, inv_quat = p.invertTransform(pos_world, quat_world)
        relative_pos, relative_quat = p.multiplyTransforms(inv_pos, inv_quat, cur_pos, cur_quat)

        relative_angle = np.rad2deg(p.getEulerFromQuaternion(relative_quat))
        # while relative_angle > math.pi:
        #     relative_quat = p.getQuaternionFromAxisAngle(p.getAxisAngleFromQuaternion(relative_quat)[0], relative_angle - 2 * math.pi)

        return relative_pos, relative_angle

    # -----------------------------------------------------------------------------------------------------------
    def set_next_dq(self, 
                    v: NDArray[np.float64], 
                    w: NDArray[np.float64], 
                    world_frame: bool = False
    ) -> None:
        """ 
        Update next_dq based on desired linear velocities, angular velocities, link, and frame

        Parameters:
        v: Desired linear velocity [vx, vy, vz] in m/s
        w: Desired angular velocity [wx, wy, wz] in rad/s
        link_name: Name of the link to move (default: "wrist")
        world_frame: If True, interpret given velocities as world frame velocities (default: False)

        Returns:
        None
        """

        # Get the link we're moving based on
        link = self.idxs["wrist"]
        ls = self.link_states["wrist"]
        link_quat_world = ls[5]
        R_link2world = np.array(p.getMatrixFromQuaternion(link_quat_world)).reshape(3, 3)

        # convert speed to world frame
        if world_frame:
            v_world = v.copy()
            w_world = w.copy()
            v_link = R_link2world.T @ v_world
            w_link = R_link2world.T @ w_world
        else:
            v_link = v.copy()
            w_link = w.copy()
            v_world = R_link2world @ v_link
            w_world = R_link2world @ w_link

        twist_world = np.hstack((v_world, w_world))
        self.speed_world = twist_world.copy()
        self.speed_wrist = np.hstack((v_link, w_link))
        
        # print(f"DEBUG: R_link2world: {[f'{x:.6f}' for x in R_link2world.flatten()]}")
        # print(f"DEBUG set_next_dq: v_wrist: {[f'{x:.6f}' for x in v_link]}, w_wrist: {[f'{x:.6f}' for x in w_link]}")
        # print(f"DEBUG set_next_dq: v_world: {[f'{x:.6f}' for x in v_world]}, w_world: {[f'{x:.6f}' for x in w_world]}")
        
        # Get current jacobian
        q_full = [js[0] for js in self.movable_joint_states.values()]
        qd = [0.0] * self.num_movable_joints
        qdd = [0.0] * self.num_movable_joints

        # Jacobian, world frame
        J_lin, J_ang = p.calculateJacobian(
            bodyUniqueId=self.robot,
            linkIndex=link,
            localPosition=[0, 0, 0],
            objPositions=q_full,
            objVelocities=qd,
            objAccelerations=qdd,
        )

        J_full = np.vstack((np.array(J_lin), np.array(J_ang)))
        arm_mask = np.isin(self.movable_joint_idxs, self.arm_joint_idxs)
        J = J_full[:, arm_mask]
        n = J.shape[1]
        w_lin = 10.0   # prioritize linear to prevent unwanted slipping/sliding
        w_ang = 1.0
        Wtask = np.diag([w_lin, w_lin, w_lin, w_ang, w_ang, w_ang])

        Jw = Wtask @ J
        tw_w = Wtask @ twist_world

        dt = self.interval
        q = np.array(q_full)[arm_mask]
        q_min = np.array(self.joint_lower_limits)[arm_mask]
        q_max = np.array(self.joint_upper_limits)[arm_mask]
        dq_limits = np.array(self.max_joint_velocities)[arm_mask]

        # Calculate joint velocity limit based on current position and distance to limits
        # dq_q_lower = np.where(q > q_min, (q_min - q) / dt, 0)
        # dq_q_upper = np.where(q < q_max, (q_max - q) / dt, 0)
        dq_k = 0.8
        dq_q_lower = dq_k * (q_min - q) / dt
        dq_q_upper = dq_k * (q_max - q) / dt

        dq_upper_bound = np.minimum(dq_q_upper, dq_limits)
        dq_lower_bound = np.maximum(dq_q_lower, -dq_limits)

        # manipulability mu = sqrt(det(J J^T)) for the weighted task:
        JJt = Jw @ Jw.T
        # Numerical guard: add tiny diagonal if nearly singular:
        JJt_reg = JJt + 1e-12 * np.eye(6)
        try:
            mu = np.sqrt(np.linalg.det(JJt_reg))
        except np.linalg.LinAlgError:
            mu = 0.0

        lambda0 = 5e-4   # base damping
        lambda1 = 2e-1   # extra damping as singularity nears
        lam_sq = lambda0**2 + (lambda1**2) / (mu**2 + 1e-7)

        # Quadratic cost: 0.5 * dq.T H dq + g.T dq
        H = Jw.T.dot(Jw) + lam_sq * np.eye(n)
        g = -Jw.T.dot(tw_w)

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
        dq_task = res.x

        # Joint-limit repulsion gradient:
        eps = 1e-3
        alpha = 0.02     # push from limits (tune)
        q_center = 0.5 * (q_min + q_max) # center of joint ranges
        beta = 0.2       # pull to posture (tune)

        repulse = alpha * ( 1.0/((q - q_min + eps)**2) - 1.0/((q_max - q + eps)**2) )
        # prevent the repulsion from causing random movement spikes/drifts
        repulse_max = 0.3
        repulse = np.clip(repulse, -repulse_max, repulse_max) 
        toward  = beta * (q_center - q)
        dq_post = repulse + toward

        # Damped pseudoinverse for projection N = I - J#J, using weighted J:
        # Form J# = (Jw^T (Jw Jw^T + lam^2 I)^{-1})
        # (this matches the primary-task damping)
        M = Jw @ Jw.T + lam_sq * np.eye(6)
        J_sharp = Jw.T @ np.linalg.solve(M, np.eye(6))

        N = np.eye(n) - J_sharp @ Jw

        tw_norm = np.linalg.norm(twist_world)
        if tw_norm < 6e-2:
            k_ns = 0.02
        else:
            k_ns = 0.2
        
        dq_ns = k_ns * (N @ dq_post)
        dq = dq_task + dq_ns

        # If null-space pushes us over limits, scale dq_ns down first:
        over = (dq > dq_upper_bound) | (dq < dq_lower_bound)
        if np.any(over):
            # Compute maximum scale s in (0,1] that keeps bounds satisfied
            s_candidates: list[float] = []
            for i in range(n):
                if dq_ns[i] > 1e-12:
                    s_candidates.append((dq_upper_bound[i] - dq_task[i]) / dq_ns[i])
                elif dq_ns[i] < -1e-12:
                    s_candidates.append((dq_lower_bound[i] - dq_task[i]) / dq_ns[i])
            if len(s_candidates) > 0:
                s = max(0.0, min(1.0, min(s_candidates)))
            else:
                s = 1.0
            dq = dq_task + s * dq_ns

        # print(f"DEBUG: distance to qmax: {[f'{x:.6f}' for x in (q_max - q)]}, distance to qmin: {[f'{x:.6f}' for x in (q - q_min)]}")
        # print(f"DEBUG: dq_task: {[f'{x:.4f}' for x in dq_task]}, dq_ns: {[f'{x:.4f}' for x in dq_ns]}, dq: {[f'{x:.4f}' for x in dq]}")

        # Speed for fingers stays at 0
        dq_full = np.zeros_like(q_full)
        dq_full[arm_mask] = dq
        
        self.next_dq = dq_full.copy()
        self.draw_dq_debug(link_name="wrist")
        
    # -----------------------------------------------------------------------------------------------------------
    def draw_dq_debug(self, link_name: str) -> None:
        """ Draw debug lines showing the desired velocities & orientations of the given link and end effector """
        
        if self.draw_debug:
            ls = self.link_states["ee"]
            ee_pos_world, ee_quat_world = ls[4], ls[5]
            R_ee2world = np.array(p.getMatrixFromQuaternion(ee_quat_world)).reshape(3, 3)

            if "ee" != link_name:
                ls = self.link_states[link_name]
                link_pos_world, link_quat_world = ls[4], ls[5]
                R_link2world = np.array(p.getMatrixFromQuaternion(link_quat_world)).reshape(3, 3)
            else:
                link_pos_world = ee_pos_world
                link_quat_world = ee_quat_world
                R_link2world = R_ee2world

            
            start_pos_link = np.array(link_pos_world)
            speed_lines: list[tuple[str, NDArray[np.float64], list[float]]] = [
                ("v_world", self.speed_world[:3], [0, 0, 1]),  # v = blue
                ("w_world", self.speed_world[3:], [1, 0, 0])   # w = red
            ]

            # Speed lines
            for name, speed, color in speed_lines:
                speed_world = np.where(np.abs(speed) < 1e-6, 1e-6, speed)
                speed_hat = speed_world / np.linalg.norm(speed_world)
                end_pos = start_pos_link + (speed_hat * 0.3)
                kwargs: dict[str, Any] = {"lineColorRGB": color, "lineWidth": 3, "lifeTime": 0}
                self.update_debug_line(name, start_pos_link, end_pos, kwargs)

            # Used link (usually wrist) and End Effector orientation lines
            start_pos_ee = np.array(ee_pos_world)
            line_len = 0.2
            for i in range(3):
                line_base = np.zeros(3)
                line_color = np.zeros(3)
                line_base[i] = line_len
                line_color[i] = 1.0 # RGB: x=red, y=green, z=blue
                line_name_link = f"wrist_orn_{i}"
                line_name_ee = f"ee_orn_{i}"
                link_line_world = R_link2world.dot(line_base)
                ee_line_world = R_ee2world.dot(line_base)
                
                kwargs = {"lineColorRGB": line_color, "lineWidth": 1, "lifeTime": 0}
                end_pos_link = start_pos_link + link_line_world
                end_pos_ee = start_pos_ee + ee_line_world
                self.update_debug_line(line_name_link, start_pos_link, end_pos_link, kwargs)
                self.update_debug_line(line_name_ee, start_pos_ee, end_pos_ee, kwargs)
                
    # -----------------------------------------------------------------------------------------------------------
    def update_debug_line(self, 
                          line_name: str, 
                          start_pos: NDArray[np.float64], 
                          end_pos: NDArray[np.float64], 
                          kwargs: dict[str, Any]
    ) -> None:
        """ 
        Update or create a debug line in the simulation based on the given name.
        
        Creating new lines is expensive and causes lag on the controller, 
            so we keep track of existing lines and update them instead if possible.

        Parameters:
            line_name: Unique name for the debug line
            start_pos: Starting position of the debug line
            end_pos: Ending position of the debug line
            kwargs: Additional keyword arguments for line customization (color, width, lifetime)

        Returns:
            None
        """
        
        if line_name in self.debug_lines:
            kwargs["replaceItemUniqueId"] = self.debug_lines[line_name]

        self.debug_lines[line_name] = p.addUserDebugLine(start_pos, end_pos, **kwargs)

    # -----------------------------------------------------------------------------------------------------------
    def toggle_pause_sim(self, pause: bool) -> None:
        """ This program normally runs with real-time simulation on, so we pause the simulation by turning it off """
        
        if self.pause_sim != pause:
            self.pause_sim = pause
            p.setRealTimeSimulation(not pause)
            print(f"Got input to {'pause' if self.pause_sim else 'unpause'} simulation")

    # -----------------------------------------------------------------------------------------------------------
    def next_timer(self, timer_name: str) -> None:
        """ Record incremental time between timer calls """
        
        if self.do_timers:
            cur_time = time.perf_counter()
            self.timers[timer_name] = cur_time - self.loop_timers["prev_time"]
            self.loop_timers["prev_time"] = cur_time

    # -----------------------------------------------------------------------------------------------------------
    def do_sleep_interval(self) -> None:
        """
        Sleep for the remaining time in the interval to maintain a consistent loop rate.
        If the controller is behind schedule, resets next_thread_time to the current time.
        """
        
        sleep_start = time.perf_counter()
        elapsed_time = sleep_start - self.loop_timers["thread_start"]
        desired_sleep = self.interval - elapsed_time

        self.timers["thread_time"] = elapsed_time
        self.timers["desired_sleep"] = desired_sleep

        if elapsed_time > self.interval:
            if not self.pause_sim:
                print(f"Controller thread is running slow! Thread time: {elapsed_time:.6f}s, desired sleep: {-desired_sleep:.6f}s")
                print(f"Thread times: ", [f"{name}: {t:.6f}s" for name, t in self.timers.items()])
                print(f"Loop timers: ", [f"{name}: {t:.6f}s" for name, t in self.loop_timers.items()])
                print(f"FSM timers: ", [f"{name}: {t:.6f}s" for name, t in self.fsm_timers.items()])
            
            self.loop_timers["next_thread_time"] = time.perf_counter() + self.interval
        else:
            while time.perf_counter() < self.loop_timers["next_thread_time"]:
                time.sleep(1/1000)
                
            self.loop_timers["next_thread_time"] += self.interval

        self.timers["actual_sleep"] = time.perf_counter() - sleep_start
            
    # -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        """ Main controller thread loop """
        
        self.loop_timers["next_thread_time"] = time.perf_counter() + self.interval

        while not self.shutdown_event.is_set():
            self.timers.clear()
            start_time = time.perf_counter()
            self.loop_timers["thread_start"] = start_time
            self.loop_timers["prev_time"] = start_time
            self.next_timer("thread_top")
            
            # Press Q to pause/unpause simulation at runtime
            qKey = ord("q")
            keys = p.getKeyboardEvents()
            if qKey in keys and keys[qKey] & p.KEY_WAS_TRIGGERED:
                self.toggle_pause_sim(not self.pause_sim)
            
            self.next_timer("input_check")

            if not self.pause_sim:
                self.link_states: dict[str, Any] = dict(zip(self.idxs.keys(), p.getLinkStates(self.robot, list(self.idxs.values()), computeForwardKinematics=True)))
                self.movable_joint_states: dict[int, Any] = dict(zip(self.movable_joint_idxs, p.getJointStates(self.robot, self.movable_joint_idxs)))

                self.get_contact_ft()
                self.next_timer("get_contact_ft")

                self.fsm_timers = self.fsm.next_state()
                self.next_timer("fsm_next_state")

                match self.mode:
                    case "position":
                        self.apply_next_pos()
                    case "velocity":
                        self.apply_next_dq()
                    case _:
                        print("Unknown mode, stopping movement...")
                        self.stop_movement()

                self.draw_fingertip_debug()
                
                self.next_timer("apply_next_dq")
                
                self.write_world_position()
                self.write_data_files()
                self.next_timer("write_data_files")

            self.do_sleep_interval()

        print("Exiting controller thread...")
        sys.exit(0)


if __name__ == "__main__":
    print("This script should not be executed directly, exiting ...")
    sys.exit(0)
