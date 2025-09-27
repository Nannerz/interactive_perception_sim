import math
import time
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from controller import Controller

fx = 0
fy = 1
fz = 2
tx = 3
ty = 4
tz = 5

class FSM:
    # def __init__(self, controller, initial_x, initial_y, initial_z) -> None:
    def __init__(self, controller: "Controller", initial_pos: NDArray[np.float64], initial_orn: NDArray[np.float64]) -> None:

        self.state = "start"
        self.controller = controller
        self.sim_lock = controller.sim_lock
        self.do_timers = controller.do_timers
        self.interval = controller.interval
        per_sec = self.interval
        self.thread_cntr_max = controller.thread_cntr_max
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.done = False

        # Forward/keep contact
        # self.forward_kp = 0.0675
        self.forward_kp = 0.08
        self.reset_kp = self.forward_kp
        self.touched_once = False
        self.is_touching = False

        # Wiggle
        self.wiggle_cntr = 0
        wiggles_per_sec = 3.0
        self.wiggle_max = math.ceil(1/self.interval * 1/wiggles_per_sec) + (4 - math.ceil(1/self.interval * 1/wiggles_per_sec) % 4)
        self.wiggle_dir = 1
        self.wiggles_before_align = 2
        self.wiggle_total_cntr = 0
        self.doing_wiggle = False
        yaw_kp = 1100.0
        self.wiggle_w_yaw = yaw_kp/self.wiggle_max * per_sec
        self.wiggle_w_pitch = yaw_kp/self.wiggle_max * per_sec
        print(f"DEBUG wiggle yaw speed: {self.wiggle_w_yaw}, pitch speed: {self.wiggle_w_pitch}")
        
        self.axes_aligned = False
        self.align_pos_world_frame = np.zeros(3)
        self.align_quat_world_frame = np.zeros(4)

        # Yaw align
        self.align_axes = False
        # self.yaw_algn_kp = 0.144
        self.yaw_algn_kp = 2.1
        self.yaw_algn_thresh = 0.01
        self.yaw_aligned = False

        # Pitch align
        # self.pitch_algn_kp = 0.144
        self.pitch_algn_kp = 2.1
        self.pitch_algn_thresh = 0.01
        self.pitch_aligned = False

        # Force/torque
        self.ft_contact_wrist = np.zeros(6)
        self.ft_ema = np.zeros(6)
        self.ft_contact_sum = np.zeros((1, 6))
        self.ft_avg = np.zeros((1, 6))
        self.alpha_ft_ema = 0.002
        # self.ft_feeling_sum = np.zeros((1, 6))
        # self.ft_feeling = np.zeros(6)

        self.thresh = {
            "fx": 0.1,  # threshold for force in x direction
            "fz": -0.16,  # threshold for force in z direction
            "fy": 0.1,  # threshold for force in y direction
            "tx": 0.1,  # threshold for torque in x direction
            "ty": 0.1,  # threshold for torque in y direction
            "tz": 0.1,  # threshold for torque in z direction
        }
        
        self.thread_times = []
        self.prev_thread_time = time.perf_counter()
        self.thread_cntr = 0

    # -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> None:
        self.controller.get_fingertip_pos()

        match self.state:
            case "start":
                self.controller.reset_robot_pos()
                self.state = "do_initial_pos"

            case "do_initial_pos":
                self.state_initial_pos()

            case "open_gripper":
                self.state_open_gripper()

            case "interact_perceive":
                self.state_interact_perceive()
                
            case "done":
                pass

            case _:
                pass

    # -----------------------------------------------------------------------------------------------------------
    def state_initial_pos(self) -> None:
        self.controller.mode = "velocity"
        ret = self.controller.do_move_pos(pos=self.initial_pos, euler_rad=self.initial_orn)
        if ret:
            self.state = "open_gripper"

    # -----------------------------------------------------------------------------------------------------------
    def state_open_gripper(self) -> None:
        self.controller.mode = "position"
        if self.controller.is_gripper_open():
            print(f"Gripper opened")
            self.state = "interact_perceive"
        else:
            print(f"Waiting for gripper to open ...")
            self.controller.open_gripper()

    # -----------------------------------------------------------------------------------------------------------
    def state_interact_perceive(self) -> None:
        self.ft_contact_wrist = np.array(self.controller.ft_contact_wrist) 

        # Move in the z direction to maintain light contact with object
        twist_contact = self.do_keep_contact()

        # Wiggle wrist to get the feeling force
        twist_wiggle = np.zeros(6)
        twist_yaw = np.zeros(6)
        twist_pitch = np.zeros(6)
        twist_grab = np.zeros(6)

        # First, wiggle to align pitch and yaw with the object
        if not self.axes_aligned and (self.is_touching or self.doing_wiggle):
            twist_wiggle = self.do_wiggle()
            
            self.ft_contact_sum = np.append(self.ft_contact_sum, self.ft_contact_wrist.reshape(1, 6), axis=0)
            if self.ft_contact_sum.shape[0] >= (self.wiggles_before_align * self.wiggle_max):
                self.ft_contact_sum = self.ft_contact_sum[1:]
            
            self.ft_avg = np.average(self.ft_contact_sum, axis=0)
            alpha_ema = 0.2
            self.ft_ema = alpha_ema * self.ft_contact_wrist + (1 - alpha_ema) * self.ft_ema
            self.controller.ft_ema = self.ft_ema.tolist()
            self.controller.ft_feeling = self.ft_avg.tolist()

            # Only start aligning once we have enough samples (self.wiggles_before_align * self.wiggle_max)
            if self.align_axes:
                twist_yaw = self.do_align_yaw()
                twist_pitch = self.do_align_pitch()
                pass

        # Next, move in the Y direction
        if self.axes_aligned:
            relative_pos, _ = self.controller.get_relative_pos(self.align_pos_world_frame, self.align_quat_world_frame)
            if abs(relative_pos[1]) > 0.08 or abs(relative_pos[2]) > 0.04:
                print(f"DEBUG reached max movement, relative pos: {[f"{x:.3f}" for x in relative_pos]}")
                self.done = True
                self.state = "done"
                self.controller.do_move_velocity(v=np.zeros(3), w=np.zeros(3), link_name="wrist", world_frame=False)
                return
            
            twist_grab = self.do_align_y()

        # Sum up all the speed components
        speed_total = twist_contact + twist_wiggle + twist_yaw + twist_pitch + twist_grab

        # Apply speed
        self.controller.mode = "velocity"
        self.controller.do_move_velocity(v=speed_total[:3], w=speed_total[3:], link_name="wrist", world_frame=False)

    # -----------------------------------------------------------------------------------------------------------
    def do_keep_contact(self) -> NDArray[np.float64]:
        # small offset so if we are just under, it's still "touching"
        if self.ft_contact_wrist[fz] <= self.thresh["fz"] + 0.05:
            self.is_touching = True
            if not self.touched_once:
                self.touched_once = True
        else:
            self.is_touching = False
        
        if self.axes_aligned:
            kp_mult = 0.15
            kp_max = 3.0 * self.forward_kp
            if not self.is_touching:
                if self.reset_kp < kp_max:
                    self.reset_kp += self.forward_kp * kp_mult
                if self.reset_kp > kp_max:
                    self.reset_kp = kp_max
            else:
                self.reset_kp = self.forward_kp * kp_mult
            kp = self.reset_kp
        else:
            kp = self.forward_kp

        force_diff = self.ft_contact_wrist[fz] - self.thresh["fz"]
        speed = kp * force_diff
        twist_contact = np.array([0, 0, speed, 0, 0, 0])

        return twist_contact

    # -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self) -> NDArray[np.float64]:
        if self.wiggle_cntr == 0:
            self.doing_wiggle = True
            
            if self.wiggle_total_cntr > self.wiggles_before_align:
                self.yaw_aligned = True if abs(self.ft_avg[fy]) <= self.yaw_algn_thresh else False
                self.pitch_aligned = True if abs(self.ft_avg[fx]) <= self.pitch_algn_thresh else False
                print(f"DEBUG: WIGGLE yaw ema: {self.ft_avg[fy]}, yaw aligned: {self.yaw_aligned}, pitch ema: {self.ft_avg[fx]}, pitch aligned: {self.pitch_aligned}")
                
                if self.yaw_aligned and self.pitch_aligned:
                    print("Both axes aligned, stopping wiggle")
                    self.align_pos_world_frame, self.align_quat_world_frame = self.controller.get_wrist_pos()
                    self.doing_wiggle = False
                    self.align_axes = False
                    self.axes_aligned = True
                    return np.zeros(6)

        self.wiggle_cntr += 1
        self.wiggle_dir = np.sign(np.cos(2 * np.pi * self.wiggle_cntr / self.wiggle_max))

        if self.wiggle_cntr > self.wiggle_max:
            self.wiggle_cntr = 0
            self.wiggle_total_cntr += 1
            self.doing_wiggle = False

        # Only start aligning once we've wiggled enough times (self.wiggles_before_align)
        if not self.align_axes and (self.wiggle_total_cntr > self.wiggles_before_align):
            print("Enabling axis alignment")
            self.align_axes = True
            self.ft_ema = self.ft_avg

        w_yaw = self.wiggle_w_yaw * self.wiggle_dir
        w_pitch = self.wiggle_w_pitch * self.wiggle_dir
        w: list[float] = [w_yaw, w_pitch, 0]
        v: list[float] = [0, 0, 0]

        twist_wiggle = np.array(v + w)
        # print(f"DEBUG wiggle v: {v}, w: {w}, wiggle_cntr: {self.wiggle_cntr}, align_yaw: {self.align_yaw}")
        return twist_wiggle

    # -----------------------------------------------------------------------------------------------------------
    def do_align_yaw(self) -> NDArray[np.float64]:
        yaw_speed = self.yaw_algn_kp * self.ft_avg[fy]
        w = np.array([yaw_speed, 0, 0])

        if self.ft_avg[tx] > 0:
            right_finger = True
        else:
            right_finger = False

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        # right_tip = [0, 0.030,  0.1134]
        # left_tip = [0, -0.030,  0.1134]
        # right_tip = [0, 0.040,  0.11]
        # left_tip = [0, -0.040,  0.11]
        # right_tip = [0, 0.030,  0.08]
        # left_tip = [0, -0.030,  0.08]
        r = right_tip if right_finger else left_tip
        r = np.array(r)
        v = -np.cross(w, r)
        twist_yaw = np.concatenate((v, w))

        # print(f"DEBUG yaw align ff: {[f"{self.ft_avg[fy]:.6f}"]}, v: {[f"{x:.6f}" for x in v]}, w: {[f"{x:.6f}" for x in w]}, right_finger: {right_finger}")
        return twist_yaw

    # -----------------------------------------------------------------------------------------------------------
    def do_align_pitch(self) -> NDArray[np.float64]:
        pitch_speed = self.pitch_algn_kp * self.ft_avg[fx] * -1
        w = np.array([0, pitch_speed, 0])

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        right = np.array(right_tip)
        left = np.array(left_tip)
        r = (right + left) / 2
        r = np.array([0, 0, r[2]])
        v = -np.cross(w, r)

        twist_pitch = np.concatenate((v, w))

        # print(f"DEBUG pitch align ff: {[f"{self.ft_avg[fx]:.6f}"]},v: {[f"{x:.6f}" for x in v]}, w: {[f"{x:.6f}" for x in w]}, r: {[f"{x:.6f}" for x in r]}, aligned: {self.pitch_aligned}")
        return twist_pitch

    # -----------------------------------------------------------------------------------------------------------
    def do_align_y(self) -> NDArray[np.float64]:
        twist_y = np.zeros(6)
        
        # direction based on which finger, inside or outside
        if self.ft_avg[tx] > 0:
            y_dir = -1 # go right
        else:
            y_dir = 1 # go left

        speed = 0
        if self.ft_contact_wrist[fz] <= -0.02 or abs(self.ft_contact_wrist[fy]) > 0.02:
            speed = 0.04

        v_ee: list[float] = [0, speed * y_dir, 0]
        w_ee: list[float] = [0, 0, 0]

        twist_y = np.array(v_ee + w_ee)

        return twist_y

# -----------------------------------------------------------------------------------------------------------
