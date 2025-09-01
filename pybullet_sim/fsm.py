import math
import time
import numpy as np
from typing import List

fx = 0
fy = 1
fz = 2
tx = 3
ty = 4
tz = 5


class FSM:
    def __init__(self, controller, initial_x, initial_y, initial_z) -> None:

        self.state = "start"
        self.controller = controller
        self.sim_lock = controller.sim_lock
        self.do_timers = controller.do_timers
        self.interval = controller.interval
        self.thread_cntr_max = controller.thread_cntr_max
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_z = initial_z
        self.startup_cntr = 0
        self.orn = [0, 0, 0, 1]
        self.done = False
        
        per_sec = self.interval

        # Forward/keep contact
        # self.forward_kp = 0.0675
        self.forward_kp_initial = 0.05
        self.forward_kp = 0.02
        self.touched_once = False
        self.is_touching = False

        # Wiggle
        self.wiggle_cntr = 0
        # self.wiggle_max = 60
        # wiggles_per_sec = 0.6
        wiggles_per_sec = 0.8
        self.wiggle_max = math.ceil(1/self.interval * wiggles_per_sec) + (4 - math.ceil(1/self.interval * wiggles_per_sec) % 4)
        self.wiggle_dir = 1
        self.wiggle_samples = 5
        self.wiggle_total_cntr = 0
        self.doing_wiggle = False
        # self.wiggle_w_yaw = 1.0 * per_sec
        self.wiggle_w_yaw = 4.0 * per_sec
        self.wiggle_w_pitch = 4.0 * per_sec
        self.prev_wiggle_time = time.perf_counter()
        self.axes_aligned = False
        self.axes_aligned_cntr = 0
        self.align_pos_world_frame = np.zeros(3)
        self.align_quat_world_frame = np.zeros(4)

        # Yaw align
        self.align_axes = False
        # self.yaw_algn_kp = 0.144
        self.yaw_algn_kp = 0.8
        self.yaw_algn_thresh = 0.01
        self.yaw_aligned = False

        # Pitch align
        # self.pitch_algn_kp = 0.144
        self.pitch_algn_kp = 0.4
        self.pitch_algn_thresh = 0.01
        self.pitch_aligned = False

        # Force/torque
        self.ft_contact_wrist = np.zeros(6)
        self.ft_ema = np.zeros(6)
        self.ft_contact_sum = np.zeros((1, 6))
        self.alpha_ft_ema = 0.002
        self.ft_feeling_sum = np.zeros((1, 6))
        self.ft_feeling = np.zeros(6)

        self.thresh = {
            "fx": 0.1,  # threshold for force in x direction
            "fz": -0.2,  # threshold for force in z direction
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
                self.controller.reset_home_pos()
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
        # desired_pos = [self.initial_x, self.initial_y, self.initial_z]
        # desired_euler_radians = [0, 90.0 * math.pi / 180.0, 0]
        # desired_euler_radians = [0, 100.0 * math.pi / 180.0, 0]
        desired_pos = [self.initial_x, self.initial_y, self.initial_z + 0.05]
        # desired_euler_radians = [0, 75.0 * math.pi / 180.0, 0]
        desired_euler_radians = [15.0 * math.pi / 180.0, 75.0 * math.pi / 180.0, 15.0 * math.pi / 180.0]

        self.controller.do_move_pos(pos=desired_pos, euler_rad=desired_euler_radians)

        pos_err, ang_err = self.controller.get_pos_error(desired_pos=desired_pos, desired_euler_radians=desired_euler_radians)

        if any(abs(x) > 0.02 for x in pos_err) or any(abs(x) > 0.15 for x in ang_err):
            return
        else:
            print(f"Reached initial pose, position err: {pos_err}, angular err: {ang_err}")
            self.state = "open_gripper"

    # -----------------------------------------------------------------------------------------------------------
    def state_open_gripper(self) -> None:
        self.controller.mode = "position"
        if self.controller.check_gripper_pos() is True:
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
            
            # Max samples for feeling force should be wiggle samples * wiggle_max
            # Only start aligning once we have enough samples
            self.ft_feeling_sum = np.append(self.ft_feeling_sum, self.ft_ema.reshape(1, 6), axis=0)
            if self.ft_feeling_sum.shape[0] >= (self.wiggle_samples * self.wiggle_max):
                self.ft_feeling_sum = self.ft_feeling_sum[1:]
            
            self.ft_contact_sum = np.append(self.ft_contact_sum, self.ft_contact_wrist.reshape(1, 6), axis=0)
            if self.ft_contact_sum.shape[0] >= (self.wiggle_samples * self.wiggle_max):
                self.ft_contact_sum = self.ft_contact_sum[1:]
                
            self.ft_ema = np.average(self.ft_contact_sum, axis=0)
            self.controller.ft_ema = self.ft_ema.tolist()
            
            if self.align_axes:
                twist_yaw = self.do_align_yaw()
                twist_pitch = self.do_align_pitch()
                pass

        # Next, move in the Y direction
        if self.axes_aligned:
            relative_pos, relative_quat = self.controller.get_relative_pos(self.align_pos_world_frame, self.align_quat_world_frame)
            print(f"DEBUG relative pos: {[f"{x:.3f}" for x in relative_pos]}")
            if abs(relative_pos[1]) > 0.08 or abs(relative_pos[2]) > 0.04:
                self.done = True
                self.state = "done"
                self.controller.do_move_velocity(v_des=[0, 0, 0], w_des=[0, 0, 0], link="wrist", world_frame=False)
                print("DEBUG reached max movement")
                return
            if self.ft_contact_wrist[fz] < -0.05:
                twist_grab = self.do_align_y()

        # Sum up all the speed components
        speed_total = twist_contact + twist_wiggle + twist_yaw + twist_pitch + twist_grab

        # Apply speed
        self.controller.mode = "velocity"
        v_des = speed_total[:3]
        w_des = speed_total[3:]
        self.controller.do_move_velocity(v_des=v_des, w_des=w_des, link="wrist", world_frame=False)

    # -----------------------------------------------------------------------------------------------------------
    def do_keep_contact(self) -> np.ndarray:
        # small offset so if we are just under, it's still "touching"
        if self.ft_contact_wrist[fz] <= self.thresh["fz"] + 0.1:
            self.is_touching = True
            if not self.touched_once:
                self.touched_once = True
        else:
            self.is_touching = False

        if not self.touched_once and not self.axes_aligned:
            kp = self.forward_kp_initial
        else:
            kp = self.forward_kp

        force_diff = self.ft_contact_wrist[fz] - self.thresh["fz"]
        speed = kp * force_diff
        
        v_ee = [0, 0, speed]
        w_ee = [0, 0, 0]

        twist_contact = np.array(v_ee + w_ee)

        return twist_contact

    # -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self) -> np.ndarray:
        if self.wiggle_cntr == 0:
            self.doing_wiggle = True
            
            if self.wiggle_total_cntr > self.wiggle_samples:
                self.yaw_aligned = True if abs(self.ft_ema[fy]) <= self.yaw_algn_thresh else False

                self.pitch_aligned = True if abs(self.ft_ema[fx]) <= self.pitch_algn_thresh else False
                print(f"DEBUG: WIGGLE yaw ema: {self.ft_ema[fy]}, yaw aligned: {self.yaw_aligned}, pitch ema: {self.ft_ema[fx]}, pitch aligned: {self.pitch_aligned}")

                if self.yaw_aligned and self.pitch_aligned:
                    self.axes_aligned_cntr += 1
                    print(f"DEBUG: WIGGLE axes aligned counter: {self.axes_aligned_cntr}")
                    if self.axes_aligned_cntr >= 3:
                        print("Both axes aligned, stopping wiggle")
                        self.align_pos_world_frame, self.align_quat_world_frame = self.controller.get_wrist_pos()
                        self.doing_wiggle = False
                        self.align_axes = False
                        self.axes_aligned = True
                        return np.zeros(6)
                else:
                    self.axes_aligned_cntr = 0

        self.wiggle_cntr += 1

        if (self.wiggle_cntr <= self.wiggle_max / 4 or self.wiggle_cntr > 3 * self.wiggle_max / 4):
            self.wiggle_dir = -1
        else:
            self.wiggle_dir = 1

        if self.wiggle_cntr > self.wiggle_max:
            self.wiggle_cntr = 0
            self.wiggle_total_cntr += 1
            self.doing_wiggle = False
            
        if not self.align_axes and (self.wiggle_total_cntr > self.wiggle_samples):
            print("Enabling axis alignment")
            self.align_axes = True

        # w_yaw = self.wiggle_w_yaw * self.wiggle_dir if not self.yaw_aligned else 0
        w_yaw = self.wiggle_w_yaw * self.wiggle_dir
        # w_pitch = self.wiggle_w_pitch * self.wiggle_dir if not self.pitch_aligned else 0
        w_pitch = self.wiggle_w_pitch * self.wiggle_dir
        w = [w_yaw, w_pitch, 0]
        v = [0, 0, 0]
        # v = [w_pitch, w_yaw, 0]
        # w = [0, 0, 0]

        twist_wiggle = np.array(v + w)
        # print(f"DEBUG wiggle v: {v}, w: {w}, wiggle_cntr: {self.wiggle_cntr}, align_yaw: {self.align_yaw}")
        return twist_wiggle

    # -----------------------------------------------------------------------------------------------------------
    def do_align_yaw(self) -> np.ndarray:
        yaw_speed = self.yaw_algn_kp * self.ft_ema[fy]
        max_yaw_speed = 0.2
        if abs(yaw_speed) > max_yaw_speed:
            yaw_speed = np.sign(yaw_speed) * max_yaw_speed

        w = np.array([yaw_speed, 0, 0])

        if self.ft_ema[tx] > 0:
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

        # print(f"DEBUG yaw align ff: {[f"{self.ft_ema[fy]:.6f}"]}, v: {[f"{x:.6f}" for x in v]}, w: {[f"{x:.6f}" for x in w]}, right_finger: {right_finger}")
        return twist_yaw

    # -----------------------------------------------------------------------------------------------------------
    def do_align_pitch(self) -> np.ndarray:
        # if self.pitch_aligned:
        #     return np.zeros(6)
        # else:
        #     pitch_speed = self.pitch_algn_kp * self.ft_ema[fx] * -1.0
        pitch_speed = self.pitch_algn_kp * self.ft_ema[fx] * -1
        max_pitch_speed = 0.2
        if abs(pitch_speed) > max_pitch_speed:
            pitch_speed = np.sign(pitch_speed) * max_pitch_speed

        w = np.array([0, pitch_speed, 0])

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        right = np.array(right_tip)
        left = np.array(left_tip)
        r = (right + left) / 2
        r = np.array([0, 0, r[2]])
        v = -np.cross(w, r)

        twist_pitch = np.concatenate((v, w))

        # print(f"DEBUG pitch align ff: {self.ft_ema[fx]}, v: {v}, w: {w}, r: {r}, aligned: {self.pitch_aligned}")
        return twist_pitch

    # -----------------------------------------------------------------------------------------------------------
    def do_align_y(self) -> np.ndarray:
        twist_y = np.zeros(6)
        
        # speed based on which finger, inside or outside
        if self.ft_ema[tx] > 0:
            y_dir = -1 # go right
        else:
            y_dir = 1 # go left

        speed = 0
        if self.ft_contact_wrist[fz] <= -0.01:
            speed = 0.005
            
        v_ee = [0, speed * y_dir, 0]
        w_ee = [0, 0, 0]

        twist_y = np.array(v_ee + w_ee)

        return twist_y

# -----------------------------------------------------------------------------------------------------------
