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
        
        per_sec = 10.0/1000.0

        # Forward/keep contact
        self.forward_kp = 0.0675
        self.touched_once = False
        self.is_touching = False

        # Wiggle
        self.wiggle_cntr = 0
        self.wiggle_max = 60
        self.wiggle_dir = 1
        self.wiggle_samples = 5
        self.wiggle_total_cntr = 0
        self.doing_wiggle = False
        self.wiggle_w_yaw = 4.5 * per_sec
        self.wiggle_w_pitch = 3.0 * per_sec
        self.wiggle_yaw = True
        self.wiggle_pitch = True
        self.prev_wiggle_time = time.perf_counter()
        self.axes_aligned = False
        self.align_pos_world_frame = np.zeros(3)
        self.align_quat_world_frame = np.zeros(4)

        # Yaw align
        self.align_axes = False
        self.yaw_algn_kp = 0.144
        self.yaw_algn_thresh = 0.02
        self.yaw_aligned = False

        # Pitch align
        self.pitch_algn_kp = 0.144
        self.pitch_algn_thresh = 0.05
        self.pitch_aligned = False

        # Force/torque
        self.ft_contact_wrist = np.zeros(6)
        self.ft_ema = np.zeros(6)
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
                # self.state = 'find_object'
                # self.state = 'test'
                # self.state = 'do_wiggle'
                self.state = "do_initial_pos"

            case "do_initial_pos":
                self.state_initial_pos()

            case "open_gripper":
                self.state_open_gripper()
                # self.state = 'test'

            case "test":
                self.state_test()
                # self.test_wiggle()

            case "interact_perceive":
                self.state_interact_perceive()

            case _:
                pass

    # -----------------------------------------------------------------------------------------------------------
    def state_test(self) -> None:
        self.controller.mode = "velocity"
        # self.test_pitch()

    # -----------------------------------------------------------------------------------------------------------
    def state_initial_pos(self) -> None:
        self.controller.mode = "velocity"
        desired_pos = [self.initial_x, self.initial_y, self.initial_z]
        desired_euler_radians = [0, 90.0 * math.pi / 180.0, 0]

        self.controller.do_move_pos(pos=desired_pos, euler_rad=desired_euler_radians)

        pos_err, ang_err = self.controller.get_pos_error(desired_pos=desired_pos, desired_euler_radians=desired_euler_radians)

        if any(abs(x) > 0.02 for x in pos_err) or any(abs(x) > 0.12 for x in ang_err):
            return
        else:
            print(f"Reached initial pose, position err: {pos_err}, angular err: {ang_err}")
            self.state = "open_gripper"

    # -----------------------------------------------------------------------------------------------------------
    def state_open_gripper(self) -> None:
        self.controller.mode = "position"
        if self.controller.check_gripper_pos() is True:
            self.state = "interact_perceive"
            # self.state = 'test'
        else:
            self.controller.open_gripper()

    # -----------------------------------------------------------------------------------------------------------
    def state_interact_perceive(self) -> None:
        # Calculate exponential moving average to smooth out noise from ft readings
        self.update_ft_ema()

        # Move in the z direction to maintain light contact with object
        twist_contact = self.do_keep_contact()

        # Wiggle wrist to get the feeling force
        twist_wiggle = np.zeros(6)
        twist_yaw = np.zeros(6)
        twist_pitch = np.zeros(6)
        twist_grab = np.zeros(6)
        
        # First, wiggle to align pitch and yaw with the object
        if not self.axes_aligned:
            if self.is_touching or self.doing_wiggle:
                twist_wiggle = self.do_wiggle()

                # Max samples for feeling force should be wiggle samples * wiggle_max
                # Only start aligning once we have enough samples
                if self.ft_feeling_sum.shape[0] >= (self.wiggle_samples * self.wiggle_max):
                    self.ft_feeling_sum = self.ft_feeling_sum[1:]
                self.ft_feeling_sum = np.append(self.ft_feeling_sum, self.ft_ema.reshape(1, 6), axis=0)
                
            if self.align_axes:
                twist_yaw = self.do_align_yaw()
                twist_pitch = self.do_align_pitch()
        # Next, move in the Y direction
        else:
            twist_grab = np.array([0, ])
            pass

        # Sum up all the speed components
        speed_total = twist_contact + twist_wiggle + twist_yaw + twist_pitch + twist_grab

        # Apply speed
        self.controller.mode = "velocity"
        v_des = speed_total[:3]
        w_des = speed_total[3:]
        self.controller.do_move_velocity(v_des=v_des, w_des=w_des, link="wrist", world_frame=False)

    # -----------------------------------------------------------------------------------------------------------
    def update_ft_ema(self) -> None:
        self.ft_contact_wrist = self.controller.ft_contact_wrist
        self.ft_ema = (1.0 - self.alpha_ft_ema) * self.ft_ema + (self.alpha_ft_ema * np.array(self.ft_contact_wrist))
        self.controller.ft_ema = self.ft_ema.tolist()

    # -----------------------------------------------------------------------------------------------------------
    def do_keep_contact(self) -> np.ndarray:
        if self.ft_contact_wrist[fz] < self.thresh["fz"]:
            self.is_touching = True
            if not self.touched_once:
                self.touched_once = True
                self.forward_kp /= 3
                self.ft_ema = np.array(self.ft_contact_wrist)

        speed = self.forward_kp * (self.ft_contact_wrist[fz] - self.thresh["fz"])
        v_ee = [0, 0, speed]
        w_ee = [0, 0, 0]

        twist_contact = np.array(v_ee + w_ee)

        return twist_contact

    # -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self) -> np.ndarray:
        if self.wiggle_cntr == 0:
            self.doing_wiggle = True
            
            # self.yaw_aligned = True if abs(self.ft_ema[fy]) <= self.yaw_algn_thresh else False
            fy_avg = np.average(self.ft_feeling_sum[:, fy])
            self.yaw_aligned = True if abs(fy_avg) <= self.yaw_algn_thresh else False
            # self.pitch_aligned = True if abs(self.ft_ema[fx]) <= self.pitch_algn_thresh else False
            fx_avg = np.average(self.ft_feeling_sum[:, fx])
            self.pitch_aligned = True if abs(fx_avg) <= self.pitch_algn_thresh else False
            print(f"DEBUG: yaw avg: {fy_avg}, yaw aligned: {self.yaw_aligned}, pitch avg: {fx_avg}, pitch aligned: {self.pitch_aligned}")

            self.wiggle_yaw = False if self.yaw_aligned else True
            self.wiggle_pitch = False if self.pitch_aligned else True
            
            if self.yaw_aligned and self.pitch_aligned and (self.wiggle_total_cntr > self.wiggle_samples):
                print("Both axes aligned, stopping wiggle")
                self.align_pos_world_frame, self.align_quat_world_frame = self.controller.get_wrist_pos()
                self.doing_wiggle = False
                self.align_axes = False
                self.axes_aligned = True
                return np.zeros(6)
            # now = time.perf_counter()
            # print(f"DEBUG, wiggle duration: {now - self.prev_wiggle_time:.6f}s")
            # self.prev_wiggle_time = now

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

        w_yaw = self.wiggle_w_yaw * self.wiggle_dir if self.wiggle_yaw is True else 0
        w_pitch = self.wiggle_w_pitch * self.wiggle_dir if self.wiggle_pitch is True else 0
        w = [w_yaw, w_pitch, 0]
        v = [0, 0, 0]

        twist_wiggle = np.array(v + w)
        # print(f"DEBUG wiggle v: {v}, w: {w}, wiggle_cntr: {self.wiggle_cntr}, align_yaw: {self.align_yaw}")
        return twist_wiggle

    # -----------------------------------------------------------------------------------------------------------
    def do_align_yaw(self) -> np.ndarray:
        if self.yaw_aligned:
            return np.zeros(6)
        else:
            yaw_speed = self.yaw_algn_kp * self.ft_ema[fy]

        w = np.array([yaw_speed, 0, 0])

        if self.ft_ema[tx] > 0:
            right_finger = True
        else:
            right_finger = False

        # right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        # right_tip = [0, 0.030,  0.1134]
        # left_tip = [0, -0.030,  0.1134]
        right_tip = [0, 0.040,  0.13]
        left_tip = [0, -0.040,  0.13]
        r = right_tip if right_finger else left_tip
        r = np.array(r)
        v = -np.cross(w, r)
        twist_yaw = np.concatenate((v, w))

        print(f"DEBUG yaw align ff: {self.ft_ema[fy]}, v: {v}, w: {w}, right_finger: {right_finger}")
        return twist_yaw

    # -----------------------------------------------------------------------------------------------------------
    def do_align_pitch(self) -> np.ndarray:
        if self.pitch_aligned:
            return np.zeros(6)
        else:
            pitch_speed = self.pitch_algn_kp * self.ft_ema[fx] * -1.0

        w = np.array([0, pitch_speed, 0])

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        right = np.array(right_tip)
        left = np.array(left_tip)
        r = (right + left) / 2
        r = np.array([0, 0, r[2]])
        v = -np.cross(w, r)

        twist_pitch = np.concatenate((v, w))

        print(f"DEBUG pitch align ff: {self.ft_ema[fx]}, v: {v}, w: {w}, r: {r}, aligned: {self.pitch_aligned}")
        return twist_pitch


# -----------------------------------------------------------------------------------------------------------
