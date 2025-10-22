import math, time
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from controller import Controller

fx = 0
fy = 1
fz = 2
tx = 3
ty = 4
tz = 5

class FSM:
    def __init__(self, 
                 controller: "Controller", 
                 initial_pos: NDArray[np.float64], 
                 initial_orn: NDArray[np.float64],
                 no_move: bool = False
    ) -> None:

        self.state = "start"
        self.controller = controller
        self.sim_lock = controller.sim_lock
        self.do_timers = controller.do_timers
        self.interval = controller.interval
        per_sec = self.interval
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn
        self.no_move = no_move

        self.max_speed = 0.15
        self.prev_speed = np.zeros(6)
        self.prev_wiggle = np.zeros(6)
        self.prev_yaw = np.zeros(6)

        # Flags
        self.done = False
        self.touched_once = False
        self.is_touching = False
        self.doing_wiggle = False
        self.align_axes = False

        # Wiggle
        self.wiggle_cntr = 0
        self.wiggle_dir = 1
        self.wiggles_before_align = 2
        self.wiggle_total_cntr = 0
        wiggles_per_sec = 2.0
        wiggle_kp = 180.0
        self.wiggle_max = math.ceil(1/self.interval * 1/wiggles_per_sec) + (4 - math.ceil(1/self.interval * 1/wiggles_per_sec) % 4)
        self.wiggle_w_yaw = wiggle_kp/self.wiggle_max * per_sec
        self.wiggle_w_pitch = wiggle_kp/self.wiggle_max * per_sec

        self.align_pos_world = np.zeros(3)
        self.align_quat_world = np.zeros(4)
        self.prev_pos_world = np.zeros(3)
        self.prev_quat_world = np.zeros(4)
        self.rel_angle_sum = np.zeros(3)

        # Force/torque
        self.ft_contact_wrist = np.zeros(6)
        self.ft_ema = np.zeros(6)
        self.ft_contact_sum = np.zeros((1, 6))
        self.ft_avg = np.zeros((1, 6))
        self.alpha_ft_ema = 0.9

        self.kp = {
            "z_initial": 0.8,
            "z": 0.11,
            "grab": 0.3,
            "grab_min": 0.08,
        }
        self.kp_cur = self.kp["z"]

        self.beta = {
            "aln_y": 200.0,
            "aln_yaw": 50.0,
            "aln_pitch": 320.0,
        }
        self.recovery_rate = 1.0/400.0

        self.thresh = {
            "fx": 0.01,  # threshold for force in x direction (pitch algn, unused)
            "fz": -0.09,  # threshold for force in z direction (touch)
            "fz_max": -0.15,  # threshold for force in z direction
            "fz_min": -0.05,
            "fy": 0.003,  # threshold for force in y direction (yaw algn)
            "tx": 0.0012,
            "ty": 0.0002,  # threshold for torque in y direction (pitch algn)
            "tz": 0.0005,
        }

        self.aligned = {
            "yaw": False,
            "pitch": False,
            "axes": False,
            "roll": False
        }
        self.check_fz = False
        self.max_y = False
        self.tx_flipped = False
        self.tx_sign = 0
        self.tz_flipped = False
        self.tz_sign = 0
        self.prev_roll_angle = 0

        self.testcntr = 0
        self.timers: dict[str, float] = defaultdict(float)
        self.loop_timers: dict[str, float] = defaultdict(float)

    # -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> dict[str, float]:
        """ Finite state machine to determine next control action """
        self.timers.clear()
        self.loop_timers["prev_time"] = time.perf_counter()
        self.next_timer("fsm_top")
        
        match self.state:
            case "start":
                self.controller.reset_robot_pos()
                self.state = "do_initial_pos"
                self.next_timer("state_start")

            case "do_initial_pos":
                self.state_initial_pos()
                self.next_timer("state_initial_pos")

            case "open_gripper":
                self.state_open_gripper()
                self.next_timer("state_open_gripper")

            case "interact_perceive":
                if self.no_move:
                    self.controller.toggle_pause_sim(True)
                    self.state = "done"
                else:
                    self.state_interact_perceive()

                self.next_timer("state_interact_perceive")

            case "done":
                self.controller.toggle_pause_sim(True)

            case "test":
                self.state_test()

            case _:
                pass

        return self.timers

    def next_timer(self, timer_name: str) -> None:
        """ Record incremental time between timer calls """
        
        if self.do_timers:
            cur_time = time.perf_counter()
            self.timers[timer_name] = cur_time - self.loop_timers["prev_time"]
            self.loop_timers["prev_time"] = cur_time

    def state_test(self) -> None:
        maxcnt = 400
        mydir = np.sign(np.cos(2 * np.pi * self.testcntr / maxcnt))

        self.testcntr += 1
        if self.testcntr >= maxcnt:
            self.testcntr = 0

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist()
        r = (np.array(right_tip) + np.array(left_tip)) / 2
        
        speed = 0.2

        w_y = np.array([speed * mydir, 0, 0])
        # w_y = np.array([0, 0, 0])
        # r_y = np.array(left_tip)
        # v_y = -np.cross(w_y, r_y)
        v_y = -np.cross(w_y, r)
        
        w_p = np.array([0, speed * mydir, 0])
        # w_p = np.array([0, 0, 0])
        # r_p = (np.array(right_tip) + np.array(left_tip)) / 2
        # r_p = np.array([0, 0, r_p[2]])
        # v_p = -np.cross(w_p, r_p)
        v_p = -np.cross(w_p, r)

        w_y = np.zeros(3)
        v_y = np.zeros(3)
        # w_p = np.zeros(3)
        # v_p = np.zeros(3)
        
        v = v_y + v_p
        w = w_y + w_p
        twist = np.concatenate((v, w))
        
        self.controller.mode = "velocity"
        self.controller.set_next_dq(v=twist[:3], w=twist[3:], world_frame=False)

    # -----------------------------------------------------------------------------------------------------------
    def state_initial_pos(self) -> None:
        """ Move to desired initial position """
        
        self.controller.mode = "velocity"
        ret = self.controller.do_move_pos(pos_world=self.initial_pos, quat_world=self.initial_orn)
        if ret:
            self.state = "open_gripper"
            print(f"Waiting for gripper to open ...")

    # -----------------------------------------------------------------------------------------------------------
    def state_open_gripper(self) -> None:
        """ Open grippers before starting """
        
        if self.controller.is_gripper_open():
            print(f"Gripper opened")
            self.state = "interact_perceive"
            # self.controller.toggle_pause_sim(True)
            # self.state = "test"

        else:
            self.controller.open_gripper()

    # -----------------------------------------------------------------------------------------------------------
    def state_interact_perceive(self) -> None:
        """
        Main interact/perceive control function
        
        1. Move in wrist's +Z direction to find object
        2. (Always on) Maintain light contact (Force Z) with object, increasing Z movement speed if force decreases
        3. Wiggle wrist's pitch & yaw to get feeling force/torques
        4. Align wrist's pitch and yaw axes with object based on feeling forces/torques
        5. Once aligned, move in wrist's Y direction to slide along object & attempt to grab
        6. If max Y movement reached, adjust roll to attempt to grab
        7. If max Z movement reached after alignment, stop and assume object has been grasped
        8. If max Y, Z, and roll movement reached, stop and assume failure
        """

        self.ft_contact_wrist = np.array(self.controller.ft_contact_wrist)
        alpha_ema = self.alpha_ft_ema
        self.ft_ema = alpha_ema * self.ft_contact_wrist + (1 - alpha_ema) * self.ft_ema
        self.controller.ft_ema = self.ft_ema.tolist()

        # Move in the z direction to maintain light contact with object
        twist_contact = self.do_keep_contact()
        
        twist_wiggle = np.zeros(6)
        twist_yaw = np.zeros(6)
        twist_pitch = np.zeros(6)
        twist_grab = np.zeros(6)
        twist_roll = np.zeros(6)

        # First, wiggle to get feeling force & align pitch and yaw with the object
        # if not self.aligned["axes"] and (self.is_touching or self.doing_wiggle):
        if not self.aligned["axes"] and self.touched_once:
            twist_wiggle = self.do_wiggle()

            self.update_avg_ft()

            # Only start aligning once we have enough samples (self.wiggles_before_align * self.wiggle_max)
            # if self.wiggle_total_cntr > self.wiggles_before_align and self.is_touching:
            if self.wiggle_total_cntr > self.wiggles_before_align:
                twist_yaw = self.do_align_yaw()
                twist_pitch = self.do_align_pitch()

        if self.aligned["axes"]:

            self.update_avg_ft()
            relative_pos, relative_angle = self.controller.get_relative_pos(pos_world=self.align_pos_world, quat_world=self.align_quat_world, link_name="wrist")
            
            # End if we reach max Z movement
            if abs(relative_pos[2]) > 0.04:
                print(f"DEBUG reached max movement, relative pos: {[f"{x:.3f}" for x in relative_pos]}")
                self.done = True
                self.state = "done"
                self.controller.set_next_dq(v=np.zeros(3), w=np.zeros(3), world_frame=False)
                return

            # Roll if TX flips sign or max Y movement reached
            if abs(relative_pos[1]) >= 0.08:
                self.max_y = True
                
            if self.tx_sign * np.sign(self.ft_ema[tx]) < 0 and not self.tx_flipped:
                print(f"DEBUG tx flipped!")
                self.tx_flipped = True

            self.tx_sign = np.sign(self.ft_ema[tx])
                
            if self.max_y or self.tx_flipped:
            # if (self.max_y or abs(self.ft_ema[tx]) < self.thresh["tx"]) and (self.ft_ema[fz] < self.thresh["fz_min"]):
            # if (self.max_y or self.tx_flipped) and (self.ft_ema[fz] < self.thresh["fz_min"]):
            # if (self.max_y or (self.tx_flipped and abs(self.ft_ema[tx]) < self.thresh["tx"])) and (self.ft_ema[fz] < self.thresh["fz_min"]):
            # if (self.max_y or self.tx_flipped) and (self.ft_ema[fz] < self.thresh["fz_min"]):
                # End if max roll
                if ((abs(self.prev_roll_angle) > 150.0) 
                    and (np.sign(self.prev_roll_angle) * np.sign(relative_angle[2])) < 0):
                    
                    print(f"DEBUG COULD NOT ALIGN ROLL, stopping. Relative pos: {[f"{x:.3f}" for x in relative_pos]}, relative angle: {[f"{x:.3f}" for x in relative_angle]}, prev angle: {self.prev_roll_angle:.3f}, rel angle: {relative_angle[2]:.3f}")
                    self.done = True
                    self.state = "done"
                    self.controller.set_next_dq(v=np.zeros(3), w=np.zeros(3), world_frame=False)
                    return
                
                # if (self.tz_sign * np.sign(self.ft_ema[tz]) < 0 and abs(self.ft_ema[tz]) < self.thresh["tz"]):
                
                # if (abs(self.ft_ema[tz]) >= self.thresh["tz"]):
                if (self.tz_sign * np.sign(self.ft_ema[tz]) < 0 and self.ft_ema[fz] <= self.thresh["fz"]):
                    print(f"DEBUG tz flipped!")
                    self.tz_flipped = True
                    self.done = True
                    self.state = "done"
                    self.controller.set_next_dq(v=np.zeros(3), w=np.zeros(3), world_frame=False)
                    return
                else:
                    self.prev_roll_angle = relative_angle[2]
                    # print(f"DEBUG doing roll. Relative pos: {[f"{x:.3f}" for x in relative_pos]}, relative angle: {[f"{x:.3f}" for x in relative_angle]}, prev angle: {self.prev_roll_angle:.3f}, rel angle: {relative_angle[2]:.3f}")
                    self.tz_sign = np.sign(self.ft_ema[tz])
                    twist_roll = self.do_align_roll()
                    
                # self.tz_sign = np.sign(self.ft_ema[tz])
            else:
                twist_grab = self.do_align_y()

        # Sum up all the speed components
        speed_total = twist_contact + twist_wiggle + twist_yaw + twist_pitch + twist_grab + twist_roll

        # Apply speed
        self.controller.mode = "velocity"
        self.controller.set_next_dq(v=speed_total[:3], w=speed_total[3:], world_frame=False)
        self.prev_speed = speed_total.copy()
        self.prev_wiggle = twist_wiggle.copy()
        self.prev_yaw = twist_yaw.copy()

    # -----------------------------------------------------------------------------------------------------------
    def update_avg_ft(self) -> None:
        """ Update avg forces/torques with new sample """
        self.ft_contact_sum = np.append(self.ft_contact_sum, self.ft_contact_wrist.reshape(1, 6), axis=0)
        if self.ft_contact_sum.shape[0] >= (self.wiggles_before_align * self.wiggle_max):
            self.ft_contact_sum = self.ft_contact_sum[1:]
        
        self.ft_avg = np.average(self.ft_contact_sum, axis=0)
        self.controller.ft_feeling = self.ft_avg.tolist()

    # -----------------------------------------------------------------------------------------------------------
    def do_keep_contact(self) -> NDArray[np.float64]:
        """ 
        Move in the z direction to maintain light contact with object based on force in z direction 

        Returns:
            twist_contact: 6D twist to apply to wrist to maintain contact
        """

        # small offset so if we are just under, it's still "touching"
        # if self.ft_ema[fz] <= self.thresh["fz_min"] and self.ft_ema[fz] > self.thresh["fz_max"]:
        # if self.ft_ema[fz] < self.thresh["fz"] + 0.03:
        if self.ft_ema[fz] <= self.thresh["fz_min"] and self.ft_ema[fz] > self.thresh["fz_max"]:
            self.is_touching = True
            if not self.touched_once:
                self.touched_once = True
        else:
            self.is_touching = False

        if not self.touched_once:
            dist = self.controller.get_closest_point()
            if dist > 0.005:
                kp = self.kp["z_initial"]
            else:
                kp = self.kp["z"]
        elif self.aligned["axes"]:
            if self.ft_ema[fz] > self.thresh["fz_min"]:
                if self.kp_cur < self.kp["grab"]:
                    self.kp_cur += (self.kp["grab"] - self.kp["grab_min"]) * self.recovery_rate
                else:
                    self.kp_cur = self.kp["grab"]
            else:
                # if self.kp_cur > self.kp["grab_min"]:
                #     self.kp_cur -= (self.kp["grab"] - self.kp["grab_min"]) * self.recovery_rate
                # else:
                #     self.kp_cur = self.kp["grab_min"]
                self.kp_cur = self.kp["grab_min"]

            kp = self.kp_cur
        else:
            kp = self.kp["z"]

        force_diff = self.ft_ema[fz] - self.thresh["fz"]
        speed = kp * force_diff
        if abs(speed) > self.max_speed:
            speed = self.max_speed * np.sign(speed)
        twist_contact = np.array([0, 0, speed, 0, 0, 0])

        # print(f"DEBUG contact: fz: {self.ft_contact_wrist[fz]:.4f}, fz_ema: {self.ft_ema[fz]:.4f}, speed: {speed:.4f}, kp: {kp:.4f}")

        return twist_contact

    # -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self) -> NDArray[np.float64]:
        """ 
        Wiggle wrist in pitch and yaw to get feeling forces/torques

        Returns:
            twist_wiggle: 6D twist to apply to wrist to wiggle in pitch and yaw
        """
        
        if self.wiggle_cntr == 0:
            self.doing_wiggle = True
            
            if self.wiggle_total_cntr > self.wiggles_before_align:
                self.aligned["yaw"] = True if abs(self.ft_avg[fy]) <= self.thresh["fy"] else False
                self.aligned["pitch"] = True if abs(self.ft_avg[ty]) <= self.thresh["ty"] else False
                print(f"DEBUG: WIGGLE fy(yaw) avg/ema/aligned: {self.ft_avg[fy]:.6f}/{self.ft_ema[fy]:.6f}/{self.aligned["yaw"]}, ty(pitch) avg/ema/aligned: {self.ft_avg[ty]:.6f}/{self.ft_ema[ty]:.6f}/{self.aligned["pitch"]}")

                if self.aligned["yaw"] and self.aligned["pitch"]:
                    print("Both axes aligned, stopping wiggle")
                    self.align_pos_world, self.align_quat_world = self.controller.get_pos(link_name="wrist")
                    self.doing_wiggle = False
                    self.aligned["axes"] = True
                    self.align_axes = False
                    return np.zeros(6)

        self.wiggle_cntr += 1
        self.wiggle_dir = np.sign(np.cos(2 * np.pi * self.wiggle_cntr / self.wiggle_max))

        if self.wiggle_cntr > self.wiggle_max:
            self.wiggle_cntr = 0
            self.wiggle_total_cntr += 1
            self.doing_wiggle = False

        # Only start aligning once we have enough samples (self.wiggles_before_align * self.wiggle_max)
        if not self.align_axes and (self.wiggle_total_cntr > self.wiggles_before_align):
            self.align_axes = True
            self.ft_ema = self.ft_avg

        w_yaw = self.wiggle_w_yaw * self.wiggle_dir
        w_pitch = self.wiggle_w_pitch * self.wiggle_dir
        w: list[float] = [w_yaw, w_pitch, 0.0]
        v: list[float] = [0.0, 0.0, 0.0]

        twist_wiggle = np.array(v + w)
        # print(f"DEBUG wiggle v: {v}, w: {w}, wiggle_cntr: {self.wiggle_cntr}, align_yaw: {self.align_yaw}")
        return twist_wiggle

    # -----------------------------------------------------------------------------------------------------------
    def do_align_yaw(self) -> NDArray[np.floating[Any]]:
        """
        Align wrist's yaw axis with object based on Y force and X torque
            Y force determines speed
            X torque determines finger

        Returns:
            twist_yaw: 6D twist to apply to wrist to align yaw axis
        """
        
        speed = self.max_speed * math.tanh(self.beta["aln_yaw"] * self.ft_avg[fy])
        w = np.array([speed, 0, 0])

        if self.ft_avg[tx] > 0:
            right_finger = True
        else:
            right_finger = False

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist()
        # r = (np.array(right_tip) + np.array(left_tip)) / 2
        r = np.array(right_tip) if right_finger else np.array(left_tip)
        r[0] = 0.0
        r = np.array([0.0, 0.0, r[2]])
        v = -np.cross(w, r)
        twist_yaw = np.concatenate((v, w))

        # print(f"DEBUG yaw aligned: {self.aligned["yaw"]}, "
        #       f"fy avg/ema/thr: {self.ft_avg[fy]:.6f}/{self.ft_ema[fy]:.6f}/{self.thresh["fy"]:.6f}, "
        #       f"tx avg/ema: {self.ft_avg[tx]:.6f}/{self.ft_ema[tx]:.6f}/{self.thresh["tx"]:.6f}, "
        #       f"twist: {[f"{x:.6f}" for x in twist_yaw]}")
        
        return twist_yaw

    # -----------------------------------------------------------------------------------------------------------
    def do_align_pitch(self) -> NDArray[np.float64]:
        """
        Align wrist's pitch axis with object based on X force

        Returns:
            twist_pitch: 6D twist to apply to wrist to align pitch axis
        """

        speed = -1 * self.max_speed * math.tanh(self.beta["aln_pitch"] * self.ft_avg[ty])
        if abs(speed) > self.max_speed:
            speed = self.max_speed * np.sign(speed)
        w = np.array([0, speed, 0])

        right_tip, left_tip = self.controller.get_fingertip_pos_wrist()
        r = (np.array(right_tip) + np.array(left_tip)) / 2
        r = np.array([0, 0, r[2]])
        v = -np.cross(w, r)

        twist_pitch = np.concatenate((v, w))
        # print(f"DEBUG pch aligned: {self.aligned["pitch"]}: ty avg/ema/thr: {self.ft_avg[ty]:.6f}/{self.ft_ema[ty]:.6f}/{self.thresh["ty"]:.6f}, fx avg/ema/thr: {self.ft_avg[fx]:.6f}/{self.ft_ema[fx]:.6f}/{self.thresh["fx"]:.6f}, twist: {[f"{x:.6f}" for x in twist_pitch]}")

        return twist_pitch

    # -----------------------------------------------------------------------------------------------------------
    def do_align_y(self) -> NDArray[np.float64]:
        """
        Move in wrist's Y direction to slide along object

        Returns:
            twist_y: 6D twist to apply to wrist to move in Y direction
        """
        twist_y = np.zeros(6)
        
        speed = -1 * self.max_speed * math.tanh(self.beta["aln_y"] * self.ft_ema[tx])
        if abs(speed) > self.max_speed:
            speed = self.max_speed * np.sign(speed)

        v_ee: list[float] = [0, speed, 0]
        w_ee: list[float] = [0, 0, 0]

        twist_y = np.array(v_ee + w_ee)

        # print(f"DEBUG y align: tx: {self.ft_ema[tx]:.6f}, y align: {[f"{x:.6f}" for x in twist_y]}")
        return twist_y

    # -----------------------------------------------------------------------------------------------------------
    def do_align_roll(self) -> NDArray[np.float64]:
        """
        Adjust wrist's roll to attempt to grab object

        Returns:
            twist_roll: 6D twist to apply to wrist to adjust roll
        """
        twist_roll = np.zeros(6)

        self.prev_pos_world, self.prev_quat_world = self.controller.get_pos(link_name="wrist")
        max_speed_roll = 0.3

        # speed = self.max_speed * np.sign(self.ft_ema[tz]) * (1 + (self.ft_ema[tz] / 0.007))
        # speed = max_speed_roll * np.sign(self.ft_ema[tz]) * (1 + (self.ft_ema[tz] / 0.0003))
        # speed = max_speed_roll * np.sign(self.ft_ema[tz]) * (1 + (self.ft_ema[tz] / self.thresh["tz"]))
        speed = max_speed_roll * (1-(abs(self.ft_ema[tz]) / self.thresh["tz"]))
        # speed = 0.3

        # speed = 0
        # if abs(self.ft_ema[tz]) > 0.001:
        #     speed = max_speed_roll * np.sign(self.ft_ema[tz])

        if abs(speed) > max_speed_roll:
            speed = max_speed_roll * np.sign(speed)

        v_ee: list[float] = [0, 0, 0]
        w_ee: list[float] = [0, 0, speed]

        twist_roll = np.array(v_ee + w_ee)

        # print(f"DEBUG roll align: speed: {speed:.4f}, tz ema/avg: {self.ft_ema[tz]:.6f}/{self.ft_avg[tz]:.6f}, twist: {[f"{x:.6f}" for x in twist_roll]}")
        return twist_roll
    
    
# -----------------------------------------------------------------------------------------------------------