import math
import numpy as np

fx = 0
fy = 1
fz = 2
tx = 3
ty = 4
tz = 5

class FSM():
    def __init__(self, 
                 controller, 
                 initial_x, initial_y, initial_z) -> None:
        
        self.state = 'start'
        self.controller = controller
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_z = initial_z
        self.sim_lock = controller.sim_lock
        self.startup_cntr = 0
        self.orn = [0, 0, 0, 1]
        
        # Forward/keep contact
        self.forward_v = 0.1
        self.forward_kp = 2.6
        self.touched_once = False
        self.is_touching = False
        
        # Wiggle
        self.wiggle_cntr = 0
        self.wiggle_max = 24
        self.wiggle_dir = 1
        self.wiggle_samples = 5
        self.doing_wiggle = False
        self.wiggle_w = 0.35
        
        # Yaw align
        self.align_yaw = False
        self.yaw_cntr = 0
        self.yaw_max = 10
        self.yaw_algn_w = -0.5
        self.yaw_algn_kp = 9.6
        self.yaw_algn_thresh = 0.02
        self.yaw_aligned = False
        
        # Pitch align
        self.pitch_algn_w = 0.5
        self.pitch_algn_kp = 3.2
        self.pitch_algn_thresh = 0.06
        self.pitch_aligned = False
    
        # Force/torque
        self.ft_contact = np.zeros(6)
        self.ft_ema = np.zeros(6)
        self.alpha_ft_ema = 0.05
        self.ft_feeling_sum = np.zeros((1,6))
        self.ft_feeling = np.zeros(6)
        
        self.thresh = {
            'fx': 0.1,   # threshold for force in x direction
            'fz': -0.2,  # threshold for force in z direction
            'fy': 0.1,   # threshold for force in y direction
            'tx': 0.1,   # threshold for torque in x direction
            'ty': 0.1,   # threshold for torque in y direction
            'tz': 0.1    # threshold for torque in z direction
        }
# -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> None:
        self.controller.get_fingertip_pos()
        
        match self.state:
            case 'start':
                self.controller.reset_home_pos()
                # self.state = 'find_object'
                # self.state = 'test'
                # self.state = 'do_wiggle'
                self.state = 'do_initial_pos'
                
            case 'do_initial_pos':
                self.state_initial_pos()

            case 'open_gripper':
                self.state_open_gripper()
                # self.state = 'test'

            case 'test':
                self.state_test()
                # self.test_wiggle()
                
            case 'interact_perceive':
                self.state_interact_perceive()
                
            case _:
                pass
# -----------------------------------------------------------------------------------------------------------
    def state_test(self) -> None:
        self.controller.mode = 'velocity'
        self.test_pitch()
# -----------------------------------------------------------------------------------------------------------
    def state_initial_pos(self) -> None:
        self.controller.mode = 'velocity'
        pos = [self.initial_x, self.initial_y, self.initial_z]
        orn = [0, 90.0 * math.pi/180.0, 0]
        
        self.controller.do_move_pos(pos=pos, orn=orn)
        
        pos_err, orn_err = self.controller.get_pos_error(desired_pos=pos, desired_orn=orn)
        if any(abs(x) > 0.1 for x in pos_err) or any(abs(x) > 0.1 for x in orn_err):
            # print(f"Pos err: {pos_err}, orn err: {orn_err}")
            return
        else:
            self.state = 'open_gripper'
# -----------------------------------------------------------------------------------------------------------
    def state_open_gripper(self) -> None:
        self.controller.mode = 'position'
        if self.controller.check_gripper_pos() is True:
            self.state = 'interact_perceive'
            # self.state = 'test'
        else:
            self.controller.open_gripper()
# -----------------------------------------------------------------------------------------------------------
    def state_interact_perceive(self) -> None:
        # Calculate exponential moving average to smooth out noise from ft readings
        self.update_ft_ema()
        
        # Move in the z direction to maintain light contact with object
        speed_contact = self.do_keep_contact()
        
        # Wiggle wrist to get the feeling force
        speed_wiggle = [0, 0, 0, 0, 0, 0]
        if self.is_touching or self.doing_wiggle:
            speed_wiggle = self.do_wiggle()

            # Max samples for feeling force should be wiggle samples * wiggle_max
            if self.ft_feeling_sum.shape[0] >= (self.wiggle_samples * self.wiggle_max):
                self.ft_feeling_sum = self.ft_feeling_sum[1:]
                if self.wiggle_cntr == 0:
                    self.align_yaw = True
            self.ft_feeling_sum = np.append(self.ft_feeling_sum, self.ft_ema.reshape(1,6), axis=0)
        
        self.ft_feeling = self.ft_feeling_sum.mean(axis=0)
        self.controller.ft_feeling = self.ft_feeling.tolist()
        
        speed_yaw = [0, 0, 0, 0, 0, 0]
        if self.align_yaw:
            speed_yaw = self.do_align_yaw()
        
        speed_pitch = [0, 0, 0, 0, 0, 0]
        # speed_pitch = self.do_align_pitch()
        
        # Sum up all the speed components
        speed_total = [speed_contact[i] + speed_wiggle[i] + speed_yaw[i] + speed_pitch[i] for i in range(len(speed_contact))]
        
        # Apply speed
        self.controller.mode = 'velocity'
        self.controller.do_move_velocity(v_des=speed_total[:3], w_des=speed_total[3:], link='wrist', wf=False)
# -----------------------------------------------------------------------------------------------------------
    def update_ft_ema(self) -> None:
        self.ft = self.controller.ft_contact
        self.ft_ema = (1.0 - self.alpha_ft_ema) * self.ft_ema + (self.alpha_ft_ema * np.array(self.ft))
        self.controller.ft_ema = self.ft_ema.tolist()
# -----------------------------------------------------------------------------------------------------------
    def do_keep_contact(self) -> list:        
        if self.ft[fz] < self.thresh['fz']:
            self.is_touching = True
            if not self.touched_once:
                self.ft_ema = np.array(self.ft)
                self.touched_once = True
                self.ft_feeling_sum = self.ft_ema.reshape((1,6))
        
        # max_speed = 0.1
        # Kp = 4.5
        # speed = max_speed * Kp * (self.ft_ema[fz] - self.thresh['fz'])
        # Kp = 2.6
        # speed = max_speed * Kp * (self.ft[fz] - self.thresh['fz'])
        speed = self.forward_v * self.forward_kp * (self.ft[fz] - self.thresh['fz'])
        v_ee = [0, 0, speed]
        w_ee = [0, 0, 0]
        
        return v_ee + w_ee
# -----------------------------------------------------------------------------------------------------------
    def do_wiggle(self) -> list:
        self.doing_wiggle = True
                
        self.wiggle_cntr += 1
                
        if self.wiggle_cntr <= self.wiggle_max/4 or self.wiggle_cntr > 3 * self.wiggle_max/4:
            self.wiggle_dir = -1
        else:
            self.wiggle_dir = 1
            
        if self.wiggle_cntr > self.wiggle_max:
            self.wiggle_cntr = 0
            self.doing_wiggle = False

        v = [0, 0, 0]
        w = [self.wiggle_w * self.wiggle_dir, 0, 0]
        
        # print(f"DEBUG wiggle v: {v_ee}, w: {w_ee}, wiggle_cntr: {self.wiggle_cntr}, align_yaw: {self.align_yaw}")
        return v + w
# -----------------------------------------------------------------------------------------------------------
    def do_align_yaw(self):
        if abs(self.ft_feeling[fy]) > self.yaw_algn_thresh:
            yaw_speed = self.yaw_algn_kp * self.yaw_algn_w * self.ft_feeling[fy] * -1.0
            self.yaw_aligned = False
        else:
            self.yaw_aligned = True
            return [0, 0, 0, 0, 0, 0]
        
        omega = np.array([yaw_speed, 0, 0])

        if self.ft_feeling[tx] > 0:
            right_finger = True
        else:
            right_finger = False 
                           
        right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        r = right_tip if right_finger else left_tip
        r = np.array(r)        

        v = -np.cross(omega, r)
        
        self.yaw_cntr += 1
        if self.yaw_cntr >= self.yaw_max:
            self.yaw_cntr = 0
            self.align_yaw = False
        
        print(f"DEBUG yaw align v: {v}, w: {omega}, cntr: {self.yaw_cntr}, do align: {self.align_yaw}, right_finger: {right_finger}")
        return v.tolist() + omega.tolist()
# -----------------------------------------------------------------------------------------------------------
    def do_align_pitch(self) -> None:
        
        if abs(self.ft_feeling[fz]) > self.pitch_algn_thresh:
            pitch_speed = self.pitch_algn_kp * self.pitch_algn_w * self.ft_feeling[fz] * -1.0
            self.pitch_aligned = False
        else:
            self.pitch_aligned = True
            return [0, 0, 0, 0, 0, 0]

        omega = np.array([0, pitch_speed, 0])
        
        right_tip, left_tip = self.controller.get_fingertip_pos_wrist_frame()
        right = np.array(right_tip)
        left = np.array(left_tip)
        r = (right + left) / 2
        v = -np.cross(omega, r)
    
        print(f"DEBUG pitch align v: {v}, w: {omega}, r: {r}, aligned: {self.pitch_aligned}")
        return v.tolist() + omega.tolist()
# -----------------------------------------------------------------------------------------------------------
