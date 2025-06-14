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
        self.wiggle_max = 40
        self.wiggle_dir = 1
        self.wiggle_samples = 5
        self.doing_wiggle = False
        self.wiggle_w = 0.5
        
        # Yaw align
        self.align_yaw = False
        self.yaw_cntr = 0
        self.yaw_max = 10
        self.yaw_algn_w = -0.5
        self.yaw_algn_kp = 9.6
    
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
        w_des = [1.0, 0, 0]
        self.controller.do_spin_around(w_des=w_des)
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
        
        self.ft_feeling = self.ft_feeling_sum.mean(axis=0)
        self.controller.ft_feeling = self.ft_feeling.tolist()
        
        # Align the yaw if we've gotten enough wiggling samples
        speed_yaw = [0, 0, 0, 0, 0, 0]
        if self.align_yaw:
            speed_yaw = self.do_align_yaw()
        
        # Sum up all the speed components
        speed_total = [speed_contact[i] + speed_wiggle[i] + speed_yaw[i] for i in range(len(speed_contact))]
        
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
        
        # Max samples for feeling force should be wiggle samples * wiggle_max
        if self.ft_feeling_sum.shape[0] >= (self.wiggle_samples * self.wiggle_max):
            self.ft_feeling_sum = self.ft_feeling_sum[1:]
            if self.wiggle_cntr == 0:
                self.align_yaw = True
                
        self.wiggle_cntr += 1
        
        self.ft_feeling_sum = np.append(self.ft_feeling_sum, self.ft_ema.reshape(1,6), axis=0)
        
        if self.wiggle_cntr <= self.wiggle_max/4 or self.wiggle_cntr > 3 * self.wiggle_max/4:
            self.wiggle_dir = -1
        else:
            self.wiggle_dir = 1
            
        if self.wiggle_cntr > self.wiggle_max:
            self.wiggle_cntr = 0
            self.doing_wiggle = False

        v_ee = [0, 0, 0]
        w_ee = [self.wiggle_w * self.wiggle_dir, 0, 0]
        # w_ee = [0, 0, 0]
        
        # self.controller.do_move_velocity(v_ee=v_ee, w_ee=w_ee)
        print(f"DEBUG wiggle v: {v_ee}, w: {w_ee}, wiggle_cntr: {self.wiggle_cntr}, align_yaw: {self.align_yaw}")
        return v_ee + w_ee
# -----------------------------------------------------------------------------------------------------------
    def do_align_yaw(self):
        yaw_speed = self.yaw_algn_kp * self.yaw_algn_w * self.ft_feeling[fy]
            
        if self.ft_feeling[tx] > 0:
            right_finger = True
        else:
            right_finger = False

        w_des = np.array([yaw_speed, 0, 0])
                
        right_tip_pos_wrist_frame, left_tip_pos_wrist_frame = self.controller.get_fingertip_pos_wrist_frame()
        r = right_tip_pos_wrist_frame if right_finger else left_tip_pos_wrist_frame
        r = np.array(r)
                    
        v_des = -np.cross(w_des, r)
        
        self.yaw_cntr += 1
        if self.yaw_cntr >= self.yaw_max:
            self.yaw_cntr = 0
            self.align_yaw = False
        
        print(f"DEBUG yaw align v: {v_des}, w: {w_des}, yaw_speed: {yaw_speed}, yaw_cntr: {self.yaw_cntr}, align_yaw: {self.align_yaw}")
        return v_des.tolist() + w_des.tolist()
# -----------------------------------------------------------------------------------------------------------
    def test_wiggle(self) -> None:
        twist_wiggle = self.do_wiggle()
        self.controller.do_move_velocity(v_des=twist_wiggle[:3], w_des=twist_wiggle[3:], link='wrist', wf=False)
# -----------------------------------------------------------------------------------------------------------