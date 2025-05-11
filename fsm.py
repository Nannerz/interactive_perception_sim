import math

class FSM():
    def __init__(self, 
                 controller, 
                 initial_x, initial_y, initial_z) -> None:
        
        self.state = 'start'
        self.controller = controller
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_z = initial_z
        self.wiggle_cntr = 0
        self.wiggle_max = 500
        self.wiggle_dir = 1
        self.doing_wiggle = False
        self.sim_lock = controller.sim_lock
        self.startup_cntr = 0
        self.orn = [0, 0, 0, 1]
        
# -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> None:
        match self.state:
            case 'start':
                self.controller.reset_home_pos()
                # self.state = 'find_object'
                # self.state = 'test'
                # self.state = 'do_wiggle'
                self.state = 'do_initial_pos'
                
            case 'do_initial_pos':
                self.controller.mode = 'velocity'
                pos = [self.initial_x, self.initial_y, self.initial_z]
                orn = [0, math.pi/2, 0]
                
                self.controller.do_move_pos(pos=pos, orn=orn)
                
                pos_err, orn_err = self.controller.get_pos_error(desired_pos=pos, desired_orn=orn)
                if any(abs(x) > 0.03 for x in pos_err) or any(abs(x) > 0.03 for x in orn_err):
                    print(f"Pos err: {pos_err}, orn err: {orn_err}")
                    return
                else:
                    self.state = 'test'

            case 'test':
                self.controller.mode = 'velocity'
                # if self.wiggle_cntr <= 0:
                    # self.orn = p.getLinkState(self.controller.robot, self.controller.wrist_idx)[5]
                if self.wiggle_cntr > self.wiggle_max:
                    self.wiggle_cntr = 0
                    self.wiggle_dir *= -1
                
                self.wiggle_cntr += 1
                    
                # v_ee = [0, 0, 0.02 * self.wiggle_dir]
                v_ee = [0, 0, 0.2]
                w_ee = [0, 0, 0]
                self.controller.do_move_velocity(v_ee=v_ee, w_ee=w_ee, link='wrist')  
                                    
            case 'find_object':
                self.controller.mode = 'velocity'
                ft = self.controller.ft_contact
                thresh_fz = -0.3
                Kp = 0.02
                # w = 0.01
                w = 0.01
                speed = Kp * (ft[2] - thresh_fz)
                v_ee = [0, 0, speed]
                w_ee = [0, 0, 0]
                
                if ft[2] < thresh_fz:
                    self.doing_wiggle = True
                    
                if self.doing_wiggle:
                    # print("Wiggling!")
                    self.wiggle_cntr += 1
                    
                    if self.wiggle_cntr < self.wiggle_max/4 or self.wiggle_cntr >= 3 * self.wiggle_max/4:
                        self.wiggle_dir = -1
                    else:
                        self.wiggle_dir = 1
                        
                    if self.wiggle_cntr >= self.wiggle_max:
                        self.doing_wiggle = False
                        self.wiggle_cntr = 0
                                            
                    w_ee = [w * self.wiggle_dir, 0, 0]
                
                self.controller.do_move_velocity(v_ee=v_ee, w_ee=w_ee)
                
            case 'do_wiggle':
                self.controller.mode = 'velocity'
                self.wiggle_cntr += 1
                
                if self.wiggle_cntr < self.wiggle_max/4 or self.wiggle_cntr >= 3 * self.wiggle_max/4:
                    self.wiggle_dir = -1
                else:
                    self.wiggle_dir = 1
                    
                if self.wiggle_cntr >= self.wiggle_max:
                    self.wiggle_cntr = 0

                self.controller.do_move_velocity(v_ee=[0, 0, 0], w_ee=[0.1 * self.wiggle_dir, 0, 0])
                
                
            case _:
                return
# -----------------------------------------------------------------------------------------------------------
    # def startup(self) -> None:
    #     self.state = 'startup'
    #     self.controller.initial_pos()
    #     self.startup_z = 0.7
        