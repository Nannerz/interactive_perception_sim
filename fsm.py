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
        self.wiggle_max = 200
        self.wiggle_dir = 1
        
# -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> None:
        match self.state:
            case 'start':
                # self.controller.initial_reset()
                self.controller.mode = 'position'
                self.state = 'startup'
                print("Lowering into the cube...")

            case 'startup':
                self.controller.mode = 'position'
                self.controller.do_startup(self.initial_z)
                pos_error, _ = self.controller.get_joint_errors()
                # sum_error = sum(abs(x) for x in pos_error)
                # if any(abs(x) > 0.35 for x in pos_error):
                if (sum_error := sum(abs(x) for x in pos_error)) > 0.38:
                    print(f"Error sum: {sum_error}")
                    print(f"Position error: {pos_error}")
                    return
                
                self.initial_z -= 0.05
                
                if self.initial_z <= 0.3:
                    print("Lowered!")
                    self.state = 'lowered'
                    # self.state = 'nothing'
                    
            case 'lowered':
                self.controller.mode = 'velocity'
                self.wiggle_cntr += 1
                if self.wiggle_cntr >= self.wiggle_max:
                    self.wiggle_cntr = 0
                    self.wiggle_dir *= -1
                self.controller.do_wiggle(self.wiggle_dir)
                
            case _:
                return
# -----------------------------------------------------------------------------------------------------------
    def startup(self) -> None:
        self.state = 'startup'
        self.controller.initial_pos()
        self.startup_z = 0.7
        