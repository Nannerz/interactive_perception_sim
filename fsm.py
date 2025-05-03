class FSM():
    _instance = None
# -----------------------------------------------------------------------------------------------------------
    def __new__(cls, *args, **kwargs) -> 'FSM':
        if cls._instance is None:
            cls._instance = super(FSM, cls).__new__(cls)
        return cls._instance
# -----------------------------------------------------------------------------------------------------------
    def __init__(self, controller, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = 'start'
        self.startup_z = 0.7
        self.controller = controller
        self.wiggle_cntr = 0
        self.wiggle_max = 100
        self.wiggle_dir = 1
        
# -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> None:
        if self.state == 'start':
            self.controller.go_pos = True
            self.controller.initial_pos()
            self.state = 'startup'
            # self.state = 'nothing'

        elif self.state == 'startup':
            self.controller.go_pos = True
            self.startup_z -= 0.05
            self.controller.do_startup(self.startup_z)
            if self.startup_z <= 0.3:
                self.state = 'lowered'
                
        elif self.state == 'lowered':
            self.controller.go_pos = False
            self.wiggle_cntr += 1
            if self.wiggle_cntr >= self.wiggle_max:
                self.wiggle_cntr = 0
                self.wiggle_dir *= -1
            self.controller.do_wiggle(self.wiggle_dir)
        else:
            self.controller.go_pos = False
# -----------------------------------------------------------------------------------------------------------
        