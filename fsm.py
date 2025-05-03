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
        
# -----------------------------------------------------------------------------------------------------------
    def next_state(self) -> None:
        if self.state == 'start':
            self.controller.go_pos = True
            self.controller.initial_pos()
            # self.state = 'startup'
            self.state = 'nothing'

        elif self.state == 'startup':
            self.controller.go_pos = True
            self.startup_z -= 0.05
            self.controller.do_startup(self.startup_z)
            if self.startup_z <= 0.3:
                self.state = 'lowered'
                print("Contact made! Wrist readings follow:")
        else:
            self.controller.go_pos = False
# -----------------------------------------------------------------------------------------------------------
        