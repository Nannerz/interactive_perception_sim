import pybullet as p
import threading, signal, atexit

class Controller(threading.Thread):
    def __init__(self, robot, **kwargs):
        super().__init__(**kwargs)
        self.daemon = True
        self.interval = 0.01
        self.robot = robot
        p.connet()

    def exit_handler(self) -> None:
        # 3) register for Ctrl‑C and Ctrl‑Break
        signal.signal(signal.SIGINT, self.cleanup)   # Ctrl‑C
        signal.signal(signal.SIGBREAK, self.cleanup) # Ctrl‑Break

        # 4) register for normal program exit
        atexit.register(self.cleanup)
        
    def cleanup(self) -> None:
        try:
            p.disconnect()
        except:
            pass
        
        self.close()
        
    def run(self) -> None:
        pass
        # for i in range(7):
        # p.setJointMotorControl2(ip.robot, i,
        #                         controlMode=p.POSITION_CONTROL,
        #                         targetPosition=ik_vals[i],
        #                         force=5)