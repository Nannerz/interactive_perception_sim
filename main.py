import pybullet as p
import time, os, sys, threading, subprocess, signal, atexit, math
import numpy as np
# import pandas as pd
from simulation import Simulation
from controller import Controller
from plotter import Plotter

# -----------------------------------------------------------------------------------------------------------

class App():
    def __init__(self):
        super().__init__()
        self.path = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------------------------------------

    ''' start_pos_gui: Starts position gui that shows the current world frame position of the end effector '''
    def start_pos_gui(self) -> subprocess.Popen:
        position_path = os.path.join(self.path, "position_gui.py")
        if not os.path.exists(position_path):
            print(f"Position script not found at {position_path}.")
            sys.exit(1)
            
        position_gui = subprocess.Popen(
            [sys.executable, position_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        return position_gui

# -----------------------------------------------------------------------------------------------------------

    ''' cleanup: Kills subprocesses if they arte still active and disconnects from PyBullet '''
    def cleanup(self, processes, signum=None, frame=None) -> None:
        for proc in processes:
            # only terminate if still running
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
        
        try:
            p.disconnect()
        except: # already disconnected, ignore
            pass
            
        # if this was a signal, exit now
        if signum is not None:
            sys.exit(0)

# -----------------------------------------------------------------------------------------------------------

    ''' register_cleanup: Registers the "cleanup" function for Ctrl-C, Ctrl-Break, and normal program exit '''
    def register_cleanup(self, processes) -> None:
        # 3) register for Ctrl‑C and Ctrl‑Break
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup(processes, signum=s, frame=f))   # Ctrl‑C
        signal.signal(signal.SIGBREAK, lambda s, f: self.cleanup(processes, signum=s, frame=f)) # Ctrl‑Break

        # 4) register for normal program exit
        atexit.register(self.cleanup, processes)
        
# -----------------------------------------------------------------------------------------------------------

    def run(self) -> None:
        sim = Simulation(data_path=self.path)
        
        subprocesses = []
        subprocesses.append(self.start_pos_gui())
        self.register_cleanup(subprocesses)
        
        plotter_thread = Plotter(data_path=self.path)
        plotter_thread.start()
        
        controller_thread = Controller(sim=sim, 
                                       data_path=self.path)
        controller_thread.start()
        controller_thread.run()
        
        controller_thread.join()
        plotter_thread.join()
        
# -----------------------------------------------------------------------------------------------------------
        
if __name__ == "__main__":
    app = App()
    app.run()