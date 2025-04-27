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
        self.initialzied = False
        self.therads = []
        self.processes = []
        self.shutdown_event = threading.Event()

# -----------------------------------------------------------------------------------------------------------

    ''' start_pos_gui: Starts position gui that shows the current world frame position of the end effector '''
    def start_pos_gui(self) -> subprocess.Popen:
        position_path = os.path.join(self.path, "position_gui.py")
        if not os.path.exists(position_path):
            print(f"Position script not found at {position_path}.")
            sys.exit(1)
            
        position_gui = subprocess.Popen(
            [sys.executable, position_path],
        )
        
        return position_gui

# -----------------------------------------------------------------------------------------------------------

    ''' cleanup: Kills subprocesses if they arte still active and disconnects from PyBullet '''
    def cleanup(self, processes=None, threads=None, signum=None, frame=None) -> None:
        print("Cleaning up processes")
        for proc in processes:
            # only terminate if still running
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            
        print("Cleaning up threads")
        self.shutdown_event.set()
        for thread in threads:
            try:
                thread.join()
                print(f"Thread {thread} finished.")
            except:
                pass # thread already finished, ignore
        
        print("All threads finished.")
        
        try:
            p.disconnect()
        except: # already disconnected, ignore
            pass
        
        if signum is not None:
            sys.exit(0)

# -----------------------------------------------------------------------------------------------------------

    ''' register_cleanup: Registers the "cleanup" function for Ctrl-C, Ctrl-Break, and normal program exit '''
    def register_cleanup(self, processes=None, threads=None) -> None:
        # 3) register for Ctrl‑C and Ctrl‑Break
        signal.signal(signal.SIGINT, lambda s, f: self.cleanup(processes, threads, signum=s, frame=f))   # Ctrl‑C
        signal.signal(signal.SIGBREAK, lambda s, f: self.cleanup(processes, signum=s, frame=f)) # Ctrl‑Break

        # 4) register for normal program exit
        atexit.register(self.cleanup, processes, threads)
        
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        if not self.initialzied:
            # sim 
            sim = Simulation()
            sim.init_sim()
            
            # position gui
            self.processes.append(self.start_pos_gui())
            
            # matplotlib plotter thread
            plotter_thread = Plotter(data_path=self.path,
                                     shutdown_event=self.shutdown_event)
            plotter_thread.start()
            
            # controller thread
            controller_thread = Controller(sim=sim, 
                                        data_path=self.path,
                                        shutdown_event=self.shutdown_event)
            controller_thread.start()
            
            
            self.threads = [plotter_thread, controller_thread]
            self.register_cleanup(self.processes, self.threads)
            
            self.initialzied = True
            print("All threads started")
            
        try:
            print("Main thread main loop...")
            while not self.shutdown_event.is_set():
                with sim.sim_lock:
                    if not p.isConnected():
                        print("PyBullet disconnected, exiting...")
                        break
                    
                dead_threads = [thread for thread in self.threads if not thread.is_alive()]
                if dead_threads:
                    print(f"Threads {dead_threads} have died, exiting...")
                    break
            
                time.sleep(0.01)
        finally:
            print("Main thread exiting...")
# -----------------------------------------------------------------------------------------------------------
        
if __name__ == "__main__":
    app = App()
    app.run()