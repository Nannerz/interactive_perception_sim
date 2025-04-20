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
        self.simulation = Simulation()
        self.path = os.path.abspath(os.path.dirname(__file__))

# -----------------------------------------------------------------------------------------------------------
        
    # ''' start_plotter: Starts the plotter.py script that plots forces on each joint '''
    # def start_plotter(self) -> subprocess.Popen:
    #     plotter_path = os.path.join(os.path.dirname(__file__), "plotter.py")
    #     if not os.path.exists(plotter_path):
    #         print(f"Plotter script not found at {plotter_path}.")
    #         sys.exit(1)
            
    #     plotter = subprocess.Popen(
    #         [sys.executable, plotter_path],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.DEVNULL
    #     )
    #     return plotter

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
        # self.simulation.run()
        # Initialize the simulation
        ip = Simulation()
        ip.init_sim()
        ip.initialize_plot_file()
        
        subprocesses = []
        # processes.append(start_plotter())
        subprocesses.append(self.start_pos_gui())
        self.register_cleanup(subprocesses)
        
        plotter_thread = Plotter().start()
        # gui = Position_GUI_Thread().start()

        initial_x = 0.7
        initial_y = 0
        initial_z = 0.4
        up_position = [initial_x, initial_y, initial_z]
        down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        ip.reset_pose(up_position, down_orientation)
        time.sleep(1)  # let things settle

        print("Lowering into the cube…")

        # ———————————
        # 4) LOWER INTO THE CUBE
        # ———————————
        for z in np.arange(initial_z, 0.15, -0.05):  # descending heights
            target = [initial_x, initial_y, z]
            ik_vals = p.calculateInverseKinematics(ip.robot,
                                                ip.ee_link_index,
                                                target,
                                                down_orientation)
            # command the arm joints
            for i in range(7):
                p.setJointMotorControl2(ip.robot, i,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=ik_vals[i],
                                        force=5)
            # keep gripper half‑closed
            for j in ip.finger_joints:
                p.setJointMotorControl2(ip.robot, j,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=0.04,
                                        force=50)
            print("Position: ", ip.get_ee_position())
            ip.write_wf_position()
            ip.write_forces(ip.get_forces())
            time.sleep(2)  # give time for collision

        print("Contact made! Wrist readings follow:")
        
        # -----------------------------------------------------------------------------------------------------------
        # Stream force readings to csv file for plotting
        # -----------------------------------------------------------------------------------------------------------
        try:
            while True:
                ip.write_wf_position()
                ip.write_forces(ip.get_forces())
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        
        plotter_thread.join()
        

# -----------------------------------------------------------------------------------------------------------
        
if __name__ == "__main__":
    app = App()
    app.run()