import pybullet as p
import time, sys, threading, signal, atexit, math
from typing import Any
from multiprocessing import Process
from simulation import Simulation
from controller import Controller
import force_plotter
import position_gui

p: Any = p

# -----------------------------------------------------------------------------------------------------------
class App:
    def __init__(self):
        super().__init__()
        self.shutdown_event = threading.Event()
        self.sim_lock = None

    # -----------------------------------------------------------------------------------------------------------
    """ cleanup: Kills subprocesses/threads if they are still active and disconnects from PyBullet """

    def cleanup(self, 
                processes: list[Process], 
                threads: list[threading.Thread], 
                signum: int | None
    ) -> None:
        
        print("Cleaning up processes")
        for proc in processes:
            # only terminate if still running
            if proc.is_alive():
                proc.terminate()
                proc.join()

        print("Cleaning up threads")
        self.shutdown_event.set()
        for thread in threads:
            try:
                thread.join()
                print(f"Thread {thread} finished.")
            except:
                pass  # thread already finished, ignore

        print("All processes & threads finished.")

        try:
            if self.sim_lock is not None:
                with self.sim_lock:
                    p.disconnect()
            else:
                p.disconnect()
        except:  # already disconnected, ignore
            pass

        if signum is not None:
            sys.exit(0)

    # -----------------------------------------------------------------------------------------------------------
    """ register_cleanup: Registers the "cleanup" function for Ctrl-C, Ctrl-Break, and normal program exit """

    def register_cleanup(self, processes: list[Process], threads: list[threading.Thread] ) -> None:
        # register for Ctrl‑C and Ctrl‑Break
        # doesnt work on Linux, gotta fix signals
        signal.signal(
            signal.SIGINT,
            lambda s, f: self.cleanup(processes, threads, signum=s),
        )  # Ctrl‑C
        signal.signal(
            signal.SIGBREAK, 
            lambda s, f: self.cleanup(processes, threads, signum=s)
        )  # Ctrl‑Break

        # register for normal program exit
        atexit.register(self.cleanup, processes, threads, signum=None)

    # -----------------------------------------------------------------------------------------------------------
    def check_processes(self, processes: list[Process] ) -> bool:
        for proc in processes:
            if proc.exitcode is not None:
                return True
        return False

    # -----------------------------------------------------------------------------------------------------------
    def get_scenarios(self) -> dict[str, Any]:
        # Scenarios for testing
        initial_robot_pos: dict[str, list[float]] = {
            "straight":     [0.7, 0.0, 0.1],    # use with straight
            "angled_up":    [0.7, 0.0, 0.15],   # angle up
            "angled_down":  [0.68, -0.025, 0.16],   # use with angles down
        }
        initial_robot_orn: dict[str, list[float]] = {
            "straight":     [0, 90.0 * math.pi / 180.0, 0],                                             # straight forward in Z direction
            "pitch_down":   [0, 105.0 * math.pi / 180.0, 0],                                            # pitched down
            "pitch_up":     [0, 75.0 * math.pi / 180.0, 0],                                             # pitched up
            "angle3":       [15.0 * math.pi / 180.0, 75.0 * math.pi / 180.0, 15.0 * math.pi / 180.0],   # pitched up & yaw
        }
        robot_confs: dict[str, Any] = {
            "straight":     { "pos": initial_robot_pos["straight"],     "orn": initial_robot_orn["straight"] },
            "angled_down":  { "pos": initial_robot_pos["angled_down"],  "orn": initial_robot_orn["pitch_down"] },
            "angled_up":    { "pos": initial_robot_pos["angled_up"],    "orn": initial_robot_orn["pitch_up"] },
            "pitch_n_yaw":  { "pos": initial_robot_pos["angled_up"],    "orn": initial_robot_orn["angle3"] },
        }

        obj_pos: dict[str, list[float]] = {
            "mustard":              [0.8, 0.04, 0.09],
            
            "pringles_straight":    [0.8, 0.04, 0.03],
            "pringles_left":        [0.8, 0.06, 0.03],
            "pringles_right":       [0.8, 0.02, 0.03],
            
            "cracker1":             [0.8, 0.025, 0.11],
            "cracker2":             [0.8, 0.055, 0.11],
            "cracker3":             [0.8, 0.04, 0.11],
        }
        obj_orn: dict[str, list[float]] = {
            "mustard":  [0, 0, 20 * math.pi / 180],
            
            "pringles": [0, 0, 0],
            
            "cracker1": [0, 0, 75 * math.pi / 180],
            "cracker2": [0, 0, 90 * math.pi / 180],
            "cracker3": [0, 0, 105 * math.pi / 180],
        }
        
        # Valid names are: "cracker_box", "mustard_bottle", "pringles_can"
        obj_confs: dict[str, Any] = {
            "mustard":              { "name": "mustard_bottle", "pos": obj_pos["mustard"],              "orn": obj_orn["mustard"] },
            
            "pringles_straight":    { "name": "pringles_can",   "pos": obj_pos["pringles_straight"],    "orn": obj_orn["pringles"] },
            "pringles_left":        { "name": "pringles_can",   "pos": obj_pos["pringles_left"],        "orn": obj_orn["pringles"] },
            "pringles_right":       { "name": "pringles_can",   "pos": obj_pos["pringles_right"],       "orn": obj_orn["pringles"] },
            
            "cracker1":             { "name": "cracker_box",    "pos": obj_pos["cracker1"],             "orn": obj_orn["cracker1"] },
            "cracker2":             { "name": "cracker_box",    "pos": obj_pos["cracker2"],             "orn": obj_orn["cracker2"] },
            "cracker3":             { "name": "cracker_box",    "pos": obj_pos["cracker3"],             "orn": obj_orn["cracker3"] },
        }

        scenarios: dict[str, Any] = {
            "mustard":      { "robot": robot_confs["straight"],  "sim_obj": obj_confs["mustard"] },             # mustard bottle angled right
            
            "pringles1":    { "robot": robot_confs["angled_up"], "sim_obj": obj_confs["pringles_straight"] },   # hand angle up, pringles can straight ahead
            "pringles2":    { "robot": robot_confs["angled_up"], "sim_obj": obj_confs["pringles_left"] },       # hand angle up, pringles can to the left
            "pringles3":    { "robot": robot_confs["angled_up"], "sim_obj": obj_confs["pringles_right"] },      # hand angle up, pringles can to the right
            "pringles4":    { "robot": robot_confs["angled_down"], "sim_obj": obj_confs["pringles_straight"] }, # hand angle down, pringles can straight ahead
            "pringles5":    { "robot": robot_confs["angled_down"], "sim_obj": obj_confs["pringles_left"] },     # hand angle down, pringles can to the left
            "pringles6":    { "robot": robot_confs["angled_down"], "sim_obj": obj_confs["pringles_right"] },    # hand angle down, pringles can to the right
            
            "cracker1":     { "robot": robot_confs["angled_up"], "sim_obj": obj_confs["cracker1"] },
            "cracker2":     { "robot": robot_confs["angled_up"], "sim_obj": obj_confs["cracker2"] },
            "cracker3":     { "robot": robot_confs["angled_up"], "sim_obj": obj_confs["cracker3"] },
        }

        return scenarios

    # -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        threads: list[threading.Thread] = []
        processes: list[Process] = []

        ##########################################################################################
        # Choose starting scenario here

        scenarios = self.get_scenarios()
        myscenario = "pringles1"
        sim_obj_dic = scenarios[myscenario]["sim_obj"]
        robot_conf_dict = scenarios[myscenario]["robot"]
        
        ##########################################################################################

        sim = Simulation()
        sim.init_sim(sim_obj=sim_obj_dic)
        self.sim_lock = sim.sim_lock

        controller_thread = Controller(
            sim=sim, 
            shutdown_event=self.shutdown_event, 
            draw_debug=True, 
            do_timers=True, 
            initial_robot_conf=robot_conf_dict
        )
        controller_thread.start()
        threads.append(controller_thread)

        # target_processes = [position_gui, force_plotter, vel_plotter]
        target_processes = [position_gui, force_plotter]
        for i, proc in enumerate(target_processes):
            processes.append(Process(target=proc.main))
            processes[i].start()

        self.register_cleanup(processes, threads)

        print("All threads & processes started")

        try:
            while not self.shutdown_event.is_set():
                with self.sim_lock:
                    if not p.isConnected():
                        print("PyBullet disconnected, exiting...")
                        break

                dead_threads = [thread for thread in threads if not thread.is_alive()]
                if dead_threads:
                    print(f"Threads {dead_threads} have died, exiting...")
                    break

                if self.check_processes(processes):
                    break

                # sleep since main thread only kicks things off and cleans up at the end
                time.sleep(0.5)
        finally:
            print("Main thread exiting...")


# -----------------------------------------------------------------------------------------------------------
def main() -> None:
    app = App()
    app.run()


# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
