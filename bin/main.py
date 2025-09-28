import pybullet as p
import time, sys, threading, signal, atexit, yaml, os, math
from typing import Any
from multiprocessing import Process
from simulation import Simulation
from controller import Controller
import force_plotter
import position_gui

p: Any = p

# -----------------------------------------------------------------------------------------------------------
class ScenarioLoader(yaml.SafeLoader):
    # Custom constructors for loading yaml scenarios
    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self.add_constructor("!deg", self.deg_constructor)
        self.add_constructor("!quat", self.deg_constructor)

    def deg_constructor(self, loader: yaml.Loader, node: yaml.Node) -> float | list[float]:
        if isinstance(node, yaml.ScalarNode):
            val = float(loader.construct_scalar(node))
            return math.radians(val)
        elif isinstance(node, yaml.SequenceNode):
            seq = loader.construct_sequence(node)
            return [math.radians(float(x)) for x in seq]
        raise yaml.constructor.ConstructorError("!deg only supports scalar or sequence")

    def quat_constructor(self, loader: yaml.Loader, node: yaml.Node) -> list[float]:
        if isinstance(node, yaml.SequenceNode):
            seq = loader.construct_sequence(node)
            if len(seq) != 4:
                raise yaml.constructor.ConstructorError("!quat requires a sequence of 4 elements")
            return [float(x) for x in seq]
        raise yaml.constructor.ConstructorError("!quat only supports sequence")

# -----------------------------------------------------------------------------------------------------------
class App:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.sim_lock = None
        self.scenarios = self.get_scenarios()
        self.current_scenario = self.check_args()

    # -----------------------------------------------------------------------------------------------------------
    def check_args(self) -> str:
        got_err = False
        if len(sys.argv) != 2:
            got_err = True
            print("Usage: python main.py <scenario_name>")
        elif sys.argv[1] not in self.scenarios.keys():
            got_err = True
            print(f"Error: Scenario '{sys.argv[1]}' not found.")
            
        if got_err:
            print("Available scenarios:")
            for scenario in self.scenarios.keys():
                print(f" - {scenario}")
            sys.exit(1)
            
        return sys.argv[1]

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

        try:
            p.disconnect()
        except:  # already disconnected, ignore
            pass

        print("Waiting for other processes & threads to finish...")
        wait_cntr = 0
        while(True):
            keep_waiting = False
            for proc in processes:
                if proc.exitcode is None:
                    keep_waiting = True

            for thread in threads:
                if thread.is_alive():
                    keep_waiting = True

            if not keep_waiting:
                print("All processes & threads finished.")
                break
            else:
                wait_cntr += 1
                if wait_cntr > 10:
                    print("Processes & threads not responding, exiting main...")
                    break
                time.sleep(0.5)

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
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
        scenario_file = os.path.join(data_path, "scenarios.yaml")

        with open(scenario_file, "r") as f:
            scenarios = yaml.load(f, Loader=ScenarioLoader)

        return scenarios['scenarios']

    # -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        threads: list[threading.Thread] = []
        processes: list[Process] = []

        ##########################################################################################
        # Choose starting scenario here

        # myscenario = "pringles1"
        myscenario = self.current_scenario
        sim_obj_dic = self.scenarios[myscenario]["sim_obj"]
        robot_conf_dict = self.scenarios[myscenario]["robot"]
        
        ##########################################################################################

        sim = Simulation()
        sim.init_sim(sim_obj=sim_obj_dic)
        self.sim_lock = sim.sim_lock

        controller_thread = Controller(
            sim=sim, 
            shutdown_event=self.shutdown_event, 
            draw_debug=False, 
            do_timers=True, 
            initial_robot_conf=robot_conf_dict
        )
        controller_thread.start()
        threads.append(controller_thread)

        target_processes = [position_gui, force_plotter]
        for i, proc in enumerate(target_processes):
            processes.append(Process(target=proc.main))
            processes[i].start()

        self.register_cleanup(processes, threads)

        print("All threads & processes started")

        try:
            while not self.shutdown_event.is_set():
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
