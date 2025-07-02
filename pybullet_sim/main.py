import pybullet as p
import time, sys, threading, subprocess, signal, atexit, traceback
import numpy as np
from multiprocessing import Process
from simulation import Simulation
from controller import Controller
import force_plotter
import vel_plotter
import position_gui


# -----------------------------------------------------------------------------------------------------------
class App:
    def __init__(self):
        super().__init__()
        self.shutdown_event = threading.Event()
        self.sim_lock = None

    # -----------------------------------------------------------------------------------------------------------
    """ cleanup: Kills subprocesses if they arte still active and disconnects from PyBullet """

    def cleanup(self, processes=None, threads=None, signum=None, frame=None) -> None:
        print("Cleaning up processes")
        if processes is not None:
            for proc in processes:
                # only terminate if still running
                if isinstance(proc, subprocess.Popen):
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
                elif isinstance(proc, Process):  # multiprocessing.Process
                    if proc.is_alive():
                        proc.terminate()
                        proc.join()
                else:
                    print(f"Unknown process type: {type(proc)}")

        print("Cleaning up threads")
        self.shutdown_event.set()
        if threads is not None:
            for thread in threads:
                try:
                    thread.join()
                    print(f"Thread {thread} finished.")
                except:
                    pass  # thread already finished, ignore

        print("All threads finished.")

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

    def register_cleanup(self, processes=None, threads=None) -> None:
        # register for Ctrl‑C and Ctrl‑Break
        # doesnt work on Linux, gotta fix signals
        signal.signal(
            signal.SIGINT,
            lambda s, f: self.cleanup(processes, threads, signum=s, frame=f),
        )  # Ctrl‑C
        signal.signal(
            signal.SIGBREAK, 
            lambda s, f: self.cleanup(processes, signum=s, frame=f)
        )  # Ctrl‑Break

        # register for normal program exit
        atexit.register(self.cleanup, processes, threads)

    # -----------------------------------------------------------------------------------------------------------
    def check_processes(self, processes) -> bool:
        for p in processes:
            if hasattr(p, "poll"):  # subprocess.Popen
                if p.poll() is not None:
                    return True
            elif hasattr(p, "exitcode"):  # multiprocessing.Process
                if p.exitcode is not None:
                    return True
        return False

    # -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        threads = []
        processes = []

        sim = Simulation()
        sim.init_sim()
        self.sim_lock = sim.sim_lock

        controller_thread = Controller(
            sim=sim, shutdown_event=self.shutdown_event, draw_debug=True, do_timers=False
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
                with sim.sim_lock:
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
