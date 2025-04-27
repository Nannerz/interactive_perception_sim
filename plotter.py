#!/usr/bin/env python3
# realtime_plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading, os, sys, csv, re

class Plotter(threading.Thread):
    _instance = None
    
    def __new__(cls, *args, **kwargs) -> 'Plotter':
        if cls._instance is None:
            cls._instance = super(Plotter, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 data_path: str=None, 
                 shutdown_event: threading.Event=threading.Event(), 
                 interval=0.01, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.interval = interval
        self.shutdown_event = shutdown_event
        self.daemon = True
        
        self.data_path = data_path or os.path.dirname(os.path.abspath(__file__))
        self.csv_file = os.path.join(self.data_path, "ft_data.csv")
        self.joint_ids = self.get_joint_ids()
        self.channels = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.lines = {}
        self.max_samples = 5000

    def get_joint_ids(self) -> list:
        # Read only the header row
        with open(self.csv_file, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)

        # Use a set comprehension + regex to grab the number after the last “_”
        ids = sorted({
            int(re.search(r'_(\d+)$', col).group(1))
            for col in header
            if re.search(r'_(\d+)$', col)
        })
        
        return ids

    def run(self) -> None:
        fig, axes = plt.subplots(len(self.joint_ids), 1, sharex=True, figsize=(8, 12))
        lines = {}  # will map (joint, channel) -> Line2D object
        
        # initialize each subplot
        for ax, j in zip(axes, self.joint_ids):
            ax.set_title(f"Joint {j}")
            for ch in self.channels:
                # empty line for each channel
                line, = ax.plot([], [], label=ch)
                lines[(j, ch)] = line
            ax.legend(loc="upper right")
            ax.set_ylabel("value")

        axes[-1].set_xlabel("Sample index")

        def animate(frame):
            if self.shutdown_event.is_set():
                print("Shutdown event is set, exiting plotter thread.")
                plt.close(fig)
                sys.exit(0)
            
            # reload all data
            df = pd.read_csv(self.csv_file).tail(self.max_samples)
            x = df.index  # use row index as x-axis; switch to a timestamp column if you add one

            # update each joint’s lines
            for j in self.joint_ids:
                for ch in self.channels:
                    key = f"{ch}_{j}"
                    y = df[key]
                    line = lines[(j, ch)]
                    line.set_data(x, y)

                ax = axes[j - 1]
                ax.relim()            # recalc data limits
                ax.autoscale_view()   # rescale axes

            return list(lines.values())

        # animate every 500 ms
        print("Plotter thread main loop...")
        
        ani = animation.FuncAnimation(
            fig,
            animate,
            interval=500,
            blit=False
        )

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    Plotter().run()