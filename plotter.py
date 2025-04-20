#!/usr/bin/env python3
# realtime_plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

class Plotter(threading.Thread):
    def __init__(self, interval=0.01, **kwargs):
        super().__init__(**kwargs)
        self.interval = interval
        self.daemon = True
        # self._shared_ft = data
        self.csv_file = "ft_data.csv"
        self.joint_ids = range(1, 8)
        self.channels = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.lines = {}
        
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
            # reload all data
            df = pd.read_csv(self.csv_file).tail(1000)
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