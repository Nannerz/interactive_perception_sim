#!/usr/bin/env python3
# realtime_plotter.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    CSV_FILE = "ft_data.csv"

    # define your joints and channels
    JOINT_IDS = range(1, 8)
    CHANNELS = ["fx", "fy", "fz", "tx", "ty", "tz"]

    # make figure with 6 rows × 1 col
    fig, axes = plt.subplots(len(JOINT_IDS), 1, sharex=True, figsize=(8, 12))
    lines = {}  # will map (joint, channel) -> Line2D object

    # initialize each subplot
    for ax, j in zip(axes, JOINT_IDS):
        ax.set_title(f"Joint {j}")
        for ch in CHANNELS:
            # empty line for each channel
            line, = ax.plot([], [], label=ch)
            lines[(j, ch)] = line
        ax.legend(loc="upper right")
        ax.set_ylabel("value")

    axes[-1].set_xlabel("Sample index")

    def animate(frame):
        # reload all data
        df = pd.read_csv(CSV_FILE).tail(1000)
        x = df.index  # use row index as x-axis; switch to a timestamp column if you add one

        # update each joint’s lines
        for j in JOINT_IDS:
            for ch in CHANNELS:
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
    
main()
