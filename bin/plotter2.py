import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, csv

from typing import Any
plt: Any = plt

# -----------------------------------------------------------------------------------------------------------
def do_plot(csv_file: str) -> None:
    header = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
    fig, ax = plt.subplots()
    # dataname = "contact_ft"
    dataname = "feeling_ft"
    df = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))

    interval = 0.005
    x = np.arange(len(df), dtype=float) * interval

    for col in df.columns:
        if col.startswith("f"):
            ax.plot(x, df[col], label=col[:2])

    ax.relim()
    ax.autoscale_view()
    ax.legend(loc="upper left")
    ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Torque (N*m)")
    ax.set_ylabel("Force (N)")
    plt.show()

def do_plot2(csv_file: str) -> None:
    header = []
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        
    fig, ax = plt.subplots()

    interval = 0.005
    
    # plottype = "forces"
    # plottype = "torques"
    plottype = "feeling"
    # plottype = "vwrist"
    # plottype = "wwrist"

    match plottype:
        case "forces":
            dataname = "contact_ft"
            df = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))
            x = np.arange(len(df), dtype=float) * interval
            ftype = "f"
            for col in df.columns:
                if col.startswith(ftype):
                    ax.plot(x, df[col], label=col[:2])
            ax.set_ylabel("Force (N)")
        case "torques":
            dataname = "contact_ft"
            df = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))
            x = np.arange(len(df), dtype=float) * interval
            ftype = "t"
            for col in df.columns:
                if col.startswith(ftype):
                    ax.plot(x, df[col], label=col[:2])
            ax.set_ylabel("Torque (N*m)")
        case "feeling":
            dataname = "contact_ft"
            df1 = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))
            dataname = "feeling_ft"
            df2 = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))
            x = np.arange(len(df1), dtype=float) * interval
            
            ftype = "fy"
            for col in df1.columns:
                if col.startswith(ftype):
                    ax.plot(x, df1[col], label=col[:2])

            for col in df2.columns:
                if col.startswith(ftype):
                    ax.plot(x, df2[col], label=f"{col[:2]} feeling")
            ax.set_ylabel("Force (N)")
            ax.set_ylim([-0.03, 0.04])
        case "vwrist":
            dataname = "v_wrist"
            df = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))
            x = np.arange(len(df), dtype=float) * interval
            ftype = "v"
            for col in df.columns:
                if col.startswith(ftype):
                    ax.plot(x, df[col], label=col[:2])
            ax.set_ylabel("Linear velocity (m/s)")
            ax.set_ylim([-0.1, 0.1])
        case "wwrist":
            dataname = "v_wrist"
            df = pd.read_csv(csv_file, usecols=lambda x: dataname in str(x))
            x = np.arange(len(df), dtype=float) * interval
            ftype = "w"
            for col in df.columns:
                if col.startswith(ftype):
                    ax.plot(x, df[col], label=col[:2])
            ax.set_ylabel("Angular velocity (rad/s)")
            ax.set_ylim([-0.1, 0.2])
        case _:
            pass

    ax.relim()
    ax.autoscale_view()
    ax.legend(loc="upper left")
    ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Torque (N*m)")
    # ax.set_ylabel("Force (N)")
    # ax.set_ylim([-0.5, 0.1])
    plt.show()

# -----------------------------------------------------------------------------------------------------------
def main() -> None:

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    csv_file = os.path.join(data_path, "ft_data.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    do_plot2(csv_file)
# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()