import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, csv

from typing import Any
plt: Any = plt

# -----------------------------------------------------------------------------------------------------------
def main() -> None:

    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    # csv_file = os.path.join(data_path, "ft_data.csv")
    csv_file = os.path.join(data_path, "ft_data_rollfail.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file, header=0)
    interval = 0.005  # 5ms

    axes_x = 2
    axes_y = 4
    fig, axes = plt.subplots(axes_y, axes_x, figsize=(10, 12))
    fig.tight_layout(pad=4.0)
    x = np.arange(len(df), dtype=float) * interval

    raw_forces = df[["fx_contact_ft", "fy_contact_ft", "fz_contact_ft"]]
    ax = axes[0,0]
    for name, data in raw_forces.items():
        ax.plot(x, data, label=name[:2])
    ax.set_title("Raw Forces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend()

    raw_torques = df[["tx_contact_ft", "ty_contact_ft", "tz_contact_ft"]]
    ax = axes[0,1]
    for name, data in raw_torques.items():
        ax.plot(x, data, label=name[:2])
    ax.set_title("Raw Torques")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend()

    ema_forces = df[["fx_ema_ft", "fy_ema_ft", "fz_ema_ft"]]
    ax = axes[1,0]
    for name, data in ema_forces.items():
        ax.plot(x, data, label=name[:2])
    ax.set_title("EMA Forces")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend()

    ema_torques = df[["tx_ema_ft", "ty_ema_ft", "tz_ema_ft"]]
    ax = axes[1,1]
    for name, data in ema_torques.items():
        ax.plot(x, data, label=name[:2])
    ax.set_title("EMA Torques")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend()
        
    # feeling_force = df[["fx_feeling_ft", "fy_feeling_ft", "fz_feeling_ft"]]
    # ax = axes[2,0]
    # for name, data in feeling_force.items():
    #     ax.plot(x, data, label=name[:2])
    # ax.set_title("Feeling Forces")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Force (N)")
    # ax.legend()
    
    # feeling_torque = df[["tx_feeling_ft", "ty_feeling_ft", "tz_feeling_ft"]]
    # ax = axes[2,1]
    # for name, data in feeling_torque.items():
    #     ax.plot(x, data, label=name[:2])
    # ax.set_title("Feeling Torques")
    # ax.set_xlabel("Time (s)")
    # ax.set_ylabel("Torque (Nm)")
    # ax.legend()
    
    feeling_force = df[["fy_ema_ft", "fy_feeling_ft"]]
    ax = axes[2,0]
    for name, data in feeling_force.items():
        ax.plot(x, data, label=name[:-3])
    ax.set_title("Feeling Force")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    
    feeling_torque = df[["ty_ema_ft", "ty_feeling_ft"]]
    ax = axes[2,1]
    for name, data in feeling_torque.items():
        ax.plot(x, data, label=name[:-3])
    ax.set_title("Feeling Torque")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend()
    
    wrist_linear = df[["vx_v_wrist", "vy_v_wrist", "vz_v_wrist"]]
    ax = axes[3,0]
    for name, data in wrist_linear.items():
        ax.plot(x, data, label=name[:2])
    ax.set_title("Wrist Linear Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linear Velocity (m/s)")
    ax.set_ylim([-0.05, 0.05])
    ax.legend()

    wrist_angular = df[["wx_v_wrist", "wy_v_wrist", "wz_v_wrist"]]
    ax = axes[3,1]
    for name, data in wrist_angular.items():
        ax.plot(x, data, label=name[:2])
    ax.set_title("Wrist Angular Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular Velocity (rad/s)")
    ax.set_ylim([-0.15, 0.32])
    ax.legend()

    plt.show()

# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()