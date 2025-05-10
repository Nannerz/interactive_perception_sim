import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, csv
# -----------------------------------------------------------------------------------------------------------
class FTPlotter():
    def __init__(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.csv_file = os.path.join(self.data_path, "ft_data.csv")
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        self.joint_ids = self.get_joint_ids()
        self.ft_types = ["fx", "fy", "fz", "tx", "ty", "tz"]
        self.max_samples = 2000
# -----------------------------------------------------------------------------------------------------------
    def get_joint_ids(self) -> list:
        # Read only the header row
        with open(self.csv_file, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
        
        ids = sorted({col.split('_', 1)[1] for col in header if '_' in col})
        
        return ids
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        fig, axes = plt.subplots(len(self.joint_ids), 1, sharex=True, figsize=(8, 12))
        lines = {}  # (id, type) -> Line2D object
        
        # initialize each subplot
        for ax, id in zip(axes, self.joint_ids):
            ax.set_title(f"Joint FT {id}")
            for type in self.ft_types:
                # start with an empty line for each type
                line, = ax.plot([], [], label=type)
                lines[(id, type)] = line
            ax.legend(loc="upper right")
            ax.set_ylabel("Force/Torque")

        axes[-1].set_xlabel("Sample #")

        def animate(frame):
            # read previous max_samples data & update plots
            df = pd.read_csv(self.csv_file).tail(self.max_samples)
            for i, id in enumerate(self.joint_ids):
                for type in self.ft_types:
                    line = lines[(id, type)]
                    line.set_data(df.index, df[f"{type}_{id}"])

                ax = axes[i]
                ax.relim()
                ax.autoscale_view()

            return list(lines.values())

        ani = animation.FuncAnimation(
            fig,
            animate,
            interval=500,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()
# -----------------------------------------------------------------------------------------------------------
def main() -> None:
    ft_plotter = FTPlotter()
    ft_plotter.run()
# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()