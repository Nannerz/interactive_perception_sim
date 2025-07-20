import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os, csv
# -----------------------------------------------------------------------------------------------------------
class FTPlotter():
    def __init__(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.csv_file = os.path.join(self.data_path, "ft_data.csv")
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        self.data_map = self.get_data_map()
        self.max_samples = 500
# -----------------------------------------------------------------------------------------------------------
    def get_data_map(self) -> list:
        with open(self.csv_file, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)

        data_map = {}
        for col in header:
            if '_' in col:
                data_type, name = col.split('_', 1)
                if name not in data_map:
                    data_map[name] = {}
                data_map[name][data_type] = 0

        return data_map
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        fig, axes = plt.subplots(len(self.data_map), 1, sharex=True, figsize=(8, 12))
        lines = {}  # (id, type) -> Line2D object
        if len(self.data_map) == 1:
            axes = [axes]
        
        for ax, name in zip(axes, self.data_map):
            ax.set_title(f"{name}")
            for val in self.data_map[name]:
                # start with an empty line for each type
                line, = ax.plot([], [], label=f"{val}_{name}")
                lines[(name, val)] = line
            ax.legend(loc="upper right")
            ax.set_ylabel(".")

        axes[-1].set_xlabel(".")

        def animate(frame):
            # read previous max_samples data & update plots
            df = pd.read_csv(self.csv_file).tail(self.max_samples)
            x = np.arange(len(df), dtype=float)
                    
            for i, name in enumerate(self.data_map):
                for val in self.data_map[name]:
                    line = lines[(name, val)]
                    y = df[f"{val}_{name}"].to_numpy(dtype=float)
                    line.set_data(x, y)

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