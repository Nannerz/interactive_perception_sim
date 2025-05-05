import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
# -----------------------------------------------------------------------------------------------------------
class VelocityPlotter():
    def __init__(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.csv_file = os.path.join(self.data_path, "vel_data.csv")
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        self.max_samples = 2000
# -----------------------------------------------------------------------------------------------------------
    def run(self) -> None:
        fig, axes = plt.subplots(1, 1, sharex=True, figsize=(4, 4))
        ax = axes[0] if type(axes) is list else axes
        ax.set_title(f"Joint Velocities")
        ax.legend(loc="upper right")
        ax.set_ylabel("value")
        lines = {}
        df = pd.read_csv(self.csv_file)
        
        # Plot each joint on the same graph
        for key in df.keys():
            line, = ax.plot([], [], label=key)
            lines[key] = line

        def animate(frame):
            # load last max_samples data
            df = pd.read_csv(self.csv_file).tail(self.max_samples)
            x = df.index
            
            # Update each line with new data
            for key in df.keys():
                y = df[key]
                line = lines[key]
                line.set_data(x, y)

            # Rescale/resize as needed
            ax.relim()
            ax.autoscale_view()

            return list(lines.values())
        
        ani = animation.FuncAnimation(
            fig,
            animate,
            interval=500,
            blit=True
        )

        plt.tight_layout()
        plt.show()
# -----------------------------------------------------------------------------------------------------------
def main() -> None:
    plotter = VelocityPlotter()
    plotter.run()
# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()