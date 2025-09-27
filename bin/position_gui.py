import json, os
import tkinter as tk
from tkinter import ttk
from typing import Any
# -----------------------------------------------------------------------------------------------------------
class Position_GUI(tk.Tk):
    def __init__(self, 
                 refresh_ms: int = 100,
                 **kwargs: Any) -> None:
        
        super().__init__(**kwargs)
        self.title("End Effector Position")
        
        self.refresh_ms = refresh_ms
        self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.pos_file = os.path.join(self.data_path, "pos.json")
        self.prev_data: dict[str, Any] = {}
        if not os.path.exists(self.pos_file):
            raise FileNotFoundError(f"CSV file not found: {self.pos_file}")

        # Create labels for each value
        cntr = 0
        self.vars: dict[str, tk.StringVar] = {}
        for i, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            ttk.Label(self, text=f"{name.capitalize()}:").grid(row=i, column=0, sticky="e", padx=5, pady=2)
            var = tk.StringVar(value="---")
            ttk.Label(self, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=5)
            self.vars[name] = var
            cntr += 1

        ttk.Label(self, text=f"R_e2w").grid(row=cntr, column=0, sticky="e", padx=5, pady=2)
        for i, name in enumerate(("x", "y", "z")):
            for j in range(3):
                var = tk.StringVar(value="---")
                ttk.Label(self, textvariable=var, width=7).grid(row=i + cntr, column=j+1, sticky="w", padx=5)
                self.vars[f"r{name}{j+1}"] = var
            cntr += 1

        # register periodic update
        self.after(self.refresh_ms, self._update)
# -----------------------------------------------------------------------------------------------------------
    def _update(self) -> None:
        try:
            with open(self.pos_file, "r") as f:
                data = json.load(f)
            self.prev_data = data
        except:
            data = self.prev_data
        
        for key, var in self.vars.items():
            if key in data:
                var.set(f"{data[key]:.3f}")

        self.after(self.refresh_ms, self._update)
# -----------------------------------------------------------------------------------------------------------
def main() -> None:
    position_gui = Position_GUI()
    position_gui.mainloop()
# -----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()