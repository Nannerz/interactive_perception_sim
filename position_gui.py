import json, os
import tkinter as tk
from tkinter import ttk

# class Position_GUI_Thread(threading.Thread):
#     def __init__(self, refresh_ms=100, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.daemon = True
#         self.interval = 0.01
#         self.gui = Position_GUI(refresh_ms=refresh_ms)

#     def run(self):
#         self.gui.mainloop()
        
class Position_GUI(tk.Tk):
    def __init__(self, refresh_ms=100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.title("End Effector Position")
        
        self.refresh_ms = refresh_ms
        self.pos_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pos.json")

        # Create labels for each value
        self.vars = {}
        for i, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            ttk.Label(self, text=f"{name.capitalize()}:").grid(row=i, column=0, sticky="e", padx=5, pady=2)
            var = tk.StringVar(value="---")
            ttk.Label(self, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=5)
            self.vars[name] = var

        # Kick off the periodic update
        self.after(self.refresh_ms, self._update)

    def _update(self) -> None:
        try:
            with open(self.pos_file, "r") as f:
                data = json.load(f)

            for key, var in self.vars.items():
                if key in data:
                    var.set(f"{data[key]:.3f}")
        except Exception as e:
            pass

        self.after(self.refresh_ms, self._update)
    
if __name__ == "__main__":
    Position_GUI(refresh_ms=100).mainloop()
    # Position_GUI_Thread().run()
