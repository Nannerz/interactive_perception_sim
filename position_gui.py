import json
import tkinter as tk
from tkinter import ttk
import os

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pos.json")

class EEViewer(tk.Tk):
    def __init__(self, refresh_ms=100):
        super().__init__()
        self.title("End‑Effector State")
        self.refresh_ms = refresh_ms

        # Create labels for each value
        self.vars = {}
        for i, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            ttk.Label(self, text=f"{name.capitalize()}:").grid(row=i, column=0, sticky="e", padx=5, pady=2)
            var = tk.StringVar(value="---")
            ttk.Label(self, textvariable=var, width=10).grid(row=i, column=1, sticky="w", padx=5)
            self.vars[name] = var

        # Kick off the periodic update
        self.after(self.refresh_ms, self._update)

    def _update(self):
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
            # Update each label if present
            for key, var in self.vars.items():
                if key in data:
                    var.set(f"{data[key]:.3f}")
        except Exception as e:
            print("DEBUG: got exception: ", e)
            # If file missing or invalid, just skip this cycle
            pass
        # schedule next refresh
        self.after(self.refresh_ms, self._update)


if __name__ == "__main__":
    app = EEViewer(refresh_ms=100)  # update every 0.1 s
    app.mainloop()
