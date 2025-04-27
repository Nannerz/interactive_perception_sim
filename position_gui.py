import json, os, sys, threading
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
    _instance = None
    
    def __new__(cls, *args, **kwargs) -> 'Position_GUI':
        if cls._instance is None:
            cls._instance = super(Position_GUI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 shutdown_event: threading.Event=threading.Event(), 
                 refresh_ms=100, 
                 **kwargs) -> None:
        
        super().__init__(**kwargs)
        self.title("End Effector Position")
        self.shutdown_event = shutdown_event
        
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
        print("GUI thread main loop...")
        self.after(self.refresh_ms, self._update)

    def _update(self) -> None:
        if self.shutdown_event.is_set():
            print("Exiting position GUI thread.")
            self.destroy()
            sys.exit(0)
        
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
    Position_GUI().mainloop()