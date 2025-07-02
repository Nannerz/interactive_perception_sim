import threading
import csv
import queue
import json

class DataWriter(threading.Thread):
    def __init__(self, data_q, shutdown_event, ft_file, pos_file, csv_keys):
        self.data_q = data_q
        self.shutdown_event = shutdown_event
        self.ft_file = ft_file
        self.pos_file = pos_file
        self.csv_keys = csv_keys
        self.writer_thread = threading.Thread(target=self.data_writer_thread)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        
    def data_writer_thread(self):
        """Dedicated thread for file writing operations"""
        while not self.shutdown_event.is_set():
            try:
                item = self.data_q.get(timeout=0.5)
                if item["type"] == "csv":
                    with open(self.ft_file, "a", newline="") as csv_file:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=self.csv_keys)
                        csv_writer.writerow(item["data"])

                elif item["type"] == "json":
                    with open(self.pos_file, "w") as f:
                        json.dump(item["data"], f)
                        
                self.data_q.task_done()
            except queue.Empty:
                pass

if __name__ == "__main__":
    print("This class should not be run directly.")
    exit(1)