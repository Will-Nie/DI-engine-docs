import time
import atexit
import json
from datetime import datetime
from rich import print


class Recorder():
    def __init__(self, task_name, seed):
        # Calculate execution time
        # Save statistics to report.txt
        self.start_time = time.time()
        self.task_name = task_name
        self.seed = seed
        self.extra = {}
        self._record = False

    def start(self):
        self.start_time = time.time()
        self.extra = {}
        self._record = False

    def add(self, key, value):
        self.extra[key] = value

    def record(self):
        duration = time.time() - self.start_time
        data = {
            "start": datetime.fromtimestamp(self.start_time).isoformat(),
            "task": self.task_name,
            "seed": self.seed,
            "duration": duration
        }
        for key, value in self.extra.items():
            data[key] = value
        with open("./reports/report_{}.txt".format(self.task_name), "a+") as f:
            report = json.dumps(data)
            print(report)
            f.write(report + "\n")
        self._record = True
        self.start_time = time.time()

