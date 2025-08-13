import csv
import os
import time
from typing import Dict, Iterable


class CsvStepLogger:
    def __init__(self, out_dir: str, filename_prefix: str = "harfang_rl_llm"):
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"{filename_prefix}_{ts}.csv")
        self.file = open(self.path, mode="w", newline="", encoding="utf-8")
        self.writer = None

    def log(self, row: Dict):
        if self.writer is None:
            fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


