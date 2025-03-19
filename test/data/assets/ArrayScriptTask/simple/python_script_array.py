
import sys
import time
import json
from pathlib import Path

calcpath = Path(sys.argv[1])
INPUT_FILE = calcpath / "input.json"
CONFIG_FILE = calcpath / "config.json"

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)
    print(config)

print(data["A"] + data["B"])

time.sleep(5)

with open(calcpath / "is_finished.info", "w") as f:
    f.write("I'm finished af")
