import json
import sys
import time

file = sys.argv[1]

with open(file, "r") as f:
    data = json.load(f)

print(data["A"] + data["B"])

time.sleep(1)

with open("is_finished.info", "w") as f:
    f.write("I'm finished af")
