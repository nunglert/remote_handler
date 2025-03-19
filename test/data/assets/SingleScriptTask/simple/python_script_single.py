
import sys
import time
import json

file = sys.argv[1]

with open(file, "r") as f:
    data = json.load(f)

print(data["A"] + data["B"])

time.sleep(5)

with open("is_finished.info", "w") as f:
    f.write("I'm finished af")
