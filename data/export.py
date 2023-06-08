# Exports data to a file
import json

def data_export(data, where):
    with open(where, "w") as f:
        f.write(json.dumps(data))
