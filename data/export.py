# Exports data to a file
import json

def data_export(data, where):
    with open(where, "w") as f:
        f.write(json.dumps(data))

def data_import(where):
    with open(where, "r") as f:
        data = f.read()
        return json.loads(data)
