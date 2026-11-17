"""Write role data in bounded chunks, then stream the case manifest."""
from pathlib import Path
import json
import shutil

from experiments.numerics_rows.common import write_json


class RoleFile:
    def __init__(self, path, role):
        self.path = Path(path).with_suffix('.json')
        write_json(self.path, role)


def write_manifest(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    def emit(value, stream):
        if isinstance(value, RoleFile):
            with value.path.open() as source:
                shutil.copyfileobj(source, stream, 65536)
        elif isinstance(value, dict):
            stream.write('{')
            for index, (key, item) in enumerate(value.items()):
                stream.write((',' if index else '') + json.dumps(key) + ':')
                emit(item, stream)
            stream.write('}')
        elif isinstance(value, list):
            stream.write('[')
            for index, item in enumerate(value):
                if index: stream.write(',')
                emit(item, stream)
            stream.write(']')
        else:
            stream.write(json.dumps(value, allow_nan=False))
    with pending.open('w') as stream:
        emit(value, stream)
        stream.write('\n')
    pending.replace(path)
