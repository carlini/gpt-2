from flask import Flask
from flask import request
import importlib
import base64
import sys
import traceback
import pickle

app = Flask(__name__)

class Unbuffered(object):
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
    
def do_it(name, which, args):
    name = base64.b64decode(name).decode('ascii')
    which = base64.b64decode(which).decode('ascii')
    args, kwargs = pickle.loads(base64.b64decode(args))
    module = __import__(name)
    importlib.reload(module)
    fout = Unbuffered(open("/tmp/data.stdout","w"))
    ferr = Unbuffered(open("/tmp/data.stderr","w"))
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = fout, ferr
    try:
        res = getattr(module, which)(*args, **kwargs)
    except Exception as err:
        #traceback.print_tb(err.__traceback__)
        traceback.print_exc()
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    fout.close()
    ferr.close()
    return pickle.dumps(res)

@app.route("/<name>/<function>/<args>", methods=['GET'])
def do(name, function, args):
    return do_it(name, function, args)

if __name__ == "__main__":
    app.run()
