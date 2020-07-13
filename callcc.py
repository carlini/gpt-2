import requests
import base64
import inspect
import threading
import time
import sys
import pickle

def send_request(url, out):
    r = requests.get(url)
    out.append(r.content)


def wrap(function_ref):
    module_name = dict(inspect.getmembers(function_ref))['__globals__']['__file__'][:-3]
    module_name = base64.b64encode(module_name.encode("ascii")).decode("ascii")
    function_name = function_ref.__name__
    #print("OK", function_name)
    function_name = base64.b64encode(function_name.encode("ascii")).decode("ascii")
    def fn(*args, **kwargs):
        arguments = base64.b64encode(pickle.dumps([args, kwargs])).decode("ascii")
        open("/tmp/data.stdout", "w").write("")
        open("/tmp/data.stderr", "w").write("")
        output = []
        t = threading.Thread(target=send_request, args=["http://localhost:5000/"+module_name+"/"+function_name+"/"+arguments, output])
        last_stdout = 0
        last_stderr = 0
        
        t.start()
        while t.isAlive():
            stdout = open("/tmp/data.stdout").read()
            sys.stdout.write(stdout[last_stdout:])
            last_stdout = len(stdout)

            stderr = open("/tmp/data.stderr").read()
            sys.stderr.write(stderr[last_stderr:])
            last_stderr = len(stderr)
            
            time.sleep(.02)
        stdout = open("/tmp/data.stdout").read()
        sys.stdout.write(stdout[last_stdout:])

        stderr = open("/tmp/data.stderr").read()
        sys.stderr.write(stderr[last_stderr:])

        t.join()
        return pickle.loads(output[0])
    return fn
