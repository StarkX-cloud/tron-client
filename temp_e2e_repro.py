import subprocess
import time
import sys
import os

exe = sys.executable

env1 = os.environ.copy()
env1['TRON_PORT'] = '9100'
server = subprocess.Popen([exe, '-u', 'queue_server.py'], env=env1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print('server pid', server.pid)
sys.stdout.flush()
time.sleep(3)

env2 = os.environ.copy()
env2['TRON_URL'] = 'http://127.0.0.1:9100'
worker = subprocess.Popen([exe, '-u', 'worker.py'], env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print('worker pid', worker.pid)
sys.stdout.flush()
time.sleep(3)

proc = subprocess.Popen([exe, '-u', 'test_sdk.py'], env=env2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print('test pid', proc.pid)
sys.stdout.flush()

start = time.time()
while True:
    line = proc.stdout.readline()
    if line:
        sys.stdout.write('[TEST] ' + line)
        sys.stdout.flush()
    if proc.poll() is not None:
        break
    if time.time() - start > 120:
        print('TIMEOUT test 120s')
        sys.stdout.flush()
        proc.kill()
        break
print('test exit', proc.poll())
sys.stdout.flush()

end = time.time() + 5
while time.time() < end:
    for p, name in [(server, 'SERVER'), (worker, 'WORKER')]:
        if p.stdout is None:
            continue
        line = p.stdout.readline()
        if line:
            sys.stdout.write(f'[{name}] ' + line)
            sys.stdout.flush()
    time.sleep(0.1)

server.terminate()
worker.terminate()
print('done')
sys.stdout.flush()
