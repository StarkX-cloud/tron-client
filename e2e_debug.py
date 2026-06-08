import subprocess, time, sys, os
exe=sys.executable
env=os.environ.copy()
env['TRON_PORT']='9100'
p1=subprocess.Popen([exe, 'queue_server.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(3)
env2=os.environ.copy()
env2['TRON_URL']='http://127.0.0.1:9100'
p2=subprocess.Popen([exe, 'worker.py'], env=env2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(3)
try:
    proc=subprocess.run([exe, 'test_sdk.py'], env=env2, capture_output=True, text=True, timeout=60)
    print('=== TEST STDOUT ===')
    print(proc.stdout)
    print('=== TEST STDERR ===')
    print(proc.stderr)
finally:
    p2.terminate()
    p1.terminate()
    out1,err1=p1.communicate(timeout=5)
    out2,err2=p2.communicate(timeout=5)
    print('=== SERVER OUT ===')
    print(out1)
    print('=== SERVER ERR ===')
    print(err1)
    print('=== WORKER OUT ===')
    print(out2)
    print('=== WORKER ERR ===')
    print(err2)
