import subprocess, time, os, sys
exe=sys.executable
env=os.environ.copy()
env['TRON_URL']='http://127.0.0.1:9100'
env['PYTHONUNBUFFERED']='1'
p=subprocess.Popen([exe, '-u', 'worker.py'], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
time.sleep(2)
proc=subprocess.run([exe, '-u', 'test_sdk.py'], env=env, capture_output=True, text=True, timeout=90)
print('=== TEST STDOUT ===')
print(proc.stdout)
print('=== TEST STDERR ===')
print(proc.stderr)
p.terminate()
out,err=p.communicate(timeout=10)
print('=== WORKER OUT ===')
print(out)
print('=== WORKER ERR ===')
print(err)
