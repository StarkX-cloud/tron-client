import tron
print('TRON SDK import ok')
url = tron.ensure_server()
print('ensure_server ->', url)

@tron.remote
def add(a, b):
    return a + b


if __name__ == "__main__":
    future = add(2, 3)
    print('future created:', future)
    result = future.get(timeout=10)
    print('future result:', result)
