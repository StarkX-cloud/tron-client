import tron
import time

print("\n=== TRON SIMPLE REMOTE DEMO ===\n")


@tron.remote
def giant_ai_training():

    time.sleep(5)

    return {
        "model": "TRON-ULTRA",
        "parameters": "70B"
    }


result = giant_ai_training.run()

print("\nJOB COMPLETE:\n")

print(result)