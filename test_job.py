from tron_sdk import Tron

tron = Tron(
    "http://127.0.0.1:9000"
)

job = tron.run(
    "train model on images"
)

print(
    "[JOB ID]",
    job.job_id
)

result = job.wait()

print(
    "[RESULT]",
    result
)