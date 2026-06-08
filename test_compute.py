from tron_sdk import Tron

tron = Tron("http://127.0.0.1:9000")

result = tron.compute(
    "train model on images"
)

print(result)