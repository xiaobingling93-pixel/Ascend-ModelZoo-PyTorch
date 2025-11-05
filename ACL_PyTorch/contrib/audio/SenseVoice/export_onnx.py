import argparse
from funasr import AutoModel
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model path")
args = parser.parse_args()
model = AutoModel(model=args.model, device="cpu")

res = model.export(quantize=False)
print(res)