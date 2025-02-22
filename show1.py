from transformers import pipeline
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

classifier = pipeline('sentiment-analysis')
result = classifier('我想死了')
print(result)
