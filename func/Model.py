import random
import string
import torch
import torch.nn as nn
from langchain_huggingface import HuggingFaceEmbeddings



cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"


class SpamNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fcx = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sig(self.fcx(x))
        return x


class Embedding():
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device },
        encode_kwargs={"normalize_embeddings": False}
    )

    def __init__(self):
        pass 

    def embed_doc(self, text: str):
        return self.model.embed_documents([text])
    

def generate_token(n: int = 24):
    return ''.join(random.choices(string.ascii_letters, k=n))