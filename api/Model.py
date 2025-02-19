import os
import random
import string
import pickle
import numpy as np 
import torch
import torch.nn as nn
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.naive_bayes import GaussianNB
from .models import  Deployment




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




class SpamDetection():
    device = device
    def __init__(self, mode: str = 'naive-bayes', model_id: str = None):
        self.mode = mode
        self.model_id = model_id
        self.embedding = Embedding()
        self.NaiveBayes = GaussianNB()
        self.Network = SpamNN().to(device)
        if model_id:
            self.load_weights(model_id=model_id)

    def load_weights(self, model_id: str = None):
        nai_path = f'models/{model_id}.pkl'
        net_path = f'models/{model_id}.pth'
        if self.mode == 'naive-bayes':
            if os.path.exists(nai_path):
                with open(nai_path, 'rb') as f:
                    self.NaiveBayes = pickle.load(f)
            else: 
                raise FileNotFoundError(
                    f'Naive Bayes model "{model_id}" not found.'
                )
        elif self.mode == 'nn':
            if os.path.exists(net_path):
                self.Network.load_state_dict(torch.load(net_path))
            else: 
                raise FileNotFoundError(
                    f'NN model "{model_id}" not found.'
                )
        else:
            if os.path.exists(nai_path) and os.path.exists(net_path):
                with open(nai_path, 'rb') as f:
                    self.NaiveBayes = pickle.load(f)
                self.Network.load_state_dict(torch.load(net_path))
            else: 
                raise FileNotFoundError(
                    f'{model_id} model not found.'
                )

    def inference(self, text: str = None):
        with torch.no_grad():
            if self.mode == 'naive-bayes':
                embeds = self.embedding.embed_doc(text)
                try:
                    pred = self.NaiveBayes.predict(embeds)
                except:
                    return 'Model not trained.'
                return pred[0]
            elif self.mode == 'nn':
                embeds = torch.tensor(self.embedding.embed_doc(text)).to(self.device)
                return self.Network(embeds)[0, 0]
            else: 
                embeds = self.embedding.embed_doc(text)
                torch_embeds = torch.tensor(embeds).to(self.device)
                try:
                    nn_pred = round(float(self.Network(torch_embeds)[0, 0]), 6)
                    bay_pred = round(float(self.NaiveBayes.predict_proba(embeds)[0][1]), 6)
                    pred = (bay_pred * 1.3) + (nn_pred * 1.7)
                    return round(pred / 3)
                except:
                    return 'Model not trained.'




def labelEncoding(labels, cls: dict = {'ham': 0, 'spam': 1}):
    encoded = []
    for label in labels:
        l = label[0].lower()
        encoded.append([cls[l]])
    return np.array(encoded)

def generate_token(n: int = 24):
    return ''.join(random.choices(string.ascii_letters, k=n))

def get_Deployment_Config():
    deployment = Deployment.objects.all().last()
    if deployment:
        return {
            "deployed-model": deployment.model_id,
            "inference-type": deployment.type
        }
    else: 
        return {
            "deployed-model": None,
            "inference-type": "nn"
        }
