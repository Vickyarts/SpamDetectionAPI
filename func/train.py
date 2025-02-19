import os
import json
import pickle
import datetime
import argparse
import pymysql
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
from torch.nn import BCELoss
from torch.optim import Adam
from Model import SpamNN, Embedding, generate_token
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score





parser = argparse.ArgumentParser()
parser.add_argument("--admin_id", dest="admin_id", required=True)
parser.add_argument("--dataset_id", dest="dataset_id", required=True)
parser.add_argument("--job_id", dest="job_id", required=True)
parser.add_argument("--test_size", dest="test_size", default=0.1)
args = parser.parse_args()


with open('../config.json', 'r') as f:
    config = json.load(f)
    epochs = config['training']['epochs']
    db_config = config['training']['db']

cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"


embedding = Embedding()
NaiveBayes = GaussianNB()
Network = SpamNN().to(device)
NetOptimizer = Adam(Network.parameters(), lr=1e-4)
loss_fn = BCELoss()
model_id = generate_token(n=12)



def main():
    db_command(f"INSERT INTO api_trainingjob (job_id, dataset_id, status, created_by, created_at) VALUES ('{args.job_id}','{args.dataset_id}','Running',{args.admin_id},'{datetime.datetime.now()}')")
    try:
        log = open(f"training_logs/{args.job_id}_training.log", "a")
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Data Embedding Started.\n")
        naive_data, naive_test_data, network_data, network_test_data = load_dataset(args.dataset_id, args.test_size)
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Data Embedding completed.\n{ str(datetime.datetime.now()).split('.')[0] }: Naive Bayes Model training started.\n")
        NaiveBayes.fit(naive_data[0], naive_data[1])
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Naive Bayes Model training completed.\n{ str(datetime.datetime.now()).split('.')[0] }: Torch Model training started.\n")
        for epoch in range(1, epochs+1):
            loss_list = []
            for i in range(0, len(network_data), 64):
                if i+64 > len(network_data):
                    x = network_data[0][i:]
                    y = network_data[1][i:]
                else: 
                    x = network_data[0][i:i+64]
                    y = network_data[1][i:i+64]
                pred = Network(x)
                NetOptimizer.zero_grad()
                loss = loss_fn(pred, y)
                loss.backward()
                NetOptimizer.step()
                loss_list.append(loss.detach())
            log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Epoch: {epoch}/{epochs}  Epoch Loss: {np.mean(loss_list)}\n")
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Torch Model training completed.\n")
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Torch Model evaluation started.\n")
        with torch.no_grad():
            x = network_test_data[0]
            y = network_test_data[1]
            pred = Network(x)
            loss = loss_fn(pred, y)
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Torch Model evaluation completed.\n")
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Naive Bayes Model evaluation started.\n")
        naive_pred = NaiveBayes.predict(naive_test_data[0])
        acc = accuracy_score(naive_test_data[1], naive_pred)
        f1 = f1_score(naive_test_data[1], naive_pred)
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Naive Bayes Model evaluation completed.\n")
        net_loss = loss.data 
        db_command(f"INSERT INTO api_models (model_id, dataset_id, naive_accuracy, naive_f1_score, network_bce_loss, created_by, created_at) VALUES ('{model_id}','{args.dataset_id}',{acc},{f1},{net_loss},{args.admin_id},'{datetime.datetime.now()}')")
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Saving models....\n")
        torch.save(Network.state_dict(), f'../models/{model_id}.pth')
        with open(f'../models/{model_id}.pkl', 'wb') as f: 
            pickle.dump(NaiveBayes, f)
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Model Saved.\n")
        db_command(f"UPDATE api_trainingjob SET status = 'Completed' WHERE job_id = '{args.job_id}'")
    except Exception as e: 
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Error while training model. [ERROR]: {e}\n")
        db_command(f"UPDATE api_trainingjob SET status = 'Terminated' WHERE job_id = '{args.job_id}'")
        try:
            os.remove(f'../models/{model_id}.pth')
            os.remove(f'../models/{model_id}.pkl')
        except:
            pass
    finally:
        log.write(f"{ str(datetime.datetime.now()).split('.')[0] }: Training job ID:{args.job_id} completed.\n")
        log.close()




def load_dataset(dataset: str = 'dataset', test_size: float = 0.1): 
    path = f'../data/{dataset}.csv'
    df = pd.read_csv(path)
    X = df['message'].values #[:64]
    Y = df['label'].values #[:64]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=float(test_size), random_state=24)
    X_train_embed = embedding.model.embed_documents(map(str, X_train))
    X_test_embed = embedding.model.embed_documents(map(str, X_test))
    X_train = np.array(X_train_embed)
    X_test = np.array(X_test_embed)

    X_torch_train = torch.tensor(X_train_embed).float().to(device)
    X_torch_test = torch.tensor(X_test_embed).float().to(device)
    Y_torch_train = torch.tensor(Y_train.reshape(-1, 1)).float().to(device)
    Y_torch_test = torch.tensor(Y_test.reshape(-1, 1)).float().to(device)
    return (
        (X_train, Y_train.reshape(-1, 1)), 
        (X_test, Y_test.reshape(-1, 1)), 
        (X_torch_train, Y_torch_train), 
        (X_torch_test, Y_torch_test)
    )


def db_command(query):
    connection = None
    try:
        connection = pymysql.connect(host=db_config["HOST"], user=db_config["USER"], password=db_config["PASSWD"], database=db_config["NAME"])
        with connection.cursor() as cursor:
            cursor.execute(query)
            connection.commit()
    except pymysql.Error as e:
        print(f"Error: {e}")
        connection.rollback()
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    main()
