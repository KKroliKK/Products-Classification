import numpy as np
import torch

from torch.nn.functional import softmax

device = 'cuda' if torch.cuda.is_available() else 'cpu'
            

def test_model(model, dataloader):
    model.eval()

    y_true = []
    y_pred = []

    correct_all = 0
    all = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        with torch.no_grad():
            outp = model(x_batch)

        preds = outp.argmax(-1)

        y_true += y_batch.tolist()
        y_pred += preds.tolist()

        correct_batch = (preds == y_batch).sum()
        correct_all += correct_batch.item()
        all += len(outp)

    return y_true, y_pred


def predict(model, dataloader, index_to_id):
    model.eval()
    model.to(device)
    y_pred = []

    for x_batch in dataloader:
        x_batch = x_batch.to(device)

        model.eval()
        with torch.no_grad():
            outp = model(x_batch)

        preds = outp.argmax(-1)
        y_pred += preds.tolist()

    predictions = [index_to_id[id] for id in y_pred]

    return predictions


def predict_sample(model, emb, index_to_id):
    model.eval()
    model.cpu()

    with torch.no_grad():
        outp = model(torch.tensor(np.expand_dims(emb, axis=0)))
    prob = softmax(outp, dim=1)
    index = int(prob.argmax(-1))
    id = index_to_id[index]

    return id