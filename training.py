import numpy as np
import pandas as pd
from IPython.display import clear_output
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from inference import test_model
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_labels(targets, target_to_index=None):
    if target_to_index == None:
        unique = np.unique(targets)
        labels = [np.where(unique == code)[0][0] for code in targets]
        index_to_target = {index: code for index, code in enumerate(unique)}
        target_to_index = {code: index for index, code in enumerate(unique)}
        return np.array(labels), index_to_target, target_to_index
    else:
        labels = [target_to_index[code] for code in targets]
        return np.array(labels)


def train_model(
    model, 
    model_name: str,
    train_loader, 
    valid_loader,
    num_epochs=10,
    plot=False,
    erase_epochs_info=True
):
    
    model.double()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loaders = {"train": train_loader, "valid": valid_loader}
    f1 = {"train": [], "valid": []}
    best_f1 = 0


    for epoch in tqdm(range(num_epochs)):
        for k, dataloader in loaders.items():
            epoch_preds = []
            epoch_ys = []

            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)

                preds = outp.argmax(-1)
                epoch_preds += preds.tolist()
                epoch_ys += y_batch.tolist()

                if k == "train":
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()

            if k == 'train':
                scheduler.step()

            f1_epoch = f1_score(epoch_ys, epoch_preds, average='weighted')
            f1[k].append(f1_epoch)
            
            if k == 'valid' and f1['valid'][-1] > best_f1:
                torch.save(model.state_dict(), f'./models/_{model_name}.pt')
                best_f1 = f1['valid'][-1]
            if k == 'train':
                print(f'\nEpoch: {epoch + 1}')

            print(f"Loader: {k}. f1 score: {round(f1_epoch, 4)}")

    model.load_state_dict(torch.load(f'./models/{model_name}.pt', map_location=device))

    if erase_epochs_info == True:
        clear_output()

    if plot == True:
        plt.plot(f1['train'], label='train')
        plt.plot(f1['valid'], label='valid')
        plt.xlabel('epoch')
        plt.ylabel('f1 score')
        plt.legend()
        plt.show()

    print('\nTrain f1 score:', round(max(f1['train']), 4))
    print('Valid f1 score:', round(max(f1['valid']), 4))


def get_loaders(dataset, embeddings, target_col: str,  batch_size=4, num_workers=2):
    train_idxs = dataset['train']
    valid_idxs = dataset['valid']
    test_idxs = dataset['test']


    embeddings = embeddings.astype(np.double)
    x_train = embeddings[train_idxs]
    x_valid = embeddings[valid_idxs]
    x_test = embeddings[test_idxs]

    y = dataset[target_col]
    y_train, index_to_target, target_to_index = create_labels(y[train_idxs])
    y_valid = create_labels(y[valid_idxs], target_to_index)
    y_test = create_labels(y[test_idxs], target_to_index)

    train_loader = DataLoader(
        list(zip(x_train, y_train)), 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=True, 
        drop_last=True
    )

    valid_loader = DataLoader(
        list(zip(x_valid, y_valid)), 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        drop_last=False
    )

    test_loader = DataLoader(
        list(zip(x_test, y_test)), 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        drop_last=False
    )

    return train_loader, valid_loader, test_loader


def ad_hoc_train(
    emb, 
    model_name, 
    num_epochs=30, 
    hidden_layers=2, 
    hid_size=100, 
    plot=False, 
    model=None, 
    erase_epochs_info=True
):
    dataset = pd.read_csv('./dataset.csv')
    labels, index_to_id, id_to_index = create_labels(dataset.category_id)

    train_loader, valid_loader, test_loader = get_loaders(dataset, emb, 'category_id', batch_size=1024)

    input_size = emb.shape[1]
    hid_size = 100
    num_classes = len(index_to_id)

    if model == None:
        if hidden_layers == 2:
            model = nn.Sequential(
                nn.Linear(input_size, hid_size),
                nn.BatchNorm1d(hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, hid_size),
                nn.BatchNorm1d(hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, num_classes)
            )
        elif hidden_layers == 1:
            model = nn.Sequential(
                nn.Linear(input_size, hid_size),
                nn.BatchNorm1d(hid_size),
                nn.ReLU(),
                nn.Linear(hid_size, num_classes)
            )
        elif hidden_layers == 0:
            model = nn.Linear(input_size, num_classes)

    train_model(
        model, 
        model_name, 
        train_loader, 
        valid_loader, 
        num_epochs=num_epochs, 
        plot=plot,
        erase_epochs_info=erase_epochs_info
    )

    y_true, y_pred = test_model(model, test_loader)
    test_score = f1_score(y_pred, y_true, average='weighted')
    print(' Test f1 score:', round(test_score, 3))


def train_parametrized( 
    model, 
    model_name: str,
    dataloader,
    num_epochs=10,
    erase_epochs_info=True,
    optimizer=torch.optim.AdamW,
    lr=1e-3,
    step_size=7,
    gamma=0.1
):
    model.double()
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma)

    f1 = []

    for epoch in tqdm(range(num_epochs)):
        epoch_preds = []
        epoch_ys = []

        for x_batch, y_batch in dataloader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outp = model(x_batch)

            preds = outp.argmax(-1)
            epoch_preds += preds.tolist()
            epoch_ys += y_batch.tolist()

            loss = criterion(outp, y_batch)
            loss.backward()
            optimizer.step()

        scheduler.step()

        f1_epoch = f1_score(epoch_ys, epoch_preds, average='weighted')
        f1.append(f1_epoch)
        
        print(f'\nEpoch: {epoch + 1}')
        print(f"f1 score: {round(f1_epoch, 4)}")

    if erase_epochs_info == True:
        clear_output()

    print('\nTrain f1 score:', round(max(f1), 4))
    torch.save(model.state_dict(), f'./models/{model_name}.pt')