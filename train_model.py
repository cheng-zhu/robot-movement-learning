from Data_Loaders import Data_Loaders
from Ori_Data_Loaders import Ori_Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn


def train_model(no_epochs):

    batch_size = 64
    rate_learning = 0.0001
    data_loaders = Data_Loaders(batch_size)
    ori_data_loaders = Ori_Data_Loaders(batch_size)
    model = Action_Conditioned_FF()

    loss_function = nn.MSELoss()

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning)
    for epoch_i in range(no_epochs):
        model.train()
        train_loss = 0
        for idx, sample in enumerate(data_loaders.train_loader):
            optimizer.zero_grad()
            model_output = model(sample['input'])
            target_output = sample['label']
            target_output = torch.unsqueeze(target_output, 1)

            loss = loss_function(model_output, target_output)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(model.evaluate(model, data_loaders.test_loader, loss_function))
    torch.save(model.state_dict(), 'saved/saved_model.pkl',  _use_new_zipfile_serialization=False)
    real_ones = 0
    real_zeros = 0
    false_ones = 0
    false_zeros = 0
    model.eval()
    for idx, sample in enumerate(ori_data_loaders.train_loader):
        model_output = torch.squeeze(model(sample['input'])).detach().numpy().tolist()
        target_output = sample['label'].detach().numpy().tolist()
        for i in range(len(model_output)):
            if target_output[i] == 1:
                real_ones += 1
            else:
                real_zeros += 1
            if model_output[i] < 0.5 and target_output[i] == 1:
                false_zeros += 1
            if model_output[i] > 0.5 and target_output[i] == 0:
                false_ones += 1
    for idx, sample in enumerate(ori_data_loaders.test_loader):
        model_output = torch.squeeze(model(sample['input'])).detach().numpy().tolist()
        target_output = sample['label'].detach().numpy().tolist()
        for i in range(len(model_output)):
            if target_output[i] == 1:
                real_ones += 1
            else:
                real_zeros += 1
            if model_output[i] < 0.5 and target_output[i] == 1:
                false_zeros += 1
            if model_output[i] > 0.5 and target_output[i] == 0:
                false_ones += 1
    print("fp: ", false_ones/real_zeros)
    print("fn: ", false_zeros/real_ones)
    print("acc: ", (1 - (false_ones+false_zeros)/(real_zeros+real_ones)) * 100)


if __name__ == '__main__':
    no_epochs = 45
    train_model(no_epochs)
