import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# __init__() initiatizes nn.Module and define network's custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(6, 400)
        self.nonlinear_activation = nn.Sigmoid()
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 400)
        self.fc6 = nn.Linear(400, 400)
        self.fc7 = nn.Linear(400, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
# forward() completes a single forward pass through the network
# and return the output which should be a tensor
        hidden = self.fc1(input)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.fc2(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.fc3(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.fc4(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.fc5(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc6(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.dropout(hidden)
        output = self.fc7(hidden)
        output = self.nonlinear_activation(output)
        return output


    def evaluate(self, model, test_loader, loss_function):
# evaluate() returns the loss (a value, not a tensor) over the testing dataset.
        model.eval()
        loss = 0
        for idx, sample in enumerate(test_loader):
            with torch.no_grad():
                model_output = model(sample['input'])
                target_output = sample['label']
                target_output = torch.unsqueeze(target_output, 1)
                test_loss = loss_function(model_output, target_output)
                loss += test_loss.item()
        return loss / len(test_loader)

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
