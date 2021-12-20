import torch
import torch.nn as nn


class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, output_size=1, dropout=0.05):
        super(Action_Conditioned_FF, self).__init__()
        l1_size = 256
        l2_size = 128
        l3_size = 256
        l4_size = 64
        l5_size = 8
        self.input_to_l1 = nn.Linear(input_size, l1_size)
        self.l1_to_l2 = nn.Linear(l1_size, l2_size)
        self.l2_to_l3 = nn.Linear(l2_size, l3_size)
        self.l3_to_l4 = nn.Linear(l3_size, l4_size)
        self.l4_to_l5 = nn.Linear(l4_size, l5_size)
        self.l5_to_output = nn.Linear(l5_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.relu(self.input_to_l1(input))
        hidden = self.relu(self.l1_to_l2(hidden))
        hidden = self.relu(self.l2_to_l3(hidden))
        hidden = self.relu(self.l3_to_l4(hidden))
        hidden = self.relu(self.l4_to_l5(hidden))
        output = self.l5_to_output(hidden)
        output = self.sigmoid(output)
        return output

    def evaluate(self, model, test_loader, loss_function):
        total_loss = 0
        for _, sample in enumerate(test_loader):
            # print("DATA", data)
            input = torch.tensor(sample['input'], dtype=torch.float32)
            label = torch.tensor([sample["label"]], dtype=torch.float32)
            output = model.forward(input)
            loss = loss_function(output, label)
            # print("LOSS: ", loss)
            total_loss += loss.item()
        # print("TOTAL LOSS: ", total_loss)
        return float(total_loss / len(test_loader))


def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
