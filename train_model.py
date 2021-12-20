from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 256
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.BCEWithLogitsLoss()
    save_path = "./saved/saved_model.pkl"
    learning_rate = 0.003
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    training_losses = []
    testing_losses = []
    missed_collisions = []
    false_positives = []
    max_training_loss = model.evaluate(model, data_loaders.train_loader, loss_function)
    max_testing_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    training_losses.append(max_training_loss)
    testing_losses.append(max_testing_loss)

    print("\n================ Training ================")
    train_count = len(data_loaders.train_loader)
    test_count = len(data_loaders.test_loader)
    missed_collisions.append(train_count)
    false_positives.append(train_count)
    print(f"Training with {train_count} samples...")

    for epoch_i in range(no_epochs):
        model.train()
        epoch_loss = 0
        for _, sample in enumerate(data_loaders.train_loader):
            input = torch.tensor(sample['input'], dtype=torch.float32)
            label = torch.tensor([sample["label"]], dtype=torch.float32)
            optimizer.zero_grad()
            output = model.forward(input)
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        missed_collision_count = 0
        false_positive_count = 0

        with torch.no_grad():
            for _, sample in enumerate(data_loaders.test_loader):
                input = torch.tensor(sample['input'], dtype=torch.float32)
                label = torch.tensor([sample["label"]], dtype=torch.float32)
                output = model.forward(input)

                expected = label[0].item()
                prediction = output[0].item()
                diff = abs(expected - prediction)

                if diff > 0.3:
                    if expected == 1:
                        missed_collision_count += 1
                    else:
                        false_positive_count += 1

        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        train_loss = epoch_loss/train_count
        testing_losses.append(test_loss)
        training_losses.append(train_loss)
        missed_collisions.append(missed_collision_count)
        false_positives.append(false_positive_count)
        print(f"-------------- [Epoch {epoch_i + 1}] ------------------")
        print(f"    Training Loss: {train_loss:.4f}")
        print(f"     Testing Loss: {test_loss:.4f}")
        print(f"Missed Collisions: {missed_collision_count} / {data_loaders.nav_dataset.collision_count}")
        print(f"  False Positives: {false_positive_count} / {data_loaders.nav_dataset.non_collision_count}")

    torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False, pickle_protocol=3)

    plt.plot(range(0, no_epochs + 1), missed_collisions, label='Missed Collisions')
    plt.plot(range(0, no_epochs + 1), false_positives, label='False Positive')
    plt.title('Missed Collisions, False Positives')
    plt.xlabel('Epochs')
    plt.ylabel('Counts')
    plt.xticks(range(0, no_epochs + 1))
    plt.legend()
    plt.show()

    # if test_count < 1000:
    #     multiplier = round(1000 / test_count)
    #     missed_collisions = missed_collisions*multiplier
    #     false_positives = false_positives*multiplier
    #
    # if missed_collisions == 0 and false_positives <= 10:
    #     print("Success, saving model...")
    #     torch.save(model.state_dict(), save_path, _use_new_zipfile_serialization=False, pickle_protocol=3)
    #     print("Model saved.")
    # else:
    #     print("Model did not achieve desired results.")


if __name__ == '__main__':
    no_epochs = 1000
    train_model(no_epochs)

