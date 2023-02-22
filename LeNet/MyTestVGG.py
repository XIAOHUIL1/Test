# PyTorch
import torch
# Define the model
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d((2, 2)),
    torch.nn.Conv2d(64, 128, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d((2, 2)),
    torch.nn.Conv2d(128, 256, (3, 3), padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d((2, 2)),
    torch.nn.Flatten(),
    torch.nn.Linear(256 * 7 * 7, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 4096),
    torch.nn.ReLU(),
    torch.nn.Linear(4096, 1000),
    torch.nn.Softmax(dim=1)
)
# Print the model summary
print(model)