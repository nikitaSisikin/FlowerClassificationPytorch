import torch
import torchvision
import argparse
import json
import os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print(torch.__version__)
print(torchvision.__version__)

class FlowerClassifier:
    """
    Flower image classifier using PyTorch and transfer learning.

    Attributes:
        data_dir (str): Path to the data directory containing train, valid, and test sets.
        save_dir (str): Directory to save checkpoints.
        arch (str): Architecture name (e.g., 'resnet18' or 'densenet121').
        learning_rate (float): Learning rate for training.
        hidden_units (int): Number of hidden units in the classifier.
        epochs (int): Number of training epochs.
        gpu (bool): Flag to enable GPU training.
    """

    def __init__(self, data_dir, save_dir='checkpoints', arch='resnet18', learning_rate=0.001, hidden_units=1024, epochs=10, gpu=False):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.gpu = gpu

        # Load and preprocess the data
        self.train_data, self.valid_data, self.test_data = self.load_data()

        # Build the model and specify parameters to optimize
        self.model, self.parameters_to_optimize = self.build_model()

        # Loss function and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters_to_optimize, lr=self.learning_rate)

        # Learning rate scheduler
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

    def load_data(self):
        """
        Load and preprocess the image data.

        Returns:
            train_loader (DataLoader): DataLoader for the training set.
            valid_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
        """
        # Define data transforms for train, valid, and test sets
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(degrees=(-30, 30)),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # Load datasets
        data_sets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in ['train', 'valid', 'test']}
        
        # Create data loaders
        data_loaders = {x: DataLoader(data_sets[x], batch_size=64, shuffle=True) for x in ['train', 'valid', 'test']}
        
        return data_loaders['train'], data_loaders['valid'], data_loaders['test']

    def build_model(self):
        """
        Build and configure the neural network model.

        Returns:
            model (nn.Module): The neural network model.
        """
        # Load pre-trained model
        if self.arch == 'resnet18':
            model = models.resnet18()
            num_ftrs = model.fc.in_features
        elif self.arch == 'densenet121':
            model = models.densenet121()
            num_ftrs = model.classifier.in_features
        else:
            raise ValueError("Unsupported architecture")

        # Freeze parameters
        if self.arch == 'resnet18':
            for param in model.parameters():
                param.requires_grad = False
            # Modify the classifier
            classifier = nn.Sequential(
                nn.Linear(num_ftrs, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_units, 102),
                nn.LogSoftmax(dim=1)
            )
            model.fc = classifier
        elif self.arch == 'densenet121':
            for param in model.parameters():
                param.requires_grad = False
            # Modify the classifier
            classifier = nn.Sequential(
                nn.Linear(num_ftrs, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_units, 102),
                nn.LogSoftmax(dim=1)
            )
            model.classifier = classifier

        # Specify the parameters to optimize
        parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())

        return model, parameters_to_optimize
    
    def train(self):
        """
        Train the neural network model and save checkpoints.
        """
        device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        self.model.to(device)

        best_val_loss = float('inf')
        patience = 3  # Early stopping patience
        early_stopping_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0

            for inputs, labels in self.train_data:
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # Validation on the validation set
            self.model.eval()
            val_loss, val_accuracy = self.validate(self.valid_data)

            # Validation on the testing set
            test_loss, test_accuracy = self.validate(self.test_data)

            print(f"Epoch {epoch + 1}/{self.epochs}.. "
                  f"Train loss: {running_loss / len(self.train_data):.3f}.. "
                  f"Validation loss: {val_loss:.3f}.. "
                  f"Validation accuracy: {val_accuracy:.3f}.. "
                  f"Testing loss: {test_loss:.3f}.. "
                  f"Testing accuracy: {test_accuracy:.3f}")

            self.scheduler.step()  # Step the learning rate scheduler

            # Early stopping based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                self.save_checkpoint()
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def validate(self, data_loader):
        """
        Validate the neural network model.

        Args:
            data_loader (DataLoader): DataLoader for validation or test set.

        Returns:
            val_loss (float): Validation loss.
            val_accuracy (float): Validation accuracy.
        """
        val_loss = 0
        accuracy = 0
        device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = self.model.forward(inputs)
                batch_loss = self.criterion(logps, labels)
                val_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        return val_loss / len(data_loader), accuracy / len(data_loader)

    def save_checkpoint(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        checkpoint = {
            'arch': self.arch,
            'learning_rate': self.learning_rate,
            'hidden_units': self.hidden_units,
            'epochs': self.epochs,
            'gpu': self.gpu,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'class_to_idx': self.train_data.dataset.class_to_idx,
        }

        save_path = os.path.join(self.save_dir, 'checkpoint.pth')
        torch.save(checkpoint, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Image Classifier")
    parser.add_argument("data_dir", help="Path to the data directory")
    parser.add_argument("--save_dir", default="saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", default="resnet18", help="Architecture (e.g., resnet18 or densenet121)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=1024, help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print("Error: The specified data directory does not exist.")
        print("Please provide the correct path to the data directory.")
        print("Usage: python train.py data_dir [options]")
        exit(1)

    if args.arch not in ["resnet18", "densenet121"]:
        print("Error: Unsupported architecture. Supported architectures: 'resnet18', 'densenet121'.")
        print("Usage: python train.py data_dir [options]")
        exit(1)

    # Determine the device (CPU or GPU)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    classifier = FlowerClassifier(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        gpu=args.gpu
    )

    try:
        classifier.train()
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
