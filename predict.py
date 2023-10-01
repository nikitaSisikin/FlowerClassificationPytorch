import argparse
import torch
from torchvision import transforms
from PIL import Image
from json_util import load_category_names
from torchvision import models
from torch import nn

def create_model(arch, hidden_units):
    """
    Create a pre-trained model with a custom classifier.

    Args:
        arch (str): Architecture name (e.g., 'resnet18' or 'densenet121').
        hidden_units (int): Number of hidden units in the classifier.

    Returns:
        model (torch.nn.Module): Pre-trained model with a custom classifier.
    """
    if arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        classifier = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.fc = classifier
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        classifier = nn.Sequential(
            nn.Linear(num_ftrs, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    else:
        raise ValueError("Unsupported architecture")

    return model


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    # Load the model architecture and parameters
    arch = checkpoint['arch']
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    epochs = checkpoint['epochs']
    gpu = checkpoint['gpu']

    # Create the model using the architecture specified in the checkpoint
    model = create_model(arch, hidden_units)

    # Load the state dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Retrieve the class-to-index mapping from the model state dict
    class_to_idx = checkpoint.get('class_to_idx', None)

    return model, class_to_idx, learning_rate, hidden_units, epochs, gpu

def process_image(image_path):
    """
    Process an image for prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        image (torch.Tensor): Processed image as a PyTorch tensor.
    """
    img = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(img)
    return image

def predict(image_path, model, top_k=1, category_names=None, gpu=False):
    """
    Predict the class of an image.

    Args:
        image_path (str): Path to the image file.
        model (torch.nn.Module): Pre-trained model for prediction.
        top_k (int): Number of top classes to display. Default is 1.
        category_names (dict): Mapping of category numbers to category names.
        gpu (bool): Flag to use GPU for inference. Default is False.

    Returns:
        top_classes (list): List of top class IDs.
        top_probs (list): List of top class probabilities.
        top_class_names (list): List of top class names (if category_names is provided).
    """
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities, indices = output.topk(top_k)
        probabilities = probabilities.exp().cpu().numpy()[0]
        indices = indices.cpu().numpy()[0]

    if category_names:
        class_names = [category_names[str(idx)] for idx in indices]
    else:
        class_names = [str(idx) for idx in indices]

    return indices, probabilities, class_names

def main():
    parser = argparse.ArgumentParser(description="Flower Image Prediction")
    parser.add_argument("image_path", help="Path to the image file for prediction")
    parser.add_argument("checkpoint", help="Path to the model checkpoint file")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top classes to display")
    parser.add_argument("--category_names", default=None, help="Path to JSON file mapping categories to names")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    # Load category names if provided
    if args.category_names:
        category_names = load_category_names(args.category_names)
        # print("Loaded Category Names:", category_names)
    else:
        category_names = None

    # Load the model checkpoint and additional parameters
    model, class_to_idx, learning_rate, hidden_units, epochs, gpu = load_checkpoint(args.checkpoint)

    # Perform prediction
    top_classes, top_probs, top_class_names = predict(args.image_path, model, args.top_k, category_names, args.gpu)

    # Print the results
    for i in range(len(top_classes)):
        class_name = top_class_names[i] if category_names else top_classes[i]
        print(f"Class: {class_name}, Probability: {top_probs[i]:.4f}")

if __name__ == "__main__":
    main()