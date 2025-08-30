import torch
from torchsummary import summary

from nebula.commons import Logger
from nebula.models.cnn import CNN
from nebula.models.enn import ENN

def cnn_nebula_galaxy(input_size=(100, 100)):
    """CNN model for 3-class galaxy classification (elliptical, spiral, irregular)."""
    model = CNN(num_channels=3, num_classes=3, input_size=input_size)
    return model


def enn_nebula_galaxy(input_size=(100, 100), N=4, dihedral=True):
    """D4-equivariant model for 3-class galaxy classification."""
    model = ENN(num_channels=3, num_classes=3, N=N, dihedral=dihedral, input_size=input_size)
    return model

if __name__ == "__main__":
    logger = Logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    input_size = (224, 224)

    logger.info("Testing CNN model for nebula galaxy classification:")
    cnn_model = cnn_nebula_galaxy(input_size=input_size).to(device)
    summary(cnn_model, (3, *input_size), device=str(device))

    logger.info("\nTesting ENN model for nebula galaxy classification:")
    enn_model = enn_nebula_galaxy(input_size=input_size).to(device)
    summary(enn_model, (3, *input_size), device=str(device))
