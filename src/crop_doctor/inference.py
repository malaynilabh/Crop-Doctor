from pathlib import Path

import torch

from .model import CropDoctorCNN


def load_model(model_path, device):
    """
    Load the trained Crop Doctor CNN checkpoint.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model checkpoint.
    device : torch.device
        Device on which the model should run.

    Returns
    -------
    model : CropDoctorCNN
        Loaded trained model.
    idx_to_class : dict
        Mapping from class index to class name.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = CropDoctorCNN(
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    idx_to_class = checkpoint["idx_to_class"]

    return model, idx_to_class


def predict_image(
    image,
    model,
    transform,
    idx_to_class,
    device,
    top_k=5
):
    """
    Predict the crop/disease classes for a single image.

    Parameters
    ----------
    image : PIL.Image
        Input image.
    model : CropDoctorCNN
        Trained CNN model.
    transform : torchvision.transforms.Compose
        Evaluation transform.
    idx_to_class : dict
        Mapping from class index to class name.
    device : torch.device
        Device on which inference should run.
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    results : list of dict
        Top-k predictions with probability scores.
    """

    model.eval()

    # Apply preprocessing
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move image to device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        # Model outputs raw logits
        logits = model(image_tensor)

        # Convert logits into probabilities
        probabilities = torch.softmax(
            logits,
            dim=1
        )

    # Get top-k predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
        dim=1
    )

    results = []

    for probability, index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        results.append(
            {
                "class": idx_to_class[index.item()],
                "probability": probability.item()
            }
        )

    return results