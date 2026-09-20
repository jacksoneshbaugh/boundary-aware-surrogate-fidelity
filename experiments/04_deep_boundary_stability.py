"""
Experiment 4: Stability of boundary-proximal regions across deep teachers.

Independently trained deep classifiers may have similar predictive
performance while inducing different boundary-proximal regions. This
experiment trains each teacher exactly once and caches its test predictions
and classification margins for later pairwise comparison.

Boundary proximity is defined within each teacher by the top-1 minus top-2
logit margin. Because absolute logit scales are not directly comparable
across independently trained networks, subsequent analysis uses fixed
fractions of the lowest-margin test examples rather than fixed thresholds.

Datasets / models
-----------------
MNIST:
    Small convolutional network trained with Adam.

CIFAR-10:
    ResNet-18 adapted for 32x32 images, trained using random cropping,
    horizontal flipping, SGD with momentum, weight decay, and cosine
    learning-rate annealing.

Outputs
-------
One compressed NPZ file per dataset x seed containing:
    - test labels
    - predicted classes
    - top-two logit margins
    - test accuracy
    - seed
    - epoch count

Pairwise stability metrics are computed separately by
scripts/summarize_deep_boundary_stability.py.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


DEFAULT_SEEDS = list(range(5))


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Favor reproducibility over autotuning.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_model(dataset):
    if dataset == "mnist":
        return MNISTCNN()

    if dataset == "cifar10":
        model = resnet18(
            weights=None,
            num_classes=10,
        )

        # CIFAR-sized input.
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        model.maxpool = nn.Identity()

        return model

    raise ValueError(
        f"Unknown dataset: {dataset}"
    )


def get_datasets(dataset, data_dir):
    if dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,),
                    (0.3081,),
                ),
            ]
        )

        trainset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )

        testset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

        return trainset, testset

    if dataset == "cifar10":
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(
                    32,
                    padding=4,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        trainset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )

        return trainset, testset

    raise ValueError(
        f"Unknown dataset: {dataset}"
    )


def make_loaders(
    trainset,
    testset,
    seed,
    batch_size,
    num_workers,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    pin_memory = torch.cuda.is_available()

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader


def train_model(
    model,
    trainloader,
    dataset,
    epochs,
    device,
):
    criterion = nn.CrossEntropyLoss()

    if dataset == "mnist":
        optimizer = optim.Adam(
            model.parameters(),
            lr=1e-3,
        )
        scheduler = None

    elif dataset == "cifar10":
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
        )

    else:
        raise ValueError(dataset)

    for epoch in range(epochs):
        model.train()

        correct = 0
        total = 0
        running_loss = 0.0

        for inputs, labels in trainloader:
            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predictions = logits.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        if scheduler is not None:
            scheduler.step()

        interval = (
            10
            if dataset == "mnist"
            else 20
        )

        if (
            (epoch + 1) % interval == 0
            or epoch + 1 == epochs
        ):
            print(
                f"  epoch {epoch + 1:3d}/{epochs} | "
                f"loss={running_loss / len(trainloader):.4f} | "
                f"train_acc={correct / total:.4f}"
            )


def evaluate(
    model,
    testloader,
    device,
):
    model.eval()

    all_predictions = []
    all_margins = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            logits = model(inputs)

            predictions = logits.argmax(
                dim=1
            )

            # The classifier changes class when the two largest
            # logits exchange order. Therefore the top-two logit
            # gap provides a natural output-space margin.
            top_two = torch.topk(
                logits,
                k=2,
                dim=1,
            ).values

            margins = (
                top_two[:, 0]
                - top_two[:, 1]
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_margins.append(
                margins.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    predictions = np.concatenate(
        all_predictions
    )

    margins = np.concatenate(
        all_margins
    )

    labels = np.concatenate(
        all_labels
    )

    accuracy = float(
        np.mean(predictions == labels)
    )

    return (
        predictions,
        margins,
        labels,
        accuracy,
    )


def run_seed(
    dataset,
    seed,
    epochs,
    data_dir,
    output_dir,
    batch_size,
    num_workers,
    device,
):
    print()
    print("=" * 70)
    print(
        f"{dataset.upper()} | seed={seed} | "
        f"epochs={epochs}"
    )
    print("=" * 70)

    seed_everything(seed)

    trainset, testset = get_datasets(
        dataset,
        data_dir,
    )

    trainloader, testloader = make_loaders(
        trainset,
        testset,
        seed,
        batch_size,
        num_workers,
    )

    model = make_model(dataset).to(device)

    train_model(
        model,
        trainloader,
        dataset,
        epochs,
        device,
    )

    (
        predictions,
        margins,
        labels,
        accuracy,
    ) = evaluate(
        model,
        testloader,
        device,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{dataset}_seed_{seed}.npz"
    )

    np.savez_compressed(
        output_path,
        dataset=dataset,
        seed=seed,
        epochs=epochs,
        labels=labels,
        predictions=predictions,
        margins=margins,
        accuracy=accuracy,
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
    )

    print(
        f"Saved {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10"],
        required=True,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/deep_boundary_stability/cache"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs is None:
        epochs = (
            30
            if args.dataset == "mnist"
            else 200
        )
    else:
        epochs = args.epochs

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    for seed in args.seeds:
        run_seed(
            dataset=args.dataset,
            seed=seed,
            epochs=epochs,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )


if __name__ == "__main__":
    main()