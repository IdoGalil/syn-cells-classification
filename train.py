import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import timm

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on a custom dataset')
    parser.add_argument('--data-dir', default='./training_dataset', type=str, help='Directory with the training data')
    parser.add_argument('--checkpoint-dir', default='./model_checkpoints', type=str, help='Directory to save checkpoints')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train')
    parser.add_argument('--batch-size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    return parser.parse_args()


# Custom Dataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, indices=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = os.listdir(root_dir)

        for label, class_name in enumerate(tqdm(self.classes, desc="Loading classes")):
            class_folder = os.path.join(root_dir, class_name)
            filenames = [f for f in os.listdir(class_folder) if f.endswith('.npy')]
            self.images.extend([os.path.join(class_folder, f) for f in filenames])
            self.labels.extend([label] * len(filenames))

        if indices is not None:
            self.images = [self.images[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_synis_indices(self):
        return [i for i, img_path in enumerate(self.images) if 'SynIS' in img_path]

def make_weights_for_balanced_classes(dataset):
    label_counts = Counter(dataset.dataset.labels)
    N = sum(label_counts.values())
    weight_per_class = {label: N / count for label, count in label_counts.items()}
    weights = [weight_per_class[label] for (_, label) in dataset]
    return weights

# Transformations
size = (224, 224)
train_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training function
def train(epoch, loader, model, criterion, optimizer, device, log_to_wandb=False):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * len(loader.dataset.classes)
    class_total = [0] * len(loader.dataset.classes)
    pbar = tqdm(loader, desc=f'Training Epoch {epoch}')

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += (predicted[i] == labels[i]).item()
            class_total[label] += 1

        pbar.set_description(f'Training Epoch {epoch} - Loss: {total_loss/total:.4f}, Acc: {100 * correct / total:.2f}%')

    if log_to_wandb:
        import wandb
        wandb.log({"train_loss": total_loss / total, "train_accuracy": 100 * correct / total})

    return total_loss / total, 100 * correct / total

# Validation function
def validate(epoch, loader, model, criterion, device, log_to_wandb=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = [0] * len(loader.dataset.classes)
    class_total = [0] * len(loader.dataset.classes)
    pbar = tqdm(loader, desc=f'Validation Epoch {epoch}')

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1

            pbar.set_description(f'Validation Epoch {epoch} - Loss: {total_loss/total:.4f}, Acc: {100 * correct / total:.2f}%')

    if log_to_wandb:
        import wandb
        wandb.log({"val_loss": total_loss / total, "val_accuracy": 100 * correct / total})

    return total_loss / total, 100 * correct / total

def main():
    args = parse_args()

    dataset = CustomDataset(args.data_dir)

    synis_indices = dataset.get_synis_indices()
    non_synis_indices = [i for i in range(len(dataset)) if i not in synis_indices]
    train_size = int(0.9 * len(non_synis_indices))
    test_size = len(non_synis_indices) - train_size
    non_synis_train_indices, non_synis_test_indices = random_split(non_synis_indices, [train_size, test_size])

    train_indices = non_synis_train_indices + synis_indices
    test_indices = non_synis_test_indices

    train_dataset = CustomDataset(args.data_dir, transform=train_transformations, indices=train_indices)
    test_dataset = CustomDataset(args.data_dir, transform=test_transformations, indices=test_indices)

    weights = make_weights_for_balanced_classes(Subset(dataset, train_indices))
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model_name = "tf_efficientnetv2_b0.in1k"
    model = timm.create_model(model_name, pretrained=True).eval().cuda()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.wandb:
        import wandb
        run_name = model_name + '-' + str(len(dataset.classes)) + '-' + '-'.join(dataset.classes)
        wandb.init(project='SynCells', name=run_name)
        wandb.config.update(args)

    best_val_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(epoch, train_loader, model, criterion, optimizer, device, args.wandb)
        val_loss, val_accuracy = validate(epoch, test_loader, model, criterion, device, args.wandb)

        current_lr = scheduler.get_last_lr()[0]
        if args.wandb:
            wandb.log({"learning_rate": current_lr})

        scheduler.step()

        if epoch == args.epochs or val_accuracy > best_val_accuracy:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
            }
            if not os.path.exists(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir)
            if epoch == args.epochs:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}_checkpoint_last.pt')
                torch.save(checkpoint, checkpoint_path)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}_best_model.pt')
                torch.save(checkpoint, best_checkpoint_path)

        print(
            f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()