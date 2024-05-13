import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from utils.ensemble import *
import pandas as pd
import re
import cv2
import numpy as np
import torch
from torchvision import transforms
import timm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import shutil
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import argparse


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Process and classify images using neural network models.')
    parser.add_argument('--source-folders', type=str, required=True, help='Directory containing the image folders and data files')
    parser.add_argument('--output-folder', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--fast-model-path', type=str, default='path_to_fast_model.pt', help='Path to the fast model')
    parser.add_argument('--ensemble-path', type=str, default='path_to_ensemble_model', help='Path to the ensemble model')
    parser.add_argument('--use-ensemble', action='store_true', help='Flag to use the ensemble model instead of the fast model')
    parser.add_argument('--sort-images', action='store_true', help='Flag to sort images when using the fast model')
    parser.add_argument('--coverage', type=float, default=0.93, help='Coverage percentage for classification confidence')
    return parser.parse_args()


# Label mapping
label_names = {
    0: "OilDroplets",
    1: "InactiveSCs",
    2: "ActiveSC"
}

class CustomImageDataset(Dataset):
    def __init__(self, source_folder, transform=None):
        self.source_folder = source_folder
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.samples, self.sample_ids = self._load_samples()

    def _load_samples(self):
        pattern = re.compile(r'^(\d+)_Ch(\d+)\.ome\.tif$')
        files = os.listdir(self.source_folder)
        grouped = {}
        sample_ids = []

        for file in files:
            match = pattern.match(file)
            if match:
                prefix, channel = match.groups()
                if prefix not in grouped:
                    grouped[prefix] = []
                    sample_ids.append(prefix)
                grouped[prefix].append(file)

        samples = [grouped[key] for key in grouped if len(grouped[key]) == 3]
        return samples, sample_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        files = self.samples[idx]
        images = []

        for file in files:
            image_path = os.path.join(self.source_folder, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (224, 224))  # Resize to match model input
            images.append(image)

        # Stack along the third dimension to create a single 3-channel image
        merged_image = np.stack(images, axis=-1)

        # Convert numpy array to PIL Image
        merged_image = Image.fromarray(merged_image)

        if self.transform:
            merged_image = self.transform(merged_image)

        return merged_image, self.sample_ids[idx]


def load_model(model_path, device):
    model_name = 'tf_efficientnetv2_b0.in1k'
    model = timm.create_model(model_name, pretrained=True).eval().to(device)
    num_ftrs = model.classifier.in_features
    model.classifier = torch.nn.Linear(num_ftrs, 3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def classify_and_move(model, dataset, device, new_path):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    class_count = defaultdict(int)

    for images, sample_ids in tqdm(dataloader, desc="Classifying and moving images"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        for pred, sample_id in zip(preds.cpu().numpy(), sample_ids):
            label_str = label_names[pred]

            target_folder = os.path.join(new_path, label_str)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            # Copy files based on the sample_id and predefined channels
            for channel in ['2', '3', '5']:
                src_filename = f"{sample_id}_Ch{channel}.ome.tif"
                src = os.path.join(dataset.source_folder, src_filename)
                dst = os.path.join(target_folder, src_filename)
                shutil.copy(src, dst)

            class_count[label_str] += 1  # Use the string label for counting

    print(f"Class distribution: {dict(class_count)}")


def classify_and_process(model, dataset, device, output_path, dataframe, sort_images=True):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    class_counts = defaultdict(int)
    sample_predictions = defaultdict(list)

    for images, sample_ids in tqdm(dataloader, desc="Classifying and processing"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

        for pred, sample_id in zip(preds.cpu().numpy(), sample_ids):
            label_str = label_names[pred]
            class_counts[label_str] += 1
            sample_predictions[label_str].append(sample_id)

            if sort_images:
                target_folder = os.path.join(output_path, 'sorted', label_str)
                os.makedirs(target_folder, exist_ok=True)

                # Copy files based on the sample_id
                for channel in ['2', '3', '5']:
                    src_filename = f"{sample_id}_Ch{channel}.ome.tif"
                    src = os.path.join(dataset.source_folder, src_filename)
                    dst = os.path.join(target_folder, src_filename)
                    shutil.copy(src, dst)

    total_samples = sum(class_counts.values())

    for class_name, sample_ids in sample_predictions.items():
        filtered_df = dataframe[dataframe['Object Number'].astype(str).isin(sample_ids)]
        filtered_df.to_csv(os.path.join(output_path, f"{class_name}.csv"), index=False)

        possible_diameter_names = ['Diameter_M05', 'Diameter Brightfiled']
        diameter_M05 = None
        diameter_M05_mean = None
        diameter_M05_median = None
        diameter_M05_std = None
        columns_and_files = [('Intensity_MC_Ch02', f'{class_name}_Ch2_Intensity.png'), ]
        # Iterate through possible column names and try to calculate median, mean & std
        for column_name in possible_diameter_names:
            try:
                diameter_M05_mean = filtered_df[column_name].mean()
                diameter_M05_median = filtered_df[column_name].median()
                diameter_M05_std = filtered_df[column_name].std()
                diameter_M05 = column_name
                columns_and_files.append((diameter_M05, f'{class_name}_Diameter.png'))
                break
            except KeyError:
                pass

        if not filtered_df.empty:
            summary_stats = {
                'Mean Diameter': diameter_M05_mean,
                'Median Diameter': diameter_M05_median,
                'Std Diameter': diameter_M05_std,
                'Mean Intensity': filtered_df['Intensity_MC_Ch02'].mean(),
                'Median Intensity': filtered_df['Intensity_MC_Ch02'].median(),
                'Std Intensity': filtered_df['Intensity_MC_Ch02'].std(),
                'Count': filtered_df.shape[0],
                'Percentage': (filtered_df.shape[0] / total_samples) * 100
            }
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(os.path.join(output_path, f"{class_name}Summary.csv"), index=False)

            for column, filename in columns_and_files:
                plt.figure()
                filtered_df[column].hist()
                plt.title(f'{class_name} {column}')
                plt.savefig(os.path.join(output_path, filename))
                plt.close()
        else:
            print(f"No samples found for {class_name}")

    print(f"Class distribution: {dict(class_counts)}")


def selective_classify_and_process(model, dataset, device, output_path, dataframe, sort_images=True, coverage=1.0):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions = []

    # First pass: collect predictions and confidence scores
    for images, sample_ids in tqdm(dataloader, desc="Collecting predictions and confidences"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            softmax_scores = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidence_scores, _ = torch.max(softmax_scores, dim=1)

        predictions.extend(zip(sample_ids, preds.cpu().numpy(), confidence_scores.cpu().numpy()))

    # Determine confidence threshold
    sorted_confidences = sorted(predictions, key=lambda x: x[2], reverse=True)
    cutoff_index = int(len(sorted_confidences) * coverage)
    confident_predictions = sorted_confidences[:cutoff_index]
    uncertain_predictions = sorted_confidences[cutoff_index:]

    # Extract confident sample IDs for filtering
    confident_sample_ids = {sample_id for sample_id, _, _ in confident_predictions}

    # Counters for uncertain class predictions
    uncertain_class_counts = defaultdict(int)

    # Sort images if requested and count uncertain predictions
    if sort_images:
        for sample_id, pred, _ in confident_predictions:
            label_str = label_names[pred]
            target_folder = os.path.join(output_path, 'sorted', label_str)
            os.makedirs(target_folder, exist_ok=True)
            copy_images_to_folder(dataset.source_folder, target_folder, sample_id)

        uncertain_folder = os.path.join(output_path, 'sorted', 'uncertain')
        os.makedirs(uncertain_folder, exist_ok=True)
        for sample_id, pred, _ in uncertain_predictions:
            copy_images_to_folder(dataset.source_folder, uncertain_folder, sample_id)
            uncertain_class_counts[label_names[pred]] += 1

    # Process confident predictions for CSV and histograms
    class_counts = process_predictions_for_csv_and_histograms(confident_predictions, dataframe, output_path, confident_sample_ids, label_names)

    # Print class distribution for confident samples
    print("Class distribution for confident samples:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")
    print(f"Uncertain: {len(uncertain_predictions)}")

    # Print predicted class distribution for uncertain samples
    print("Predicted class distribution for uncertain samples:")
    for class_name, count in uncertain_class_counts.items():
        print(f"{class_name} (would be): {count}")

def copy_images_to_folder(source_folder, target_folder, sample_id):
    for channel in ['2', '3', '5']:
        src_filename = f"{sample_id}_Ch{channel}.ome.tif"
        src = os.path.join(source_folder, src_filename)
        dst = os.path.join(target_folder, src_filename)
        shutil.copy(src, dst)

def process_predictions_for_csv_and_histograms(predictions, dataframe, output_path, confident_sample_ids, label_names):
    class_counts = defaultdict(int)
    sample_predictions = defaultdict(list)

    # Filter the dataframe to only include rows with sample IDs in confident_sample_ids
    dataframe = dataframe[dataframe['Object Number'].astype(str).isin(confident_sample_ids)]

    for sample_id, pred, _ in predictions:
        label_str = label_names[pred]
        class_counts[label_str] += 1
        sample_predictions[label_str].append(sample_id)

    total_confident_samples = sum(class_counts.values())
    # If we want to subtract OilDroplets and include only ActiveSCs and InactiveSCs in the total:
    total_confident_samples -= class_counts[0]  # Subtract the count for OilDroplets

    for class_name, sample_ids in sample_predictions.items():
        filtered_df = dataframe[dataframe['Object Number'].astype(str).isin(sample_ids)]
        filtered_df.to_csv(os.path.join(output_path, f"{class_name}.csv"), index=False)

        if not filtered_df.empty:
            generate_summary_and_histograms(filtered_df, class_name, output_path, total_confident_samples)

    return class_counts

def generate_summary_and_histograms(filtered_df, class_name, output_path, total_confident_samples):
    possible_diameter_names = ['Diameter_M05', 'Diameter Brightfiled']
    diameter_M05 = None
    diameter_M05_mean = None
    diameter_M05_median = None
    diameter_M05_std = None
    columns_and_files = [('Intensity_MC_Ch02', f'{class_name}_Ch2_Intensity.png'), ]
    # Iterate through possible column names and try to calculate median, mean & std
    for column_name in possible_diameter_names:
        try:
            diameter_M05_mean = filtered_df[column_name].mean()
            diameter_M05_median = filtered_df[column_name].median()
            diameter_M05_std = filtered_df[column_name].std()
            diameter_M05 = column_name
            columns_and_files.append((diameter_M05, f'{class_name}_Diameter.png'))
            break
        except KeyError:
            pass
    summary_stats = {
        'Mean Diameter': diameter_M05_mean,
        'Median Diameter': diameter_M05_median,
        'Std Diameter': diameter_M05_std,
        'Mean Intensity': filtered_df['Intensity_MC_Ch02'].mean(),
        'Median Intensity': filtered_df['Intensity_MC_Ch02'].median(),
        'Std Intensity': filtered_df['Intensity_MC_Ch02'].std(),
        'Count': filtered_df.shape[0],
        'Percentage': (filtered_df.shape[0] / total_confident_samples) * 100
    }
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(os.path.join(output_path, f"{class_name}Summary.csv"), index=False)

    # Generate and save histograms
    for column, filename in columns_and_files:
        plt.figure()
        filtered_df[column].hist()
        plt.title(f'{class_name} {column}')
        plt.savefig(os.path.join(output_path, filename))
        plt.close()


def classify_and_move_directory(source_folder, output_folder, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_ensemble:
        model = EnsembleModel(args.ensemble_path)
    else:
        fast_model_path = args.fast_model_path
        fast_model = load_model(fast_model_path, device)
        fast_output_folder = f'{output_folder}\\fast'
        os.makedirs(fast_output_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    txt_file_name = None
    subdirectory_name = None
    # List all entries in the directory
    entries = os.listdir(source_folder)
    # Iterate over the entries to find the text file and the directory
    for entry in entries:
        full_path = os.path.join(source_folder, entry)
        if os.path.isfile(full_path) and entry.endswith('.txt'):
            txt_file_name = entry  # Found the text file
        elif os.path.isdir(full_path):
            subdirectory_name = entry  # Found the directory

    # Now you have the names of the text file and the directory
    print("Automatically determining the data.txt and data folder within source directory:")
    print(f"Data text file name: {txt_file_name}")
    print(f"Data directory (images) name: {subdirectory_name}")

    # You can then use the text file name to load it into a pandas DataFrame
    file_path = os.path.join(source_folder, txt_file_name)
    df = pd.read_csv(file_path, sep='\t', skiprows=1)

    dataset = CustomImageDataset(os.path.join(source_folder, subdirectory_name), transform=transform)

    if args.use_ensemble:
        selective_classify_and_process(model, dataset, device, output_folder, df, sort_images=args.sort_images, coverage=args.coverage)
    else:
        selective_classify_and_process(fast_model, dataset, device, fast_output_folder, df,
                                       sort_images=args.sort_images, coverage=args.coverage)


def main():
    args = parse_args()

    folders = os.listdir(args.source_folders)
    for folder in folders:
        source_folder = f'{args.source_folders}\\{folder}'
        target_folder = f'{args.output_folder}\\{folder}'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        classify_and_move_directory(source_folder=source_folder, output_folder=target_folder, args=args)


if __name__ == "__main__":
    main()