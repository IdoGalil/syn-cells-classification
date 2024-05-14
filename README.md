# Synthetic Cell Classification System

This repository contains the code for the paper "Scaling Up Synthetic Cells Production through Robotics and AI Towards Therapeutic Applications". It provides tools for classifying and training models on synthetic cell images captured with the ImageStream®X Mk II instrument, specifically designed to differentiate between the classes OilDroplets, InactiveSC, and ActiveSC.

## Quick Start

Clone the repository to get started:

```bash
git clone https://github.com/IdoGalil/syn-cells-classification
cd syn-cells-classification
pip install -r requirements.txt
```

## Classifying Images

To allow ensemble classification, download and prepare the model checkpoints (ensemble model checkpoints are available under "releases"):

```bash
unzip ensemble_p1.zip -d ./model_checkpoints/ensemble
unzip ensemble_p2.zip -d ./model_checkpoints/ensemble
```

To classify images, the input directory should contain sub-folders, each representing an image collection session. Each sub-folder must include a metadata `.txt` file and another sub-folder containing all the images in ImageStream®X Mk II format. Images should be named in the format `ID_ChX.ome.tif`, where `ID` is a unique identifier for the cell instance, and `X` is the channel number (Ch2 for GFP, Ch3 for Rhodamine, Ch5 for Brightfield).

### Usage

For faster classification using a single model:

```bash
python classify_and_move_folders.py --source-folders <path-to-input-folder> --output-folder <path-to-output-folder> --fast-model-path ./model_checkpoints/tf_efficientnetv2_b0.in1k.pt
```

For more accurate but slower classification using an ensemble model:

```bash
python classify_and_move_folders.py --source-folders <path-to-input-folder> --output-folder <path-to-output-folder> --ensemble-path ./model_checkpoints/ensemble --use-ensemble
```

### Advanced Options

Adjust the confidence threshold and enable image sorting into class-specific folders:

```bash
python classify_and_move_folders.py --source-folders <path-to-input-folder> --output-folder <path-to-output-folder> --use-ensemble --sort-images --coverage 0.93
```

## Training the Model

First, prepare the training dataset (the training dataset is available under "releases"):

```bash
unzip training_dataset.zip -d ./training_dataset
```

To train a model on the dataset:

```bash
python train.py --data-dir ./training_dataset --checkpoint-dir ./model_checkpoints
```
