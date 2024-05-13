import torch
import torch.nn as nn
import timm
import os
from torch.utils.data import DataLoader

class EnsembleModel(nn.Module):
    def __init__(self, checkpoint_dir):
        super(EnsembleModel, self).__init__()
        self.models = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load each checkpoint
        for checkpoint_file in os.listdir(checkpoint_dir):
            if checkpoint_file.endswith('.pt'):
                model_path = os.path.join(checkpoint_dir, checkpoint_file)
                model = self.init_model()
                # model.load_state_dict(torch.load(model_path, map_location='cpu'))
                checkpoint = torch.load(model_path, map_location='cpu')
                # Load state dict specifically
                model.load_state_dict(checkpoint['model_state_dict'])
                self.models.append(model)
        print(f'Loaded {len(self.models)} models into the ensemble')

    def init_model(self):
        model_name = "tf_efficientnetv2_b0.in1k"
        model = timm.create_model(model_name, pretrained=False).eval()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 3)
        return model

    def forward(self, x):
        softmax_scores = []

        for model in self.models:
            model = model.to(self.device)
            with torch.no_grad():
                output = model(x)
                softmax = nn.Softmax(dim=1)(output)
                softmax_scores.append(softmax.cpu())

            model = model.cpu()  # Free up GPU memory

        # Calculate the mean softmax score across all models
        mean_softmax_score = torch.mean(torch.stack(softmax_scores), dim=0).to(self.device)
        return mean_softmax_score