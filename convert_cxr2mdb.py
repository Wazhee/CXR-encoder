from io import BytesIO
import multiprocessing
from functools import partial
import os

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CheXPert(Dataset):
    def __init__(self, path_to_data, label_file_path, uncertainty=False, transform=None, orientation="all", filter_label=None):
        """ Dataset class for CheXPert

        Args:
            path_to_data (string): Path to dataset
            uncertainty (bool): Changes label calculation
            transform (torchvision.Transforms): transforms performed on the samples
            orientation (string): "all", "frontal", "lateral"
        """

        self.uncertainty = uncertainty
        self.transform = transform
        self.path_to_data = path_to_data
        self.orientation = orientation
        self.filter_label = filter_label

        self.PRED_LABEL = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices"]

        if filter_label is not None:
          self.filter_index = self.PRED_LABEL.index(filter_label)

        self.labels = pd.read_csv(label_file_path)
        
        # Deleting either lateral or frontal images of the Dataset or keep all
        if self.orientation == "lateral":
            self.labels = self.labels[~self.labels.Path.str.contains("frontal")]
        elif self.orientation == "frontal":
            self.labels = self.labels[~self.labels.Path.str.contains("lateral")]
        elif self.orientation == "all":
            pass
        else:
            raise Exception("Wrong orientation input given!")

    def __len__(self):
        """
        Returns:
            (int): length of the pandas dataframe
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Arguments:
        - idx (int) : Index of the image to return
        Returns:
        - image (PIL.Image): PIL format image
        """

        image_path = os.path.join(self.path_to_data, self.labels.iloc[idx]['Path'])
        #image = Image.open(image_path).convert('RGB')

        # Get labels from the dataframe for current image
        label = self.labels.iloc[idx, :].loc[self.PRED_LABEL]
        label = label.to_numpy()

        # Uncertainty labels are mapped to 0.0
        if not self.uncertainty:
            tmp = np.zeros(len(self.PRED_LABEL))
            tmp[label == 1] = 1.0
            label = tmp

        if self.filter_label:
          labels = labels[self.filter_index].reshape([1])

        return image_path, label