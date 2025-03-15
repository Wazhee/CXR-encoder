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

def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file, label = img_file
    img = Image.open(file)
    img = img.convert("RGB")
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out, label


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = [(i, file, label) for i, (file, label) in enumerate(dataset)]
    total = 0

    idx = 0
    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, label in tqdm(pool.imap_unordered(resize_fn, files)):
            # skip uncertain
            if label[0] == 1 and label[1] == 1:
              continue

            i = idx
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
                label_key = f"{size}-{str(i).zfill(5)}-label".encode("utf-8")

                with env.begin(write=True) as txn:
                  txn.put(key, img)
                  txn.put(label_key, label.astype(np.uint8))
            idx +=1
            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))

def main():
    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map["lanczos"]

    sizes = [256]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    dataset_path = "./1"
    label_file_path = "./1/train.csv"
    imgset = CheXPert(dataset_path, label_file_path, orientation="frontal")

    print("Images to process: %d" % len(imgset))

    with lmdb.open("./cxpt_mdb/", map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, n_worker= 1, sizes=sizes, resample=resample)
        
if __name__ == "__main__":
    main()

