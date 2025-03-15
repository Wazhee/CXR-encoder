import argparse
import kagglehub
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument('-data', default='cxpt', choices=['cxpt', 'nih', 'rsna', 'mimc'])
args = parser.parse_args()
dataset = args.data
valid = True

# os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if args.data == 'cxpt':
    # Download latest version
    path = kagglehub.dataset_download("ashery/chexpert")
elif args.data == 'nih':
    path = kagglehub.dataset_download("nih-chest-xrays/data")
elif args.data == 'rsna':
    path = kagglehub.dataset_download("jasonuwaeze/image2stylegan-latents-30k-chest-x-ray-dataset")
# elif args.data == 'mimic':
#     # comming soon
else:
    valid = False
    print(f"{args.data} invalid entry or is currently unavailable")

if valid:
    # Define destination directory (current directory)
    destination = os.getcwd()

    # Move downloaded dataset to the current directory
    shutil.move(path, os.path.join(destination, os.path.basename(path)))

    print("Dataset moved to:", destination)