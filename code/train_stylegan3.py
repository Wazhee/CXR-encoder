import os
import click
import torch
import torch.nn as nn
import torch.optim as optim
from model import Encoder
from networks_stylegan3 import Generator
import dnnlib
import legacy
import re
import sys
from train import train_encoder

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# class CSVImageDataset(Dataset):
#     def __init__(self, csv_path, root_dir, transform=None):
#         self.data_frame = pd.read_csv(csv_path)
#         self.root_dir = root_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_frame)

#     def __getitem__(self, idx):
#         # Recover image path from "Path" column
#         img_path = "../" + self.data_frame["Path"].iloc[idx]
#         image = Image.open(img_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image




# Define StyleGAN3 Encoder-Decoder Framework
class StyleGAN3EncoderDecoder(nn.Module):
    def __init__(self, encoder, generator):
        super(StyleGAN3EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.generator = generator

    def forward(self, x):
        latent_code = self.encoder(x)
        latent_code = latent_code.unsqueeze(1).repeat(1, self.generator.num_ws, 1)  # Shape: [batch_size, num_ws, w_dim]
        generated_image = self.generator.synthesis(latent_code)
        return generated_image

@click.command()
@click.argument('data', type=str)
@click.option('--arch', type=str, default='autoencoder', help='Architecture type (default: autoencoder)')
@click.option('--batch', type=int, default=32, help='Batch size for training')
@click.option('--output_path', type=str, required=True, help='Path to save training results')
@click.option('--classifier_ckpt', type=str, help='Path to classifier checkpoint')
@click.option('--filter_label', type=str, help='Filter label for specific conditions')
@click.option('--compare_to_healthy', is_flag=True, help='Compare generated images to healthy samples')
@click.option('--ckpt', type=str, help='Path to pre-trained checkpoint')

def train(data, arch, batch, output_path, classifier_ckpt, filter_label, compare_to_healthy, ckpt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize Encoder and Generator
    encoder = Encoder(size=512).to(device)
#     generator = Generator(z_dim=512, # Input latent (Z) dimensionality.
#                           c_dim=16,  # Conditioning label (C) dimensionality.
#                           w_dim=512,   # Intermediate latent (W) dimensionality.
#                           img_resolution=256, # Output resolution.
#                           img_channels=3, # Number of output color channels.
#                           mapping_kwargs={}, # Arguments for MappingNetwork.
#                          ).to(device)  
    network_pkl = "../../stylegan3/results/00022-stylegan3-t-nih_chexpert-gpus4-batch16-gamma1/network-snapshot-004000.pkl"
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        generator = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    # Load pre-trained checkpoint if provided
    if ckpt:
        checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
#          = torch.load(ckpt, map_location=device)
        encoder.load_state_dict(checkpoint['e'])
#         generator.load_state_dict(checkpoint['g'])
        print(f'Loaded checkpoint from {ckpt}')

    # Initialize the model
    model = StyleGAN3EncoderDecoder(encoder, generator).to(device)
      # Freeze generator parameters
    for param in model.generator.parameters():
        param.requires_grad = False
        
    # Check for available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Move the model to the available device
    model.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    
    # Prepare DataLoader
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

#     loader = get_dataloader(args)
    dataset = datasets.ImageFolder(root="../../chexpert/versions/1/train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=8)
    
#     # Use the custom dataset
#     csv_path = "../../chexpert/versions/1/train.csv"
#     root_dir = "../../chexpert/versions/1/train"
#     dataset = CSVImageDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)

#     # Load data using DataLoader
#     batch_size = 32  # Adjust for memory efficiency
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
#     print("GOT HERE ")
    
#     accum = 0.5 ** (32 / (10 * 1000))
#     n_iter = 800000
#     start_iter = 48000
#     output_path = "../results"

#     sample_z = torch.randn(16, 512, device=device)
#     pbar = range(n_iter)
#     for idx in pbar:
#         i = idx + start_iter

#         if i > n_iter:
#             print("Done!")

#             break

#         real_img, real_labels = next(loader)
#         real_img, real_labels = real_img.to(device), real_labels.to(device)

#         real_encoded = encoder(real_img)
#         real_logits = classifier(real_img)

#         _, encoded_labels = torch.max(real_logits, 1)
#         encoded_labels = F.one_hot(encoded_labels, num_classes=args.classifier_nof_classes)
#         fake_img, _ = generator([real_encoded], labels=encoded_labels, input_is_latent=True)
#         fake_logits = classifier(fake_img)
        
#         logsoft = torch.nn.LogSoftmax(dim = 1)
#         class_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(logsoft(fake_logits), logsoft(real_logits))

#         loss_dict["class_loss"] = class_loss

#         fake_encoded = encoder(fake_img)
#         reconstruct_loss_x = F.l1_loss(fake_img, real_img)
#         reconstruct_loss_w = F.l1_loss(fake_encoded, real_encoded)
#         reconstruct_loss_lpips = lpips(fake_img/fake_img.max(), real_img/real_img.max(), net_type='alex', version='0.1').flatten()

#         reconstruct_loss = reconstruct_loss_x + reconstruct_loss_w + reconstruct_loss_lpips
#         loss_dict["reconstruct_loss"] = reconstruct_loss
#         loss_dict["reconstruct_loss_x"] = reconstruct_loss_x
#         loss_dict["reconstruct_loss_w"] = reconstruct_loss_w
#         loss_dict["reconstruct_loss_lpips"] = reconstruct_loss_lpips

#         generator.zero_grad()
#         encoder.zero_grad()
#         total_loss = class_loss + reconstruct_loss
#         loss_dict["total_loss"] = total_loss
#         total_loss.backward()
#         e_optim.step()
#         g_optim.step()
#         accumulate(g_ema, g_module, accum)
 
#         class_loss_val = loss_dict["class_loss"].mean().item()
#         reconstruct_loss_val = loss_dict["reconstruct_loss"].mean().item()
#         reconstruct_loss_x_val = loss_dict["reconstruct_loss_x"].mean().item()
#         reconstruct_loss_w_val = loss_dict["reconstruct_loss_w"].mean().item()
#         reconstruct_loss_lpips_val = loss_dict["reconstruct_loss_lpips"].mean().item()
#         total_loss_val = loss_dict["total_loss"].mean().item()

#         if get_rank() == 0:
#             print(  f"class: {class_loss_val:.6f}; "
#                     f"reconstruct: {reconstruct_loss_val:.6f} "
#                     f"reconstruct_x: {reconstruct_loss_x_val:.6f} "
#                     f"reconstruct_w: {reconstruct_loss_w_val:.6f} "
#                     f"reconstruct_lpips: {reconstruct_loss_lpips_val:.6f} "
#                     f"total_loss: {total_loss_val:.6f} "
#                     )

#             if wandb and args.wandb:
#                 wandb.log(
#                     {
#                         "Class Loss": class_loss_val,
#                         "Reconstruct Loss": reconstruct_loss_val,
#                         "Reconstruct x": reconstruct_loss_x_val,
#                         "Reconstruct w": reconstruct_loss_w_val,
#                         "Reconstruct LPIPS": reconstruct_loss_lpips_val,
#                         "Total Loss": total_loss_val
#                     }
#                 )
# #                 wandb.watch(generator)
# #                 wandb.watch(encoder)

#             if i == 0:
#                 continue
            
#             if i % args.save_samples_every == 0:
#                 with torch.no_grad():
#                     encoded_image = get_image(
#                         fake_img,
#                         nrow=int(args.batch ** 0.5),
#                         normalize=True,
#                         scale_each=True
#                     )
#                     real_image = get_image(
#                         real_img,
#                         nrow=int(args.batch ** 0.5),
#                         normalize=True,
#                         scale_each=True
#                     )
#                     filename_img = os.path.join(output_path, f"sample/img_{str(i).zfill(6)}.png")
#                     combined = concat_image_by_height(encoded_image, real_image)
#                     combined.save(filename_img)

#             if i % args.save_checkpoint_every == 0:
#                 filename = os.path.join(output_path, f"checkpoint/{str(i).zfill(6)}.pt")
#                 torch.save(
#                     {
#                         "g": g_module.state_dict(),
#                         "e": e_module.state_dict(),
#                         "g_ema": g_ema.state_dict(),
#                         "g_optim": g_optim.state_dict(),
#                         "e_optim": e_optim.state_dict(),
#                         "args": args,
#                     },
#                     filename,
#                 )
    from tqdm import tqdm
    # Training Loop
  
    model.train()
    
    
    for epoch in range(100):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/100], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                
                
#     scaler = torch.cuda.amp.GradScaler()
#     for epoch in tqdm(range(100)):
#         for i, (images, _) in enumerate(dataloader):
#             print(type(images), images.shape)
#             images = images.to(device)

#             with torch.cuda.amp.autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
        
#             if (i + 1) % 50 == 0:
#                 print(f'Epoch [{epoch + 1}/100], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Save Model
    os.makedirs(output_path, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'generator': generator.state_dict(),
    }, os.path.join(output_path, 'final_model.pt'))
    print('Training Complete')

if __name__ == '__main__':
    device = "cuda"
    
    train()
