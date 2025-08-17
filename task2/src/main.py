import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pickle
import os
from tqdm import tqdm

PARAMS = {
    "epochs": 400,
    "batch_size": 32,
    "lr": 0.0002,
    "beta1": 0.5,
    "latent_dim": 100,
    "n_classes": 3,
    "img_width": 250,
    "img_height": 139,
    "channels": 3,
    "checkpoint_interval": 20
}


def load_and_prepare_data(data_paths):
    all_images, all_labels, label_map = [], [], {'a': 0, 'b': 1, 'c': 2}
    for path in data_paths:
        if not os.path.exists(path): continue
        print(f"Loading {path} file...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for part_key in data.keys():
            for layer_data in data[part_key]:
                images = layer_data['images']
                for key, img_np in images.items():
                    if img_np.ndim == 2: img_np = np.stack([img_np] * 3, axis=-1)
                    all_images.append(img_np)
                    all_labels.append(label_map[key[-1]])
    images_np = np.array(all_images, dtype=np.float32)
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2);
    images_tensor = (images_tensor / 127.5) - 1.0
    return TensorDataset(images_tensor, torch.LongTensor(all_labels))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__();
        self.label_embedding = nn.Embedding(PARAMS['n_classes'], PARAMS['latent_dim']);
        self.init_size_h, self.init_size_w = 8, 16;
        self.l1 = nn.Sequential(nn.Linear(PARAMS['latent_dim'] * 2, 128 * self.init_size_h * self.init_size_w));
        self.conv_blocks = nn.Sequential(nn.BatchNorm2d(128), nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, 1, 1),
                                         nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
                                         nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1),
                                         nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
                                         nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1),
                                         nn.BatchNorm2d(32, 0.8), nn.LeakyReLU(0.2, inplace=True),
                                         nn.Upsample(scale_factor=2), nn.Conv2d(32, PARAMS['channels'], 3, 1, 1),
                                         nn.Tanh())

    def forward(self, noise, labels):
        label_input = self.label_embedding(labels);
        gen_input = torch.mul(label_input, noise);
        combined_input = torch.cat((gen_input, noise), dim=1);
        out = self.l1(combined_input);
        out = out.view(out.shape[0], 128, self.init_size_h, self.init_size_w);
        img = self.conv_blocks(out);
        return torch.nn.functional.interpolate(img, size=(PARAMS['img_height'], PARAMS['img_width']), mode='bilinear',
                                               align_corners=False)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(PARAMS['n_classes'],
                                            PARAMS['channels'] * PARAMS['img_height'] * PARAMS['img_width'])

        self.model = nn.Sequential(
            nn.Conv2d(PARAMS['channels'] * 2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, kernel_size=(4, 7), stride=1, padding=0, bias=False),
        )

    def forward(self, img, labels):
        label_input_flat = self.label_embedding(labels)
        label_input = label_input_flat.view(-1, PARAMS['channels'], PARAMS['img_height'], PARAMS['img_width'])
        d_in = torch.cat((img, label_input), 1)
        return self.model(d_in).view(-1, 1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_paths = ["datasets/labeled_training_set.pkl", "datasets/unlabeled_training_set.pkl"]
    dataset = load_and_prepare_data(data_paths)
    dataloader = DataLoader(dataset, batch_size=PARAMS['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    adversarial_loss = nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=PARAMS['lr'], betas=(PARAMS['beta1'], 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=PARAMS['lr'], betas=(PARAMS['beta1'], 0.999))

    scheduler_G = StepLR(optimizer_G, step_size=100, gamma=0.5)
    scheduler_D = StepLR(optimizer_D, step_size=100, gamma=0.5)

    CHECKPOINT_OUTPUT_DIR = "/app/checkpoints"
    os.makedirs(CHECKPOINT_OUTPUT_DIR, exist_ok=True)

    for epoch in range(PARAMS['epochs']):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (imgs, labels) in progress_bar:
            real_label = torch.full((imgs.size(0), 1), 1.0, device=device, requires_grad=False)
            fake_label = torch.full((imgs.size(0), 1), 0.0, device=device, requires_grad=False)
            real_imgs, labels = imgs.to(device), labels.to(device)

            optimizer_D.zero_grad()
            d_real_loss = adversarial_loss(discriminator(real_imgs, labels), real_label)
            z = torch.randn(imgs.size(0), PARAMS['latent_dim']).to(device)
            gen_labels = torch.randint(0, PARAMS['n_classes'], (imgs.size(0),)).to(device)
            gen_imgs = generator(z, gen_labels)
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake_label)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), real_label)
            g_loss.backward()
            optimizer_G.step()

            progress_bar.set_description(
                f"[Epoch {epoch + 1}/{PARAMS['epochs']}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        scheduler_G.step()
        scheduler_D.step()

        if (epoch + 1) % PARAMS['checkpoint_interval'] == 0:
            torch.save(generator.state_dict(), f"{CHECKPOINT_OUTPUT_DIR}/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"{CHECKPOINT_OUTPUT_DIR}/discriminator_epoch_{epoch + 1}.pth")

    print("Training complete!")