import torch
import torch.nn as nn
import pickle
import os
import shutil
import glob
from PIL import Image
from torch_fidelity import calculate_metrics
import lpips
from tqdm import tqdm
import numpy as np

CHECKPOINTS_DIR = "/app/checkpoints"
REAL_DATA_PATHS = ["/app/datasets/labeled_training_set.pkl", "/app/datasets/unlabeled_training_set.pkl"]
FINAL_OUTPUT_DIR = "/app/submission"
LPIPS_SAMPLE_PAIRS = 1000

PARAMS = {
    "latent_dim": 100, "n_classes": 3, "img_width": 250,
    "img_height": 139, "channels": 3
}

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(PARAMS['n_classes'], PARAMS['latent_dim'])
        self.init_size_h, self.init_size_w = 8, 16
        self.l1 = nn.Sequential(nn.Linear(PARAMS['latent_dim'] * 2, 128 * self.init_size_h * self.init_size_w))
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32, 0.8), nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, PARAMS['channels'], 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_input = self.label_embedding(labels)
        gen_input = torch.mul(label_input, noise)
        combined_input = torch.cat((gen_input, noise), dim=1)
        out = self.l1(combined_input)
        out = out.view(out.shape[0], 128, self.init_size_h, self.init_size_w)
        img = self.conv_blocks(out)
        return torch.nn.functional.interpolate(img, size=(PARAMS['img_height'], PARAMS['img_width']), mode='bilinear',
                                               align_corners=False)

def load_real_images(data_paths, condition, cache):
    if condition in cache: return cache[condition]
    images = []
    for path in data_paths:
        if not os.path.exists(path): continue
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for part_key in data.keys():
            for layer_data in data[part_key]:
                key = 'A' + layer_data['layer_id'] + condition
                if key in layer_data['images']:
                    img = layer_data['images'][key]
                    if img.ndim == 2: img = np.stack([img] * 3, axis=-1)
                    images.append(img)
    cache[condition] = np.array(images)
    return cache[condition]

def generate_images(generator, device):
    images = {}
    inv_map = {0: 'a', 1: 'b', 2: 'c'}
    generator.eval()
    with torch.no_grad():
        for i in range(PARAMS['n_classes']):
            z = torch.randn(100, PARAMS['latent_dim'], device=device)
            labels = torch.LongTensor([i] * 100).to(device)
            gen_imgs = generator(z, labels)
            gen_imgs = (gen_imgs * 0.5 + 0.5) * 255.0
            gen_imgs_np = np.transpose(gen_imgs.cpu().numpy().astype(np.uint8), (0, 2, 3, 1))
            images[inv_map[i]] = gen_imgs_np
    return images

def save_images_to_dir(images, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i, img in enumerate(images): Image.fromarray(img).save(os.path.join(dir_path, f"{i:04d}.png"))

def evaluate_performance(real_images, gen_images, lpips_model, device):
    TEMP_REAL_DIR, TEMP_GEN_DIR = "temp_real_eval", "temp_gen_eval"
    save_images_to_dir(real_images, TEMP_REAL_DIR)
    save_images_to_dir(gen_images, TEMP_GEN_DIR)

    fid = \
    calculate_metrics(input1=TEMP_REAL_DIR, input2=TEMP_GEN_DIR, cuda=torch.cuda.is_available(), isc=False, fid=True,
                      kid=False, verbose=False)['frechet_inception_distance']

    gen_tensor = torch.from_numpy(gen_images).permute(0, 3, 1, 2).to(torch.float32) / 127.5 - 1.0
    total_lpips = 0
    indices = np.random.choice(len(gen_tensor), size=(LPIPS_SAMPLE_PAIRS, 2), replace=True)
    for i in range(LPIPS_SAMPLE_PAIRS):
        img1, img2 = gen_tensor[indices[i][0]:indices[i][0] + 1].to(device), gen_tensor[
            indices[i][1]:indices[i][1] + 1].to(device)
        total_lpips += lpips_model(img1, img2).item()
    lpips_score = total_lpips / LPIPS_SAMPLE_PAIRS

    shutil.rmtree(TEMP_REAL_DIR)
    shutil.rmtree(TEMP_GEN_DIR)
    return fid, lpips_score

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_paths = sorted(glob.glob(os.path.join(CHECKPOINTS_DIR, "generator_epoch_*.pth")))
    if not checkpoint_paths:
        print(f"Error: Could not find checkpoint files in '{CHECKPOINTS_DIR}' folder.")
        exit()

    print(f"\n--- Step 1: Finding the best model from a total of {len(checkpoint_paths)} checkpoints ---")
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    real_images_cache = {}
    all_results = []

    for cp_path in tqdm(checkpoint_paths, desc="Evaluating all checkpoints"):
        epoch = int(cp_path.split('_')[-1].split('.')[0])
        generator = Generator().to(device)
        generator.load_state_dict(torch.load(cp_path, map_location=device))
        generated_images = generate_images(generator, device)

        avg_fid = 0
        for cond in ['a', 'b', 'c']:
            real_imgs = load_real_images(REAL_DATA_PATHS, cond, real_images_cache)
            fid, _ = evaluate_performance(real_imgs, generated_images[cond], lpips_model, device)
            avg_fid += fid

        all_results.append({"epoch": epoch, "avg_fid": avg_fid / 3.0, "path": cp_path})

    best_model_info = min(all_results, key=lambda x: x['avg_fid'])
    print(f"\n🎉 Best model found: Epoch {best_model_info['epoch']} (Average FID: {best_model_info['avg_fid']:.4f})")

    print(f"\n--- Step 2: Generating final pkl files with the model from Epoch {best_model_info['epoch']} ---")
    best_generator = Generator().to(device)
    best_generator.load_state_dict(torch.load(best_model_info['path'], map_location=device))
    final_generated_images = generate_images(best_generator, device)

    os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)
    for cond, imgs in final_generated_images.items():
        filename = os.path.join(FINAL_OUTPUT_DIR, f"NIST_Task2_{cond}.pkl")
        with open(filename, 'wb') as f: pickle.dump(imgs, f)
        print(f"Saved '{filename}'.")

    print(f"\n--- Step 3: Detailed evaluation results for the final generated files ---")
    final_eval_results = []
    for cond in ['a', 'b', 'c']:
        real_imgs = load_real_images(REAL_DATA_PATHS, cond, real_images_cache)
        gen_imgs = final_generated_images[cond]
        fid, lpips_score = evaluate_performance(real_imgs, gen_imgs, lpips_model, device)
        final_eval_results.append({"condition": cond, "fid": fid, "lpips": lpips_score})

    print("=" * 60)
    print(f"{'Condition':<15} | {'Fidelity (FID)':<20} | {'Diversity (LPIPS)':<20}")
    print("-" * 60)
    for result in final_eval_results:
        print(f"{result['condition']:<15} | {result['fid']:<20.4f} | {result['lpips']:<20.4f}")
    print("=" * 60)