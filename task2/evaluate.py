import torch
import pickle
import os
import shutil
import glob
from PIL import Image
from torch_fidelity import calculate_metrics
import lpips
from tqdm import tqdm
import numpy as np

# --- 설정 ---
SUBMISSION_DIR = "submission"  # 평가할 pkl 파일들이 있는 폴더
REAL_DATA_PATHS = ["datasets/labeled_training_set.pkl", "datasets/unlabeled_training_set.pkl"]
LPIPS_SAMPLE_PAIRS = 1000
# -----------------------------------------------------------------------------

# --- 유틸리티 함수들 ---
def load_real_images(data_paths, condition, cache):
    if condition in cache: return cache[condition]
    images = []
    print(f"실제 이미지 로딩 중 (조건: '{condition}')...")
    for path in data_paths:
        if not os.path.exists(path): continue
        with open(path, 'rb') as f: data = pickle.load(f)
        for part_key in data.keys():
            for layer_data in data[part_key]:
                key = 'A' + layer_data['layer_id'] + condition
                if key in layer_data['images']:
                    img = layer_data['images'][key]
                    if img.ndim == 2: img = np.stack([img]*3, axis=-1)
                    images.append(img)
    cache[condition] = np.array(images)
    return cache[condition]

def load_generated_images(file_path):
    print(f"생성된 이미지 로딩 중: {file_path}")
    if not os.path.exists(file_path):
        print(f"오류: {file_path}를 찾을 수 없습니다.")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_images_to_dir(images, dir_path):
    os.makedirs(dir_path, exist_ok=True)
    for i, img in enumerate(images): Image.fromarray(img).save(os.path.join(dir_path, f"{i:04d}.png"))

def evaluate_performance(real_images, gen_images, lpips_model, device):
    TEMP_REAL_DIR, TEMP_GEN_DIR = "temp_real_eval", "temp_gen_eval"
    save_images_to_dir(real_images, TEMP_REAL_DIR)
    save_images_to_dir(gen_images, TEMP_GEN_DIR)
    
    fid = calculate_metrics(input1=TEMP_REAL_DIR, input2=TEMP_GEN_DIR, cuda=torch.cuda.is_available(), isc=False, fid=True, kid=False, verbose=False)['frechet_inception_distance']
    
    gen_tensor = torch.from_numpy(gen_images).permute(0, 3, 1, 2).to(torch.float32) / 127.5 - 1.0
    total_lpips = 0
    indices = np.random.choice(len(gen_tensor), size=(LPIPS_SAMPLE_PAIRS, 2), replace=True)
    for i in tqdm(range(LPIPS_SAMPLE_PAIRS), desc=f"LPIPS 평가 중"):
        img1, img2 = gen_tensor[indices[i][0]:indices[i][0]+1].to(device), gen_tensor[indices[i][1]:indices[i][1]+1].to(device)
        total_lpips += lpips_model(img1, img2).item()
    lpips_score = total_lpips / LPIPS_SAMPLE_PAIRS
    
    shutil.rmtree(TEMP_REAL_DIR); shutil.rmtree(TEMP_GEN_DIR)
    return fid, lpips_score

# --- 메인 실행 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    lpips_model = lpips.LPIPS(net='vgg').to(device)
    real_images_cache = {}
    final_eval_results = []

    print(f"\n--- '{SUBMISSION_DIR}' 폴더의 pkl 파일 평가를 시작합니다 ---")
    for cond in ['a', 'b', 'c']:
        generated_file_path = os.path.join(SUBMISSION_DIR, f"NIST_Task2_{cond}.pkl")
        
        real_imgs = load_real_images(REAL_DATA_PATHS, cond, real_images_cache)
        gen_imgs = load_generated_images(generated_file_path)

        if gen_imgs is None:
            print(f"'{cond}' 조건 pkl 파일이 없어 평가를 건너<binary data, 2 bytes, 15 bytes>니다.")
            continue
            
        fid, lpips_score = evaluate_performance(real_imgs, gen_imgs, lpips_model, device)
        final_eval_results.append({"condition": cond, "fid": fid, "lpips": lpips_score})

    if final_eval_results:
        print("\n\n--- 📊 최종 평가 점수 ---")
        print("="*55)
        print(f"{'조명 조건':<15} | {'Fidelity (FID)':<20} | {'Diversity (LPIPS)':<20}")
        print("-"*55)
        for result in final_eval_results:
            print(f"{result['condition']:<15} | {result['fid']:<20.4f} | {result['lpips']:<20.4f}")
        print("="*55)