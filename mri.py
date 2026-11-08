import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2

folder_path = "folder1"
valid_extensions = (".png", ".jpg")

image_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.lower().endswith(valid_extensions)
]

if len(image_files) == 0:
    print(
        "Blad nie znaleziono plikow"
    )
    image_files = [None] * 30

noise_levels = [0.05, 0.15, 0.30]

all_results = []

for idx, file_path in enumerate(image_files):
    if file_path is not None:
        original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    else:
        original_image = np.random.randint(50, 200, (256, 256), dtype=np.uint8)

    if original_image is None:
        continue

    original_float = original_image.astype(np.float64) / 255.0

    for std_dev in noise_levels:
        noise = np.random.normal(
            loc=0.0, scale=std_dev, size=original_float.shape
        )
        noisy_float = np.clip(original_float + noise, 0, 1)
        current_noisy = (noisy_float * 255).astype(np.uint8)

        filters = {}
        times = {}

        t0 = time.perf_counter()
        filters["Gauss"] = cv2.GaussianBlur(current_noisy, (5, 5), sigmaX=1.5)
        times["Gauss"] = time.perf_counter() - t0


        t0 = time.perf_counter()
        filters["Medianowy"] = cv2.medianBlur(current_noisy, ksize=5)
        times["Medianowy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filters["NLM"] = cv2.fastNlMeansDenoising(
            current_noisy, h=40, templateWindowSize=7, searchWindowSize=21
        )

        times["NLM"] = time.perf_counter() - t0



        for method_name, filtered_img in filters.items():
            val_psnr = psnr(original_image, filtered_img)
            val_ssim = ssim(original_image, filtered_img, data_range=255)

            all_results.append(
                {
                    "Obraz_ID": idx +1,
                    "Poziom_Szumu": std_dev,
                    "Metoda": method_name,
                    "PSNR": val_psnr,
                    "SSIM": val_ssim,
                    "Czas_s": times[method_name],
                }
            )
        

df_results = pd.DataFrame(all_results)
print("WYNIKI KONCOWE")
summary_table = (
    df_results.groupby(["Poziom_Szumu", "Metoda"])[
        ["PSNR", "SSIM", "Czas_s"]
    ]
    .mean()
    .reset_index()
)

print(summary_table.to_string(index=False))
print("===============================")
print(f"Liczba obrazów: {len(image_files)}")