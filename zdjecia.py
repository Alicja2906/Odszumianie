import os
import tarfile
import tempfile
import gzip
import nibabel as nib
import numpy as np
import cv2

TAR_FILE = "D:/Studia/Konferencja/Konf/IXI-T1.tar"
OUTPUT_FOLDER = "D:/Studia/Konferencja/Konf/folder1"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

with tarfile.open(TAR_FILE, "r") as tar:

    nii_files = [
        m for m in tar.getmembers()
        if m.name.endswith(".nii.gz")
    ][:10]

    print(f"Znaleziono {len(nii_files)} plików.")

    for member in nii_files:

        print("Przetwarzam:", member.name)

        temp_dir = tempfile.mkdtemp()

        gz_path = os.path.join(
            temp_dir,
            os.path.basename(member.name)
        )

        with tar.extractfile(member) as source:
            with open(gz_path, "wb") as target:
                target.write(source.read())

        nii_path = gz_path[:-3]

        with gzip.open(gz_path, "rb") as f_in:
            with open(nii_path, "wb") as f_out:
                f_out.write(f_in.read())

        img = nib.load(nii_path)
        data = img.get_fdata()

        patient_name = os.path.basename(
            member.name
        ).replace(".nii.gz", "")

        slices = {
            "axial": data[:, :, data.shape[2] // 2],
            "coronal": data[:, data.shape[1] // 2, :],
            "sagittal": data[data.shape[0] // 2, :, :]
        }

        for view_name, img_slice in slices.items():

            img_slice = (
                (img_slice - img_slice.min()) /
                (img_slice.max() - img_slice.min())
                * 255
            ).astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    OUTPUT_FOLDER,
                    f"{patient_name}_{view_name}.png"
                ),
                img_slice
            )

print("Gotowe.")