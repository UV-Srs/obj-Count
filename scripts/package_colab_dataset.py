import os
import zipfile
import time

SOURCE_DIR = r"C:\AI\data"
OUTPUT_ZIP = r"C:\AI\pipe_dataset_colab.zip"

print(f"Packaging master dataset from {SOURCE_DIR} into {OUTPUT_ZIP}...")
start_time = time.time()

# Ensure we create a temp or direct zip
temp_zip = OUTPUT_ZIP + ".tmp"
if os.path.exists(temp_zip):
    os.remove(temp_zip)

with zipfile.ZipFile(temp_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    # 1. Write Colab-tailored data.yaml
    colab_yaml_content = """path: /content/pipe_dataset
train: images/train
val: images/val

nc: 1
names: ['insulation_pipe']
"""
    zf.writestr('pipe_dataset/data.yaml', colab_yaml_content)

    # 2. Add all images and labels
    file_count = 0
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            folder_path = os.path.join(SOURCE_DIR, folder, split)
            if not os.path.exists(folder_path):
                continue
            for fname in os.listdir(folder_path):
                fpath = os.path.join(folder_path, fname)
                if os.path.isfile(fpath) and not fname.endswith('.cache'):
                    arcname = f"pipe_dataset/{folder}/{split}/{fname}"
                    zf.write(fpath, arcname)
                    file_count += 1
                    if file_count % 500 == 0:
                        print(f"  Added {file_count} files...")

# Replace old zip
if os.path.exists(OUTPUT_ZIP):
    os.remove(OUTPUT_ZIP)
os.rename(temp_zip, OUTPUT_ZIP)

elapsed = time.time() - start_time
size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
print("\n" + "="*60)
print(f" Successfully packaged {file_count} files into {OUTPUT_ZIP}!")
print(f" Total Archive Size: {size_mb:.2f} MB")
print(f" Time Taken: {elapsed:.1f} seconds")
print("="*60)

