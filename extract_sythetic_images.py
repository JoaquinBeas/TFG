import os
import gzip
import struct
import numpy as np
from PIL import Image


def extract_images_from_idx_gz(gzip_path: str) -> np.ndarray:
    """
    Read an IDX-format image file compressed with gzip, and return a
    NumPy array of shape (N, H, W) of dtype uint8.
    """
    with gzip.open(gzip_path, 'rb') as f:
        # header: magic, number of images, rows, cols
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        # read the image data
        buf = f.read(n * rows * cols)
    images = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return images


def main():
    base_dir = 'data/generated_datasets'
    output_dir = 'data/images'
    os.makedirs(output_dir, exist_ok=True)

    for ckpt_name in os.listdir(base_dir):
        ckpt_path = os.path.join(base_dir, ckpt_name)
        if not os.path.isdir(ckpt_path):
            continue
        img_gz = os.path.join(ckpt_path, 'train-images-idx3-ubyte.gz')
        if not os.path.exists(img_gz):
            print(f"No images file in {ckpt_name}, skipping.")
            continue
        images = extract_images_from_idx_gz(img_gz)
        print(f"Extracted {len(images)} images from {ckpt_name}")
        for idx, arr in enumerate(images):
            filename = f"{ckpt_name}_{idx:05d}.png"
            out_path = os.path.join(output_dir, filename)
            Image.fromarray(arr).save(out_path)

    print(f"All images extracted into '{output_dir}'")


if __name__ == '__main__':
    main()
