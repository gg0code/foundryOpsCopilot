from PIL import Image
from pathlib import Path
import os

INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("compressed_images")
TARGET_SIZE = 4900  # bytes, around 4.9 KB

OUTPUT_DIR.mkdir(exist_ok=True)

def compress_image(input_path, output_path, target_size=TARGET_SIZE):
    img = Image.open(input_path)

    # Convert to grayscale to reduce size
    img = img.convert("L")

    # Start with original size
    width, height = img.size

    quality = 85

    while True:
        temp_path = output_path

        img.save(
            temp_path,
            format="JPEG",
            quality=quality,
            optimize=True
        )

        size = os.path.getsize(temp_path)

        if size <= target_size:
            return size

        # First reduce quality
        if quality > 15:
            quality -= 5
        else:
            # Then reduce dimensions
            width = int(width * 0.9)
            height = int(height * 0.9)

            if width < 100 or height < 100:
                return size  # cannot reduce meaningfully further

            img = img.resize((width, height))


for file in INPUT_DIR.iterdir():
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        output_file = OUTPUT_DIR / f"{file.stem}_compressed.jpg"

        final_size = compress_image(file, output_file)

        print(f"{file.name} → {output_file.name} : {final_size / 1024:.2f} KB")