import os
import shutil
import glob

input_dir = r"c:\workspace\algorithm\Optical Flow\input\3"
target_dir = r"c:\workspace\algorithm\Optical Flow\train\incoming\images"

os.makedirs(target_dir, exist_ok=True)

images = glob.glob(os.path.join(input_dir, "*.jpg"))
for img in images:
    shutil.copy(img, target_dir)

print(f"Copied {len(images)} images to {target_dir}")
