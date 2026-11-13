import os
import shutil
from tqdm import tqdm

# dataset configuration
dataset1_images = r"C:\path\to\dataset1\images"
dataset1_labels = r"C:\path\to\dataset1\labels"

dataset2_images = r"C:\path\to\dataset2\images"
dataset2_labels = r"C:\path\to\dataset2\labels"

output_images = r"C:\path\to\merged\train\images"
output_labels = r"C:\path\to\merged\train\labels"


os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)


def copy_with_prefix(img_dir, lbl_dir, prefix):
    for file in tqdm(os.listdir(img_dir), desc=f"Processing {prefix}"):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name = os.path.splitext(file)[0]

        src_img = os.path.join(img_dir, file)
        src_lbl = os.path.join(lbl_dir, name + ".txt")

        new_name = f"{prefix}_{name}"

        # keep original extension
        ext = os.path.splitext(file)[1]

        dst_img = os.path.join(output_images, new_name + ext)
        dst_lbl = os.path.join(output_labels, new_name + ".txt")

        # Copy image
        shutil.copy2(src_img, dst_img)

        # Copy label OR create empty one
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            open(dst_lbl, "w").close()


def verify_dataset():
    images = set(os.path.splitext(f)[0] for f in os.listdir(output_images))
    labels = set(os.path.splitext(f)[0] for f in os.listdir(output_labels))

    missing_labels = images - labels
    missing_images = labels - images

    print("\n=== VERIFICATION ===")
    print("Total images:", len(images))
    print("Total labels:", len(labels))
    print("Missing labels:", len(missing_labels))
    print("Missing images:", len(missing_images))

    if len(missing_labels) == 0 and len(missing_images) == 0:
        print("Dataset is clean and aligned")
    else:
        print("Issues detected")


# run it
copy_with_prefix(dataset1_images, dataset1_labels, "A")
copy_with_prefix(dataset2_images, dataset2_labels, "B")

verify_dataset()

print("\n Merge complete")