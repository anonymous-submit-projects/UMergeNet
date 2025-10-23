import os
import cv2
import yaml
import shutil
import numpy as np
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from Trainer import compute_segmentation_metrics, compute_iou
import glob
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
import tempfile
import itertools


def evaluate_yolo_segmentation(model, data_loader, device='cuda', num_classes=1):
    model.to(device)
    model.eval()

    dices, ious, precisions, recalls, f1s = [], [], [], [], []

    for images, gt_masks in tqdm(data_loader, desc="Evaluating YOLO Segmentation"):
        images = images.to(device)
        gt_masks = gt_masks.to(device)

        # convert grayscale to RGB if necessary
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)  # [1, C, H, W]
            gt_mask = gt_masks[i]          # [1, H, W]

            # Inference
            results = model(img, verbose=False, show=False)
            masks = results[0].masks
            if masks is None:
                continue

            # predicted mask
            mask_array = masks.data.cpu().numpy()  # [N, H, W]
            pred_mask = torch.from_numpy(np.any(mask_array, axis=0).astype(np.uint8)).unsqueeze(0)

            # binary ground truth
            gt_mask = (gt_mask > 0).to(torch.uint8)

            # metrics using your original function
            mean_dice, mean_iou, mean_precision, mean_recall, mean_f1, _ = compute_segmentation_metrics(
                preds=pred_mask,
                targets=gt_mask,
                num_classes=num_classes
            )

            dices.append(mean_dice)
            ious.append(mean_iou)
            precisions.append(mean_precision)
            recalls.append(mean_recall)
            f1s.append(mean_f1)

    metrics_raw = {
        "Dice": np.mean(dices),
        "mIoU": np.mean(ious),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "F1": np.mean(f1s)
    }

    # format to 3 decimal places
    return {k: float(f"{v:.3f}") for k, v in metrics_raw.items()}



def show_yolo_annotation(image_path, txt_path, class_colors={0: (255,0,0),
                                                             1: (0,255,0),
                                                             2: (0,0,255), 
                                                             3:(255,0,255), 
                                                             4:(255,255,0), 
                                                             5:(255,0,255)}):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if class_colors is None:
        class_colors = {}  # assigns random colors to each class
    
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        coords[:, 0] *= w  # X
        coords[:, 1] *= h  # Y
        coords = coords.astype(int)

        color = class_colors.get(class_id, tuple(np.random.randint(0,255,3)))
        cv2.polylines(img, [coords], isClosed=True, color=color, thickness=2)

    plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()



def yolo_get_mask(model, image_path, show=True):
    # Inference
    results = model(image_path)
    masks = results[0].masks

    if masks is None:
        print("No masks detected.")
        return None

    # [N, H, W] → combines all masks
    mask_array = masks.data.cpu().numpy()
    full_mask = np.any(mask_array, axis=0).astype(np.uint8) * 255  

    if show:
        plt.imshow(full_mask, cmap="gray")
        plt.axis("off")
        plt.show()

    return full_mask



def convert_masks_to_yolo_txt(input_dir, class_names, splits=("train", "valid", "test"), multiclass_mode=False, pixel_map=None):
    num_classes = len(class_names)

    
    for split in splits:
        lbl_dir = os.path.join(input_dir, "labels", split)
        if not os.path.exists(lbl_dir):
            print(f"Folder not found: {lbl_dir}")
            continue

        extensoes = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        mask_paths = list(itertools.chain.from_iterable(
            glob.glob(os.path.join(lbl_dir, ext)) for ext in extensoes
        ))
        if not mask_paths:
            print(f"No masks found on: {lbl_dir}")
            continue

        print(f"Converting {len(mask_paths)} masks in {lbl_dir}...")

        # Create temporary directory for normalized masks
        with tempfile.TemporaryDirectory() as tmp_mask_dir:
            for mask_path in mask_paths:
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    print(f"Failed to read mask: {mask_path}")
                    continue

                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # If it's multiclass
                if multiclass_mode:
                    # Ignores pixels with 255 (keeps 0, 1, 2)
                    #mask[mask == 255] = 0 # optional: reset ignored regions
                    mask = np.clip(mask, 0, num_classes - 1)

                    if pixel_map is None:
                        pixel_map = {}

                    # ---FIXED PIXEL MAPPING BLOCK (Using 254) ---
                    if pixel_map:
                        mapped_mask = mask.copy()
                        
                        # Maps temporary values ​​(254) to "to" values
                        # This must be done value by value to maintain correspondence
                        for old_val, new_val in pixel_map.items():
                            # Only replaces areas that were marked as TEMP_VALUE
                            # and which correspond to the 'new_val' of our map.
                            # It is necessary to map back from the ORIGINAL mask to ensure accuracy.
                            
                            # More robust approach:
                            mapped_mask[mask == old_val] = new_val

                        mask = mapped_mask
                    # ---END OF THE NEW BLOCK ---
                else:
                # If there are only two classes
                    # Reclassifies 255 -> 1, keeps only 0 and 1
                    mask[mask == 255] = 1
                    mask[mask != 0] = 1  # ensures that only 0 and 1 remain

                tmp_path = os.path.join(tmp_mask_dir, os.path.basename(mask_path))
                cv2.imwrite(tmp_path, mask)

            # Converts normalized masks (without touching the original dataset)
            convert_segment_masks_to_yolo_seg(
                masks_dir=tmp_mask_dir,
                output_dir=lbl_dir,  # saves the .txt in the original location
                classes=num_classes
            )

    # Create the data.yaml file
    abs_out = os.path.abspath(input_dir)
    data_yaml = {
        "train": os.path.join(abs_out, "images", "train"),
        "val": os.path.join(abs_out, "images", "valid"),
        "test": os.path.join(abs_out, "images", "test"),
        "nc": num_classes,
        "names": class_names
    }

    yaml_path = os.path.join(abs_out, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"Conversion completed and data.yaml created in: {yaml_path}")