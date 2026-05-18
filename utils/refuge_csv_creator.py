import os
import cv2
import json
import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Mathematical Helper Functions
# ==========================================
def get_centroid(binary_img):
    coords = np.column_stack(np.where(binary_img > 0)) 
    if len(coords) == 0: return None
    cy, cx = np.mean(coords, axis=0)
    return (cx, cy)

def find_boundary_along_line(binary_img, start, direction, step=0.5, max_steps=2000):
    h, w = binary_img.shape[:2]
    length = math.hypot(direction[0], direction[1])
    if length < 1e-12: return start
    
    dx, dy = direction[0] / length, direction[1] / length
    x, y = start
    boundary_x, boundary_y = x, y
    
    for _ in range(max_steps):
        x_next, y_next = x + dx * step, y + dy * step
        ix_next, iy_next = int(round(x_next)), int(round(y_next))
        
        if ix_next < 0 or ix_next >= w or iy_next < 0 or iy_next >= h: break
        
        if binary_img[iy_next, ix_next] > 0: 
            x, y = x_next, y_next
            boundary_x, boundary_y = x, y
        else: break
    return (boundary_x, boundary_y)

def calculate_cdr(optic_cup, optic_disc):
    y_oc = np.where(optic_cup > 0)[0]
    y_od = np.where(optic_disc > 0)[0]

    if len(y_od) == 0: return None
    if len(y_oc) == 0: return 0.0 

    oc_vd, od_vd = y_oc.max() - y_oc.min(), y_od.max() - y_od.min()
    if od_vd == 0: return None
    return oc_vd / od_vd

def compute_rdr(disc_img, cup_img, n_directions=12):
    disc_center = get_centroid(disc_img)
    cup_center  = get_centroid(cup_img)
    
    if disc_center is None: return None  
    if cup_center is None: return 1.0   
    
    possible_rdr_values = []
    angle_step = 360.0 / n_directions
    
    for i in range(n_directions):
        theta = math.radians(i * angle_step)
        dx, dy = math.cos(theta), math.sin(theta)
        
        Q = find_boundary_along_line(disc_img, disc_center, (dx, dy))
        disc_radius = math.dist(disc_center, Q)
        
        P = find_boundary_along_line(cup_img, disc_center, (dx, dy))
        rim_thickness = math.dist(P, Q)
        
        if disc_radius > 1e-6:
            possible_rdr_values.append(rim_thickness / disc_radius)
            
    if not possible_rdr_values: return None
    return min(possible_rdr_values)

def verify_refuge_extraction(num_samples=3):
    """Updated to verify the unified train/val dataset"""
    print(f"\n=== Verifying Unified REFUGE Dataset ===")
    
    base_dir = "data/2_g1020_origa_refuge/REFUGE"
    img_dir  = os.path.join(base_dir, "cropped_images_train_val")
    mask_dir = os.path.join(base_dir, "segmentedRetinographies")
    csv_path = "data/final_combined_results_refuge.csv"

    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"✅ Found {len(df)} records in CSV. Drawing {num_samples} random samples...\n")

    sample_df = df.sample(n=min(num_samples, len(df)))
    plt.figure(figsize=(15, 5 * num_samples))

    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_name  = row['Original Image Name']
        base_name = img_name.split('.')[0]
        category  = row['Dataset'] 
        cdr_val   = row['CDR']
        rdr_val   = row['RDR']

        orig_img_path = os.path.join(img_dir, category, img_name)
        cup_path      = os.path.join(mask_dir, category, "opticCup", f"opticCup_{base_name}.png")
        disc_path     = os.path.join(mask_dir, category, "opticDisc", f"opticDisc_{base_name}.png")

        img = cv2.imread(orig_img_path)
        if img is None:
            print(f"⚠️ Warning: Could not load image {orig_img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cup  = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
        disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)

        overlay = img.copy()
        if disc is not None: overlay[disc > 0] = [0, 255, 255]
        if cup is not None:  overlay[cup > 0]  = [255, 0, 0]

        alpha = 0.4
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        mask_only = np.zeros_like(img)
        if disc is not None: mask_only[disc > 0] = [128, 128, 128]
        if cup is not None:  mask_only[cup > 0]  = [255, 255, 255]

        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(img)
        plt.title(f"Original ({category})\n{img_name}")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(mask_only)
        plt.title("Extracted Binary Masks\n(Gray=Disc, White=Cup)")
        plt.axis('off')

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(blended)
        plt.title(f"Overlay & Extracted Metrics\nCDR: {cdr_val}  |  RDR: {rdr_val}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 2. Main Processing & Extraction
# ==========================================
if __name__ == "__main__":
    
    # --- UNIFIED CONFIGURATION ---
    CUP_PIXEL_VALUE = 2 
    ROOT_DIR = "data/2_g1020_origa_refuge/REFUGE"
    SPLITS = ["train", "val"]
    
    # Target unified directories
    COMBINED_IMG_DIR = os.path.join(ROOT_DIR, "cropped_images_train_val")
    COMBINED_MASK_DIR = os.path.join(ROOT_DIR, "segmentedRetinographies")
    OUTPUT_CSV = "data/final_combined_results_refuge.csv"

    print(f"{'='*50}")
    print(f"🚀 INITIALIZING UNIFIED REFUGE PROCESSING")
    print(f"📂 Target Images: {COMBINED_IMG_DIR}")
    print(f"📂 Target Masks: {COMBINED_MASK_DIR}")
    print(f"{'='*50}")

    # Create target folder structures
    categories = ["glaucoma", "normal"]
    for cat in categories:
        os.makedirs(os.path.join(COMBINED_IMG_DIR, cat), exist_ok=True)
        os.makedirs(os.path.join(COMBINED_MASK_DIR, cat, "opticCup"), exist_ok=True)
        os.makedirs(os.path.join(COMBINED_MASK_DIR, cat, "opticDisc"), exist_ok=True)

    results = []
    global_index = 0
    success_count = 0

    # Process both train and val splits
    for split in SPLITS:
        split_dir = os.path.join(ROOT_DIR, split)
        json_path = os.path.join(split_dir, "index.json")
        raw_mask_dir = os.path.join(split_dir, "Masks_Cropped")
        raw_img_dir = os.path.join(split_dir, "Images_Cropped")
        
        if not os.path.exists(json_path):
            print(f"❌ ERROR: Missing JSON for '{split}' split at {json_path}")
            continue

        with open(json_path, 'r') as f:
            refuge_data = json.load(f)

        print(f"\nProcessing '{split}' split: {len(refuge_data)} images found.")

        for key in refuge_data.keys():
            item = refuge_data[key]
            
            img_name = item["ImgName"]            
            base_name = img_name.split('.')[0]    
            label = item["Label"]                 
            
            category = "glaucoma" if label == 1 else "normal"
            mask_name = f"{base_name}.png"
            
            src_mask_path = os.path.join(raw_mask_dir, mask_name)
            src_img_path = os.path.join(raw_img_dir, img_name)
            
            if not os.path.exists(src_mask_path) or not os.path.exists(src_img_path):
                print(f"  [Skip] Missing mask or image for: {img_name}")
                continue
                
            # 1. Read Raw Mask (0, 1, 2)
            raw_mask = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue

            # 2. Extract Binary Masks
            optic_cup = np.where(raw_mask == CUP_PIXEL_VALUE, 255, 0).astype(np.uint8)
            optic_disc = np.where(raw_mask > 0, 255, 0).astype(np.uint8)

            # 3. Save Binary Masks to Unified Folders
            cv2.imwrite(os.path.join(COMBINED_MASK_DIR, category, "opticCup", f"opticCup_{base_name}.png"), optic_cup)
            cv2.imwrite(os.path.join(COMBINED_MASK_DIR, category, "opticDisc", f"opticDisc_{base_name}.png"), optic_disc)

            # 4. Copy Image to Unified Folders
            dest_img_path = os.path.join(COMBINED_IMG_DIR, category, img_name)
            shutil.copy2(src_img_path, dest_img_path)

            # 5. Calculate Clinical Metrics
            cdr_val = calculate_cdr(optic_cup, optic_disc)
            rdr_val = compute_rdr(optic_disc, optic_cup, n_directions=12)
            
            cdr_str = f"{cdr_val:.4f}" if cdr_val is not None else "NaN"
            rdr_str = f"{rdr_val:.4f}" if rdr_val is not None else "NaN"

            # 6. Append to Results (using a global continuous index)
            results.append({
                "Dataset": category,
                "Image Index": global_index,
                "Original Image Name": img_name,
                "CDR": cdr_str,
                "RDR": rdr_str
            })
            
            global_index += 1
            success_count += 1

    # ==========================================
    # 3. Save to Final Unified CSV
    # ==========================================
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)

        print(f"\n✅ Done! Unified {success_count} total images and masks from Train & Val.")
        print(f"✅ Images saved to: {COMBINED_IMG_DIR}")
        print(f"✅ Masks saved to: {COMBINED_MASK_DIR}")
        print(f"✅ Unified CSV saved to: {OUTPUT_CSV}")
        
        # Uncomment the line below if you want to visually verify the results automatically!
        # verify_refuge_extraction(num_samples=3)
    else:
        print(f"\n⚠️ No records were processed. CSV was not created.")

# import os
# import cv2
# import json
# import math
# import argparse
# import numpy as np
# import pandas as pd

# # ==========================================
# # 1. Mathematical Helper Functions
# # ==========================================
# def get_centroid(binary_img):
#     coords = np.column_stack(np.where(binary_img > 0)) 
#     if len(coords) == 0: return None
#     cy, cx = np.mean(coords, axis=0)
#     return (cx, cy)

# def find_boundary_along_line(binary_img, start, direction, step=0.5, max_steps=2000):
#     h, w = binary_img.shape[:2]
#     length = math.hypot(direction[0], direction[1])
#     if length < 1e-12: return start
    
#     dx, dy = direction[0] / length, direction[1] / length
#     x, y = start
#     boundary_x, boundary_y = x, y
    
#     for _ in range(max_steps):
#         x_next, y_next = x + dx * step, y + dy * step
#         ix_next, iy_next = int(round(x_next)), int(round(y_next))
        
#         if ix_next < 0 or ix_next >= w or iy_next < 0 or iy_next >= h: break
        
#         if binary_img[iy_next, ix_next] > 0: 
#             x, y = x_next, y_next
#             boundary_x, boundary_y = x, y
#         else: break
#     return (boundary_x, boundary_y)

# def calculate_cdr(optic_cup, optic_disc):
#     y_oc = np.where(optic_cup > 0)[0]
#     y_od = np.where(optic_disc > 0)[0]

#     if len(y_od) == 0: return None
#     if len(y_oc) == 0: return 0.0 

#     oc_vd, od_vd = y_oc.max() - y_oc.min(), y_od.max() - y_od.min()
#     if od_vd == 0: return None
#     return oc_vd / od_vd

# def compute_rdr(disc_img, cup_img, n_directions=12):
#     disc_center = get_centroid(disc_img)
#     cup_center  = get_centroid(cup_img)
    
#     if disc_center is None: return None  
#     if cup_center is None: return 1.0   
    
#     possible_rdr_values = []
#     angle_step = 360.0 / n_directions
    
#     for i in range(n_directions):
#         theta = math.radians(i * angle_step)
#         dx, dy = math.cos(theta), math.sin(theta)
        
#         Q = find_boundary_along_line(disc_img, disc_center, (dx, dy))
#         disc_radius = math.dist(disc_center, Q)
        
#         P = find_boundary_along_line(cup_img, disc_center, (dx, dy))
#         rim_thickness = math.dist(P, Q)
        
#         if disc_radius > 1e-6:
#             possible_rdr_values.append(rim_thickness / disc_radius)
            
#     if not possible_rdr_values: return None
#     return min(possible_rdr_values)

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# def verify_refuge_extraction(split="val", num_samples=3):
#     print(f"=== Verifying REFUGE '{split.upper()}' Split ===")
    
#     # 1. Define Paths based on the split
#     base_dir = f"data/2_g1020_origa_refuge/REFUGE/{split}"
#     img_dir  = os.path.join(base_dir, "Images_Cropped")
#     mask_dir = os.path.join(base_dir, "segmentedRetinographies")
#     csv_path = f"data/final_combined_results_refuge_{split}.csv"

#     # 2. Load the CSV we generated
#     if not os.path.exists(csv_path):
#         print(f"❌ Error: CSV not found at {csv_path}")
#         return

#     df = pd.read_csv(csv_path)
#     print(f"✅ Found {len(df)} records in CSV. Drawing {num_samples} random samples...\n")

#     # 3. Pick random rows to visualize
#     sample_df = df.sample(n=min(num_samples, len(df)))

#     plt.figure(figsize=(15, 5 * num_samples))

#     for i, (_, row) in enumerate(sample_df.iterrows()):
#         img_name  = row['Original Image Name']
#         base_name = img_name.split('.')[0]
#         category  = row['Dataset']  # "glaucoma" or "normal"
#         cdr_val   = row['CDR']
#         rdr_val   = row['RDR']

#         # Construct specific paths for this image
#         orig_img_path = os.path.join(img_dir, img_name)
#         cup_path      = os.path.join(mask_dir, category, "opticCup", f"opticCup_{base_name}.png")
#         disc_path     = os.path.join(mask_dir, category, "opticDisc", f"opticDisc_{base_name}.png")

#         # Load the images
#         img = cv2.imread(orig_img_path)
#         if img is None:
#             print(f"⚠️ Warning: Could not load image {orig_img_path}")
#             continue
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         cup  = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
#         disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)

#         # Create the Overlay (Cyan for Disc, Red for Cup)
#         overlay = img.copy()
#         if disc is not None: overlay[disc > 0] = [0, 255, 255]  # Cyan
#         if cup is not None:  overlay[cup > 0]  = [255, 0, 0]    # Red

#         # Alpha blending for transparency
#         alpha = 0.4
#         blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

#         # Create a "Mask Only" view to verify the binary extraction worked
#         mask_only = np.zeros_like(img)
#         if disc is not None: mask_only[disc > 0] = [128, 128, 128] # Gray Disc
#         if cup is not None:  mask_only[cup > 0]  = [255, 255, 255] # White Cup

#         # --- Plotting ---
#         plt.subplot(num_samples, 3, i * 3 + 1)
#         plt.imshow(img)
#         plt.title(f"Original ({category})\n{img_name}")
#         plt.axis('off')

#         plt.subplot(num_samples, 3, i * 3 + 2)
#         plt.imshow(mask_only)
#         plt.title("Extracted Binary Masks\n(Gray=Disc, White=Cup)")
#         plt.axis('off')

#         plt.subplot(num_samples, 3, i * 3 + 3)
#         plt.imshow(blended)
#         plt.title(f"Overlay & Extracted Metrics\nCDR: {cdr_val}  |  RDR: {rdr_val}")
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()


# # ==========================================
# # 2. Main Processing & Extraction
# # ==========================================
# if __name__ == "__main__":
    
#     # --- ARGUMENT PARSER ---
#     parser = argparse.ArgumentParser(description="Process REFUGE dataset masks and calculate clinical metrics.")
#     parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], 
#                         help="Which dataset split to process (e.g., 'train' or 'val')")
#     parser.add_argument("--root_dir", type=str, default="data/2_g1020_origa_refuge/REFUGE", 
#                         help="Root directory of the REFUGE dataset")
#     parser.add_argument("--json_name", type=str, default="index.json", 
#                         help="Name of the JSON file containing the labels")
#     parser.add_argument("--cup_pixel", type=int, default=2, 
#                         help="Pixel value representing the Optic Cup in the raw masks")
    
#     # If running in Jupyter Notebook, we need to handle args differently to avoid conflicts with Jupyter's own args
#     import sys
#     if 'ipykernel' in sys.modules:
#         args = parser.parse_args(args=[]) # Use default arguments in Jupyter
#         # To change split in Jupyter, just manually overwrite the variable here:
#         # args.split = "train" 
#     else:
#         args = parser.parse_args()

#     # --- DYNAMIC CONFIGURATION ---
#     BASE_DIR = os.path.join(args.root_dir, args.split)
    
#     # Assume JSON is inside the split folder, fallback to current directory if not
#     JSON_PATH = os.path.join(BASE_DIR, args.json_name)
#     if not os.path.exists(JSON_PATH):
#         JSON_PATH = args.json_name 
        
#     RAW_MASK_DIR = os.path.join(BASE_DIR, "Masks_Cropped")
#     OUTPUT_MASK_DIR = os.path.join(BASE_DIR, "segmentedRetinographies")
#     OUTPUT_CSV = f"data/final_combined_results_refuge_{args.split}.csv"

#     print(f"{'='*50}")
#     print(f"🚀 INITIALIZING REFUGE PROCESSING FOR SPLIT: [{args.split.upper()}]")
#     print(f"📂 Base Directory: {BASE_DIR}")
#     print(f"📄 Label File: {JSON_PATH}")
#     print(f"{'='*50}")

#     # Create RIM-ONE style folders dynamically
#     folders_to_create = [
#         "glaucoma/opticCup", "glaucoma/opticDisc",
#         "normal/opticCup", "normal/opticDisc"
#     ]
#     for folder in folders_to_create:
#         os.makedirs(os.path.join(OUTPUT_MASK_DIR, folder), exist_ok=True)

#     # Load the JSON labels
#     try:
#         with open(JSON_PATH, 'r') as f:
#             refuge_data = json.load(f)
#     except FileNotFoundError:
#         print(f"❌ ERROR: Could not find label file at {JSON_PATH}. Please ensure the file exists.")
#         sys.exit(1)

#     results = []
#     success_count = 0

#     print(f"Processing {len(refuge_data)} images from {args.split} JSON...")

#     # Iterate through the JSON dictionary
#     for idx, key in enumerate(refuge_data.keys()):
#         item = refuge_data[key]
        
#         img_name = item["ImgName"]            
#         base_name = img_name.split('.')[0]    
#         label = item["Label"]                 
        
#         category = "glaucoma" if label == 1 else "normal"
#         mask_name = f"{base_name}.png"
#         raw_mask_path = os.path.join(RAW_MASK_DIR, mask_name)
        
#         if not os.path.exists(raw_mask_path):
#             print(f"  [Skip] Missing mask: {mask_name}")
#             continue
            
#         # 1. Read Raw Mask (0, 1, 2)
#         raw_mask = cv2.imread(raw_mask_path, cv2.IMREAD_GRAYSCALE)
        
#         if raw_mask is None:
#             print(f"  [Error] Failed to read: {raw_mask_path}")
#             continue

#         # 2. Extract Binary Masks (White = 255, Black = 0)
#         optic_cup = np.where(raw_mask == args.cup_pixel, 255, 0).astype(np.uint8)
#         optic_disc = np.where(raw_mask > 0, 255, 0).astype(np.uint8)

#         # 3. Save Binary Masks to Folders
#         cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, category, "opticCup", f"opticCup_{base_name}.png"), optic_cup)
#         cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, category, "opticDisc", f"opticDisc_{base_name}.png"), optic_disc)

#         # 4. Calculate Clinical Metrics
#         cdr_val = calculate_cdr(optic_cup, optic_disc)
#         rdr_val = compute_rdr(optic_disc, optic_cup, n_directions=12)
        
#         cdr_str = f"{cdr_val:.4f}" if cdr_val is not None else "NaN"
#         rdr_str = f"{rdr_val:.4f}" if rdr_val is not None else "NaN"

#         # 5. Append to Results matching your exact required CSV columns
#         results.append({
#             "Dataset": category,
#             "Image Index": idx,
#             "Original Image Name": img_name,
#             "CDR": cdr_str,
#             "RDR": rdr_str
#         })
        
#         success_count += 1

#     # ==========================================
#     # 3. Save to Final CSV
#     # ==========================================
#     if results:
#         df = pd.DataFrame(results)
#         os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#         df.to_csv(OUTPUT_CSV, index=False)

#         print(f"\n✅ Done! Extracted {success_count} binary mask pairs for '{args.split}' set.")
#         print(f"✅ CSV successfully saved to: {OUTPUT_CSV}")
#     else:
#         print(f"\n⚠️ No records were processed. CSV was not created.")