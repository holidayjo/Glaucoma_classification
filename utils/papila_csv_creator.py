import os
import cv2
import numpy as np
import pandas as pd
import math

# ==========================================
# 1. RDR & CDR Mathematical Functions
# ==========================================
def get_centroid(binary_img):
    coords = np.column_stack(np.where(binary_img > 0)) 
    if len(coords) == 0:
        return None
    cy, cx = np.mean(coords, axis=0)
    return (cx, cy)

def find_boundary_along_line(binary_img, start, direction, step=0.5, max_steps=2000):
    h, w = binary_img.shape[:2]
    length = math.hypot(direction[0], direction[1])
    if length < 1e-12:
        return start
    
    dx = direction[0] / length
    dy = direction[1] / length
    
    x, y = start
    boundary_x, boundary_y = x, y
    
    for _ in range(max_steps):
        x_next = x + dx * step
        y_next = y + dy * step
        
        ix_next = int(round(x_next))
        iy_next = int(round(y_next))
        
        if ix_next < 0 or ix_next >= w or iy_next < 0 or iy_next >= h:
            break
        
        if binary_img[iy_next, ix_next] > 0: 
            x, y = x_next, y_next
            boundary_x, boundary_y = x, y
        else:
            break
    return (boundary_x, boundary_y)

def calculate_cdr(optic_cup, optic_disc):
    y_oc = np.where(optic_cup > 0)[0]
    y_od = np.where(optic_disc > 0)[0]

    if len(y_od) == 0:
        return None # Invalid Disc
    if len(y_oc) == 0:
        return 0.0  # Healthy eye with no visible cup

    oc_vd = y_oc.max() - y_oc.min()
    od_vd = y_od.max() - y_od.min()

    if od_vd == 0:
        return None
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
        dx = math.cos(theta)
        dy = math.sin(theta)
        
        Q = find_boundary_along_line(disc_img, disc_center, (dx, dy))
        disc_radius = math.dist(disc_center, Q)
        
        P = find_boundary_along_line(cup_img, disc_center, (dx, dy))
        rim_thickness = math.dist(P, Q)
        
        if disc_radius > 1e-6:
            possible_rdr_values.append(rim_thickness / disc_radius)
            
    if not possible_rdr_values: return None
    return min(possible_rdr_values)

# ==========================================
# 2. Main Extraction Loop
# ==========================================
if __name__ == "__main__":
    
    # Define the base paths according to your structure
    IMG_BASE_DIR  = "data/3_PapilaDB/croped_images_removed_suspect"
    MASK_BASE_DIR = "data/3_PapilaDB/segmentedRetinographies_suspect_removed"
    OUTPUT_CSV    = "data/final_combined_results_papila_suspect_removed.csv"
    
    categories = ["glaucoma", "normal"]
    results = []
    global_index = 0
    
    print("🚀 Starting PapilaDB Clinical Feature Extraction...")
    
    for category in categories:
        img_dir  = os.path.join(IMG_BASE_DIR, category)
        disc_dir = os.path.join(MASK_BASE_DIR, category, "opticDisc")
        cup_dir  = os.path.join(MASK_BASE_DIR, category, "opticCup")
        
        if not os.path.exists(img_dir):
            print(f"Directory not found: {img_dir}")
            continue
            
        images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        label = 1 if category == "glaucoma" else 0
        
        print(f"Processing {len(images)} images in '{category}'...")
        
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            
            # The mask names generated previously look like RET001OD_disc.png
            disc_path = os.path.join(disc_dir, f"{base_name}_disc.png")
            cup_path = os.path.join(cup_dir, f"{base_name}_cup.png")
            
            # Fallback in case they don't have the _disc or _cup suffix
            if not os.path.exists(disc_path): disc_path = os.path.join(disc_dir, f"{base_name}.png")
            if not os.path.exists(cup_path): cup_path = os.path.join(cup_dir, f"{base_name}.png")
                
            if not os.path.exists(disc_path) or not os.path.exists(cup_path):
                print(f"  [Warning] Missing masks for {img_name}")
                continue
                
            # Load Masks
            optic_disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
            optic_cup = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
            
            # Calculate Metrics
            cdr_val = calculate_cdr(optic_cup, optic_disc)
            rdr_val = compute_rdr(optic_disc, optic_cup, n_directions=12)
            
            # Format and Append
            cdr_str = f"{cdr_val:.4f}" if cdr_val is not None else "NaN"
            rdr_str = f"{rdr_val:.4f}" if rdr_val is not None else "NaN"
            
            results.append({
                "Dataset": category,  # <--- CHANGED THIS LINE
                "Image Index": global_index,
                "Original Image Name": img_name,
                "Label": label,       # <--- ALREADY HANDLES 1 OR 0
                "CDR": cdr_str,
                "RDR": rdr_str
            })
            
            global_index += 1

    # ==========================================
    # 3. Save to CSV
    # ==========================================
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Successfully saved {len(df)} records to {OUTPUT_CSV}")


# import os
# import cv2
# import numpy as np
# import pandas as pd
# import math

# # ==========================================
# # 1. RDR & CDR Mathematical Functions
# # ==========================================
# def get_centroid(binary_img):
#     coords = np.column_stack(np.where(binary_img > 0)) 
#     if len(coords) == 0:
#         return None
#     cy, cx = np.mean(coords, axis=0)
#     return (cx, cy)

# def find_boundary_along_line(binary_img, start, direction, step=0.5, max_steps=2000):
#     h, w = binary_img.shape[:2]
#     length = math.hypot(direction[0], direction[1])
#     if length < 1e-12:
#         return start
    
#     dx = direction[0] / length
#     dy = direction[1] / length
    
#     x, y = start
#     boundary_x, boundary_y = x, y
    
#     for _ in range(max_steps):
#         x_next = x + dx * step
#         y_next = y + dy * step
        
#         ix_next = int(round(x_next))
#         iy_next = int(round(y_next))
        
#         if ix_next < 0 or ix_next >= w or iy_next < 0 or iy_next >= h:
#             break
        
#         if binary_img[iy_next, ix_next] > 0: 
#             x, y = x_next, y_next
#             boundary_x, boundary_y = x, y
#         else:
#             break
#     return (boundary_x, boundary_y)

# def calculate_cdr(optic_cup, optic_disc):
#     y_oc = np.where(optic_cup > 0)[0]
#     y_od = np.where(optic_disc > 0)[0]

#     if len(y_od) == 0:
#         return None # Invalid Disc
#     if len(y_oc) == 0:
#         return 0.0  # Healthy eye with no visible cup

#     oc_vd = y_oc.max() - y_oc.min()
#     od_vd = y_od.max() - y_od.min()

#     if od_vd == 0:
#         return None
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
#         dx = math.cos(theta)
#         dy = math.sin(theta)
        
#         Q = find_boundary_along_line(disc_img, disc_center, (dx, dy))
#         disc_radius = math.dist(disc_center, Q)
        
#         P = find_boundary_along_line(cup_img, disc_center, (dx, dy))
#         rim_thickness = math.dist(P, Q)
        
#         if disc_radius > 1e-6:
#             possible_rdr_values.append(rim_thickness / disc_radius)
            
#     if not possible_rdr_values: return None
#     return min(possible_rdr_values)

# # ==========================================
# # 2. Main Extraction Loop
# # ==========================================
# if __name__ == "__main__":
    
#     # Define the base paths according to your structure
#     IMG_BASE_DIR  = "data/3_PapilaDB/croped_images"
#     MASK_BASE_DIR = "data/3_PapilaDB/segmentedRetinographies"
#     OUTPUT_CSV    = "data/final_combined_results_papila.csv"
    
#     categories = ["glaucoma", "normal"]
#     results = []
#     global_index = 0
    
#     print("🚀 Starting PapilaDB Clinical Feature Extraction...")
    
#     for category in categories:
#         img_dir  = os.path.join(IMG_BASE_DIR, category)
#         disc_dir = os.path.join(MASK_BASE_DIR, category, "opticDisc")
#         cup_dir  = os.path.join(MASK_BASE_DIR, category, "opticCup")
        
#         if not os.path.exists(img_dir):
#             print(f"Directory not found: {img_dir}")
#             continue
            
#         images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
#         label = 1 if category == "glaucoma" else 0
        
#         print(f"Processing {len(images)} images in '{category}'...")
        
#         for img_name in images:
#             base_name = os.path.splitext(img_name)[0]
            
#             # The mask names generated previously look like RET001OD_disc.png
#             disc_path = os.path.join(disc_dir, f"{base_name}_disc.png")
#             cup_path = os.path.join(cup_dir, f"{base_name}_cup.png")
            
#             # Fallback in case they don't have the _disc or _cup suffix
#             if not os.path.exists(disc_path): disc_path = os.path.join(disc_dir, f"{base_name}.png")
#             if not os.path.exists(cup_path): cup_path = os.path.join(cup_dir, f"{base_name}.png")
                
#             if not os.path.exists(disc_path) or not os.path.exists(cup_path):
#                 print(f"  [Warning] Missing masks for {img_name}")
#                 continue
                
#             # Load Masks
#             optic_disc = cv2.imread(disc_path, cv2.IMREAD_GRAYSCALE)
#             optic_cup = cv2.imread(cup_path, cv2.IMREAD_GRAYSCALE)
            
#             # Calculate Metrics
#             cdr_val = calculate_cdr(optic_cup, optic_disc)
#             rdr_val = compute_rdr(optic_disc, optic_cup, n_directions=12)
            
#             # Format and Append
#             cdr_str = f"{cdr_val:.4f}" if cdr_val is not None else "NaN"
#             rdr_str = f"{rdr_val:.4f}" if rdr_val is not None else "NaN"
            
#             results.append({
#                 "Dataset": "PAPILA",
#                 "Image Index": global_index,
#                 "Original Image Name": img_name,
#                 "Label": label,
#                 "CDR": cdr_str,
#                 "RDR": rdr_str
#             })
            
#             global_index += 1

#     # ==========================================
#     # 3. Save to CSV
#     # ==========================================
#     df = pd.DataFrame(results)
#     os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
#     df.to_csv(OUTPUT_CSV, index=False)
    
#     print(f"\n✅ Successfully saved {len(df)} records to {OUTPUT_CSV}")