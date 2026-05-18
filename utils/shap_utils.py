# utils/shap_utils.py
import os
import glob
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

from models.gcfn import GlaucomaChemicalFusionNetwork
import utils.dataset as utils_dataset
from utils.utils import set_seed
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

class GCFNWrapper(torch.nn.Module):
    """DeepSHAP을 위해 GCFN의 두 입력(Deep, Clinical)을 하나로 묶어주는 Wrapper"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        d_feats = x[:, :-2]
        c_feats = x[:, -2:]
        return self.model(d_feats, c_feats)

# --- [추가된 부분] SHAP 에러 방지를 위한 inplace 속성 제거 함수 ---
def remove_inplace(m):
    if hasattr(m, 'inplace'):
        m.inplace = False
# -----------------------------------------------------------------

def run_notebook_shap_analysis(df, 
                               model_names      = ["custom_resnet", "convnext", "swin"],
                               weights          = "glaucolite_imagenet_epoch_75.pth",
                               root_glauc       = "data/2_g1020_origa_refuge/REFUGE/cropped_images_train_val/glaucoma",
                               root_normal      = "data/2_g1020_origa_refuge/REFUGE/cropped_images_train_val/normal",
                               load_weights_dir = "weights/1_5_refuge_glite_convt_swin_gcfn",
                               num_examples     = 3,
                               k_fold           = 10,
                               seed             = 0, 
                               device           = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                               save_dir         = "shap_reports"):
    print("=== 1. Starting Feature Extraction ===")
    set_seed(seed)
    os.makedirs(save_dir, exist_ok=True)
    
    # 특징 추출
    all_feats, base_idxs, y_final = [], None, None
    for m_name in model_names:
        print(f"Extracting features using {m_name}...")
        w          = weights if m_name == "custom_resnet" else None
        X, y, idxs = utils_dataset.build_dataset(df, m_name, root_glauc, root_normal, device, w)
        if base_idxs is None: 
            base_idxs, y_final = idxs, y # X: features, y: label
        all_feats.append(X) # 6400

    X_deep = np.hstack(all_feats)
    X_clin = df.loc[base_idxs, ["CDR", "RDR"]].values # base_idxs: rows, ["CDR", "RDR"]: columns

    print("\n=== 2. Isolating Fold 1 & Making Predictions ===")
    # Fold 1 분리
    skf                 = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed) # index 기반 분할
    train_idx, test_idx = next(skf.split(X_deep, y_final))  # 6400개의 feature와 label을 호출시마다 fold의 train/test로 분할. index 반환.
    
    X_te_d, X_te_c, y_te = X_deep[test_idx],  X_clin[test_idx], y_final[test_idx]
    X_tr_d, X_tr_c       = X_deep[train_idx], X_clin[train_idx]
    df_test_indices      = np.array(base_idxs)[test_idx]

    # 가중치 로드
    weight_path    = os.path.join(load_weights_dir, "gcfn_fold_1.pth")  # hard-coded for fold 1.,;;;
    print(f"Loading weights from: {weight_path}")
    loaded_weights = torch.load(weight_path, map_location=device, weights_only=False)
    
    model = GlaucomaChemicalFusionNetwork(num_deep_features=X_deep.shape[1], num_classes=2).to(device)
    model.load_state_dict(loaded_weights['model_state_dict'])
    model.eval()
    
    model.apply(remove_inplace) # 모델 내의 모든 inplace=True를 inplace=False로 변경 (SHAP RuntimeError 방지)
    
    scalers = loaded_weights['scalers'] 

    t_X_te_d = torch.tensor(scalers[0].transform(X_te_d), dtype=torch.float32).to(device)
    t_X_te_c = torch.tensor(scalers[1].transform(X_te_c), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # dim=1: to apply the softmax across the 2 columns (Class 0 and Class 1).
        # all the rows (:), but only for column index of "1" (Glaucoma)
        probs_test = torch.softmax(model(t_X_te_d, t_X_te_c), dim=1)[:, 1].cpu().numpy()
        
    p, r, t     = precision_recall_curve(y_te, probs_test)
    best_thresh = t[np.argmax(2 * p * r / (p + r + 1e-8))] if len(t) > 0 else 0.5 # using best f1-score
    preds       = (probs_test >= best_thresh).astype(int)

    print("\n=== 3. Selecting Representative Cases (TP, TN, FP, FN) ===")
    res_df = pd.DataFrame({'df_idx': df_test_indices, 'true_label': y_te, 'pred_label': preds, 'prob': probs_test})
    
    selected_cases = {}
    # num_examples = 3 # Change this to generate more or fewer examples per category
    
    # 1. Extract Top True Positives (Highest Confidence)
    tp_df = res_df[(res_df['true_label'] == 1) & (res_df['pred_label'] == 1)].sort_values(by='prob', ascending=False)
    for i in range(min(num_examples, len(tp_df))):
        selected_cases[f"TP_{i+1}"] = tp_df.iloc[i]
        
    # 2. Extract Top True Negatives (Lowest Confidence / Most sure it's normal)
    tn_df = res_df[(res_df['true_label'] == 0) & (res_df['pred_label'] == 0)].sort_values(by='prob', ascending=True)
    for i in range(min(num_examples, len(tn_df))):
        selected_cases[f"TN_{i+1}"] = tn_df.iloc[i]
        
    # 3. Extract Top False Positives (Highest Confidence in the WRONG prediction)
    fp_df = res_df[(res_df['true_label'] == 0) & (res_df['pred_label'] == 1)].sort_values(by='prob', ascending=False)
    for i in range(min(num_examples, len(fp_df))):
        selected_cases[f"FP_{i+1}"] = fp_df.iloc[i]
    if fp_df.empty:
        print("[Warning] No False Positive (FP) cases found in this fold.")
        
    # 4. Extract Top False Negatives (Lowest Confidence in the WRONG prediction)
    fn_df = res_df[(res_df['true_label'] == 1) & (res_df['pred_label'] == 0)].sort_values(by='prob', ascending=True)
    for i in range(min(num_examples, len(fn_df))):
        selected_cases[f"FN_{i+1}"] = fn_df.iloc[i]
    if fn_df.empty:
        print("[Warning] No False Negative (FN) cases found in this fold.")

    if not selected_cases:
        print("[Error] No cases were selected. Exiting SHAP generation.")
        return selected_cases

    for c_type, c_data in selected_cases.items():
        print(f"[{c_type}] DataFrame Index: {int(c_data['df_idx'])} | Prob: {c_data['prob']:.4f}")

    print("\n=== 4. Generating SHAP Explanations ===")
    # Background 데이터 생성 (Train set 무작위 샘플링)
    bg_size    = min(100, len(X_tr_d)) # 100
    bg_indices = np.random.choice(len(X_tr_d), bg_size, replace=False)
    
    bg_X_d      = torch.tensor(scalers[0].transform(X_tr_d[bg_indices]), dtype=torch.float32).to(device)
    bg_X_c      = torch.tensor(scalers[1].transform(X_tr_c[bg_indices]), dtype=torch.float32).to(device)
    bg_combined = torch.cat((bg_X_d, bg_X_c), dim=1)

    wrapped_model = GCFNWrapper(model).to(device)
    explainer     = shap.GradientExplainer(wrapped_model, bg_combined)

    for case_type, case_data in selected_cases.items():
        df_idx  = int(case_data['df_idx'])
        arr_idx = np.where(np.array(base_idxs) == df_idx)[0][0]
        
        c_X_d  = torch.tensor(scalers[0].transform(X_deep[arr_idx:arr_idx+1]), dtype=torch.float32).to(device)
        c_X_c  = torch.tensor(scalers[1].transform(X_clin[arr_idx:arr_idx+1]), dtype=torch.float32).to(device)
        c_comb = torch.cat((c_X_d, c_X_c), dim=1)
        
        # DeepExplainer 호출 시 발생하던 inplace 에러 해결 완료!
        # shap_values = explainer.shap_values(c_comb, check_additivity=False)
        # GradientExplainer uses true gradients, so it does not need additivity checks
        shap_values = explainer.shap_values(c_comb)
        
        # PyTorch 텐서가 섞여있을 경우 Numpy로 변환
        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()
        elif isinstance(shap_values, list) and torch.is_tensor(shap_values[0]):
            shap_values = [sv.detach().cpu().numpy() for sv in shap_values]
            
        shap_vals_class1 = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        # 다차원 배열 구조를 1차원으로 강제 평탄화 (Sequence 에러 원천 차단)
        shap_vals_class1 = np.array(shap_vals_class1).flatten()
        
        # GradientExplainer does not automatically store an 'expected_value' attribute.
        # We manually calculate it by taking the mean output of the background dataset.
        if hasattr(explainer, 'expected_value'):
            base_val_raw = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            base_value = float(np.array(base_val_raw).flatten()[0])
        else:
            with torch.no_grad():
                bg_outputs = wrapped_model(bg_combined)
                # Get the mean logit for Class 1 (Glaucoma) across all 100 background patients
                base_value = float(bg_outputs[:, 1].mean().cpu().numpy())
        
        # 시각화를 위해 딥러닝 기여도를 합치고, 완벽한 스칼라(float)로 강제 변환
        deep_shap_sum = float(np.sum(shap_vals_class1[:-2])) # sum of 6400 deep features.
        clin_shap_0   = float(shap_vals_class1[-2])
        clin_shap_1   = float(shap_vals_class1[-1])
        
        simp_shap  = np.array([deep_shap_sum, clin_shap_0, clin_shap_1])
        simp_feats = np.array(["Deep Features (Combined)", "CDR", "RDR"])
        
        c0        = float(X_clin[arr_idx][0])
        c1        = float(X_clin[arr_idx][1])
        # simp_data = np.array([np.nan, c0, c1])
        simp_data = np.array(["Combined", c0, c1], dtype=object)

        explanation = shap.Explanation(values=simp_shap, base_values=base_value, data=simp_data, feature_names=simp_feats)

        # 1. Fetch Image Path and Strip Extension
        row_data = df.loc[df_idx]
        img_name = row_data["Original Image Name"]
        img_base = os.path.splitext(img_name)[0] # Converts 'V0041.jpg' to 'V0041'
        actual_label = int(row_data["Label"])
        
        condition_folder = "glaucoma" if actual_label == 1 else "normal"
        img_path = os.path.join(root_glauc if actual_label == 1 else root_normal, img_name)
        
        # 2. Build Dynamic Paths using Glob (Now handles prefixes like 'opticCup_')
        base_seg_dir = "data/2_g1020_origa_refuge/REFUGE/segmentedRetinographies"
        
        # FIX: Added '*' before {img_base} so it finds 'opticCup_V0041.png'
        disc_matches = glob.glob(os.path.join(base_seg_dir, condition_folder, "opticDisc", f"*{img_base}*.*"))
        cup_matches  = glob.glob(os.path.join(base_seg_dir, condition_folder, "opticCup", f"*{img_base}*.*"))
        
        disc_path = disc_matches[0] if disc_matches else None
        cup_path  = cup_matches[0] if cup_matches else None

        # 3. Print the Clinical and Prediction Data Clearly
        print(f"\n{'='*50}")
        print(f"CASE ANALYSIS: {case_type}")
        print(f"{'='*50}")
        print(f"Image File : {img_name}")
        print(f"True Label : {actual_label} ({'Glaucoma' if actual_label == 1 else 'Normal'})")
        print(f"Prediction : {int(case_data['pred_label'])} (Confidence: {case_data['prob']:.2%})")
        print(f"Clinical   : CDR = {c0:.4f} | RDR = {c1:.4f}")
        print(f"{'='*50}\n")

        print("\n=== 4. Generating SHAP Explanations ===")
    
    # --- PAPER FORMATTING: Set Global Font to Times New Roman ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    # ------------------------------------------------------------
    
    # Background 데이터 생성 (Train set 무작위 샘플링)
    bg_size    = min(100, len(X_tr_d)) # 100
    bg_indices = np.random.choice(len(X_tr_d), bg_size, replace=False)
    
    bg_X_d      = torch.tensor(scalers[0].transform(X_tr_d[bg_indices]), dtype=torch.float32).to(device)
    bg_X_c      = torch.tensor(scalers[1].transform(X_tr_c[bg_indices]), dtype=torch.float32).to(device)
    bg_combined = torch.cat((bg_X_d, bg_X_c), dim=1)

    wrapped_model = GCFNWrapper(model).to(device)
    explainer     = shap.GradientExplainer(wrapped_model, bg_combined)

    for case_type, case_data in selected_cases.items():
        df_idx  = int(case_data['df_idx'])
        arr_idx = np.where(np.array(base_idxs) == df_idx)[0][0]
        
        c_X_d  = torch.tensor(scalers[0].transform(X_deep[arr_idx:arr_idx+1]), dtype=torch.float32).to(device)
        c_X_c  = torch.tensor(scalers[1].transform(X_clin[arr_idx:arr_idx+1]), dtype=torch.float32).to(device)
        c_comb = torch.cat((c_X_d, c_X_c), dim=1)
        
        shap_values = explainer.shap_values(c_comb)
        
        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()
        elif isinstance(shap_values, list) and torch.is_tensor(shap_values[0]):
            shap_values = [sv.detach().cpu().numpy() for sv in shap_values]
            
        shap_vals_class1 = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        shap_vals_class1 = np.array(shap_vals_class1).flatten()
        
        if hasattr(explainer, 'expected_value'):
            base_val_raw = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            base_value = float(np.array(base_val_raw).flatten()[0])
        else:
            with torch.no_grad():
                bg_outputs = wrapped_model(bg_combined)
                base_value = float(bg_outputs[:, 1].mean().cpu().numpy())
        
        deep_shap_sum = float(np.sum(shap_vals_class1[:-2])) 
        clin_shap_0   = float(shap_vals_class1[-2])
        clin_shap_1   = float(shap_vals_class1[-1])
        
        simp_shap  = np.array([deep_shap_sum, clin_shap_0, clin_shap_1])
        simp_feats = np.array(["Deep Features (Combined)", "CDR", "RDR"])
        
        c0        = float(X_clin[arr_idx][0])
        c1        = float(X_clin[arr_idx][1])
        simp_data = np.array([np.nan, c0, c1])

        explanation = shap.Explanation(values=simp_shap, base_values=base_value, data=simp_data, feature_names=simp_feats)

        # 1. Fetch Image Path and Strip Extension
        row_data = df.loc[df_idx]
        img_name = row_data["Original Image Name"]
        img_base = os.path.splitext(img_name)[0] 
        actual_label = int(row_data["Label"])
        
        condition_folder = "glaucoma" if actual_label == 1 else "normal"
        img_path = os.path.join(root_glauc if actual_label == 1 else root_normal, img_name)
        
        # 2. Build Dynamic Paths using Glob
        base_seg_dir = "data/2_g1020_origa_refuge/REFUGE/segmentedRetinographies"
        disc_matches = glob.glob(os.path.join(base_seg_dir, condition_folder, "opticDisc", f"*{img_base}*.*"))
        cup_matches  = glob.glob(os.path.join(base_seg_dir, condition_folder, "opticCup", f"*{img_base}*.*"))
        
        disc_path = disc_matches[0] if disc_matches else None
        cup_path  = cup_matches[0] if cup_matches else None

        # Print Output
        print(f"\n{'='*50}")
        print(f"CASE ANALYSIS: {case_type} | Image: {img_name}")
        print(f"Prediction : {int(case_data['pred_label'])} (Confidence: {case_data['prob']:.2%})")
        print(f"{'='*50}\n")

        # =====================================================================
        # NEW: SAVE PURE STANDALONE IMAGES (No Titles, No Borders)
        # =====================================================================
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            
            # Save Pure Original Fundus
            fig_raw = plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{case_type}_0a_pure_fundus.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig_raw) # Close immediately so it doesn't print to the notebook

            # Save Pure Overlay Image
            fig_overlay = plt.figure(figsize=(5, 5))
            plt.imshow(img)
            if disc_path and os.path.exists(disc_path):
                plt.imshow(get_transparent_mask(disc_path, color_rgb=(0, 1, 0), alpha=0.3))
            if cup_path and os.path.exists(cup_path):
                plt.imshow(get_transparent_mask(cup_path, color_rgb=(1, 0, 0), alpha=0.5))
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{case_type}_0b_pure_overlay.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig_overlay)
        # =====================================================================

        # 3. Plot Fundus and Segmentations Side-by-Side (For Notebook Viewing)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        if os.path.exists(img_path):
            ax1.imshow(img)
            ax2.imshow(img)
        else:
            ax1.text(0.5, 0.5, f"Image not found:\n{img_name}", ha='center', va='center')
            
        ax1.set_title(f"Original Fundus ({case_type})", fontsize=16, fontweight='bold')
        ax1.axis('off')

        masks_found = False
        if disc_path and os.path.exists(disc_path):
            ax2.imshow(get_transparent_mask(disc_path, color_rgb=(0, 1, 0), alpha=0.3))
            masks_found = True
        if cup_path and os.path.exists(cup_path):
            ax2.imshow(get_transparent_mask(cup_path, color_rgb=(1, 0, 0), alpha=0.5))
            masks_found = True
            
        if not masks_found:
            ax2.text(0.5, 0.5, f"Masks not found for:\n{img_base}", 
                     ha='center', va='center', color='white', backgroundcolor='red', fontsize=12)

        ax2.set_title("Optic Disc (Green) & Cup (Red)", fontsize=16, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{case_type}_1_image_comparison.png"), dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()

        # =====================================================================
        # 4. Draw a Custom, Publication-Ready SHAP Waterfall Plot (Arrow Style)
        # =====================================================================
        import matplotlib.patches as patches

        # 1. Set global paper fonts
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 18  
        })

        # 2. Setup exactly a 5x5 square figure
        fig, ax = plt.subplots(figsize=(5, 5))

        # 3. Prepare clean data
        features_raw = np.array(["Deep Features", "CDR", "RDR"])
        vals_raw = np.array([deep_shap_sum, clin_shap_0, clin_shap_1])
        clin_texts_raw = np.array(["\n(Combined)", f"({c0:.3f})", f"({c1:.3f})"])
        
        # Sort by absolute value
        sort_inds = np.argsort(np.abs(vals_raw))
        vals = vals_raw[sort_inds]
        features = features_raw[sort_inds]
        clin_texts = clin_texts_raw[sort_inds]
        
        y_labels = [f"{f} = {c}" for f, c in zip(features, clin_texts)]
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in vals]

        # 4. Calculate actual prediction steps for waterfall logic
        starts = []
        current_val = base_value
        for v in vals:
            starts.append(current_val)
            current_val += v
            
        final_value = current_val 

        # 5. DRAW ARROW BARS (Polygons instead of rectangles)
        h = 0.6 # Bar height
        y_pos = np.arange(len(vals))
        
        # Set Y-axis labels manually since we aren't using ax.barh
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)

        # Calculate a dynamic head length for the arrows (approx 3% of total width)
        min_x = min(min(starts), min(starts + vals))
        max_x = max(max(starts), max(starts + vals))
        x_range = max_x - min_x
        base_head_length = max(x_range * 0.03, 0.001)

        for i, (v, start, color) in enumerate(zip(vals, starts, colors)):
            end = start + v
            # Positive value: arrow points right
            if v > 0:
                hl = min(base_head_length, v) # Prevents head from being longer than bar
                verts = [
                    (start, i - h/2),       # Bottom left
                    (end - hl, i - h/2),    # Bottom right (start of arrow head)
                    (end, i),               # Arrow tip
                    (end - hl, i + h/2),    # Top right (start of arrow head)
                    (start, i + h/2)        # Top left
                ]
            # Negative value: arrow points left
            else:
                hl = min(base_head_length, -v) 
                verts = [
                    (start, i - h/2),       # Bottom right
                    (end + hl, i - h/2),    # Bottom left (start of arrow head)
                    (end, i),               # Arrow tip
                    (end + hl, i + h/2),    # Top left (start of arrow head)
                    (start, i + h/2)        # Top right
                ]
            
            # Add the custom shape to the plot
            poly = patches.Polygon(verts, facecolor=color, edgecolor='none')
            ax.add_patch(poly)

        # 6. Set limits manually because custom patches don't auto-scale the axes
        ax.set_ylim(-1, len(vals))
        ax.set_xlim(min_x - (x_range * 0.05), max_x + (x_range * 0.25))

        # 7. Add E[f(X)] at the bottom and f(x) at the top
        ax.axvline(base_value, #color='gray', 
                   linestyle=':', linewidth=1.5, zorder=0)
        ax.text(base_value, -0.7, f"$E[f(X)]$ = {base_value:.3f}", ha='center', va='top', fontsize=20, color='gray')
        
        ax.axvline(final_value, color='gray', linestyle=':', linewidth=1.5, zorder=0)
        ax.text(final_value, len(vals) - 0.3, f"$f(x)$ = {final_value:.3f}", ha='center', va='bottom', fontsize=20, color='black', fontweight='bold')

        # 8. Add the SHAP values (+0.16) safely to the RIGHT side of the arrows
        offset = max(x_range * 0.02, 0.001)
        for i, (v, start) in enumerate(zip(vals, starts)):
            right_edge = (start + v) if v > 0 else start
            ax.text(right_edge + offset, i, f"{v:+.2f}", va='center', ha='left', 
                    fontsize=20, color=colors[i], fontweight='bold')

        # 9. Formatting the plot clean-up
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0) 

        # 10. Lock the margins internally
        plt.subplots_adjust(left=0.35, right=0.90, top=0.85, bottom=0.15)
        
        # Save perfectly as 5x5
        plt.savefig(
            os.path.join(save_dir, f"{case_type}_2_custom_shap.png"), 
            # dpi=300, 
            # facecolor='white'
        )
        plt.show()
        plt.close()
        
        
    print(f"\n✅ All SHAP plots generated and saved to '{save_dir}' directory.")
    return selected_cases


def get_transparent_mask(mask_path, color_rgb, alpha=0.5):
    """Converts a binary mask into a colored RGBA image with a transparent background."""
    mask = mpimg.imread(mask_path)
    if mask.ndim == 3: 
        mask = mask.mean(axis=2) # Convert to grayscale if RGB
        
    mask_bool = (mask > 0.1).astype(float) # Binarize (ignores minor compression artifacts)
    
    # Create a blank RGBA image
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
    rgba[..., 0] = color_rgb[0] # Red channel
    rgba[..., 1] = color_rgb[1] # Green channel
    rgba[..., 2] = color_rgb[2] # Blue channel
    rgba[..., 3] = mask_bool * alpha # Only the mask area gets opacity, background is invisible
    return rgba