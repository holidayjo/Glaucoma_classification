from sched import scheduler
import pandas as pd
import time
import joblib
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
import utils.dataset as utils_dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from models.gcfn import GlaucomaChemicalFusionNetwork
from utils.utils import (set_seed, get_next_run_dir, load_dataframe, GCFNInferenceWrapper, PassthroughScaler, 
                         run_multi_model_inference, measure_inference_time, execute_timing_test, log_final_results)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. XGBoost Training Logic
# ==========================================
def train_xgboost_model_per_fold(X_train, y_train, pos_w_train, random_seed, max_depth, learning_rate):
    es_skf                           = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    es_train_indices, es_val_indices = next(es_skf.split(X_train, y_train))

    scaler_es         = StandardScaler().fit(X_train[es_train_indices])
    X_train_es_scaled = scaler_es.transform(X_train[es_train_indices])
    X_val_es_scaled   = scaler_es.transform(X_train[es_val_indices])

    temp_model        = XGBClassifier(max_depth=max_depth, n_estimators=5000, learning_rate=learning_rate, 
                                      early_stopping_rounds=10, random_state=random_seed, scale_pos_weight=pos_w_train)
    
    temp_model.fit(X_train_es_scaled, y_train[es_train_indices], eval_set=[(X_val_es_scaled, y_train[es_val_indices])], verbose=False)
    
    best_ntree   = temp_model.best_iteration + 1
    scaler_final = StandardScaler().fit(X_train)
    
    final_model  = XGBClassifier(max_depth=max_depth, n_estimators=best_ntree, learning_rate=learning_rate, 
                                 random_state=random_seed, scale_pos_weight=pos_w_train)
    final_model.fit(scaler_final.transform(X_train), y_train)
    
    return final_model, scaler_final, best_ntree

# ==========================================
# 2. GCFN Training Logic
# ==========================================
# def train_gcfn_model_per_fold(X_train_deep, X_train_clin, y_train, pos_w_train, random_seed, epochs=200, patience=25): # patence 25
def train_gcfn_model_per_fold(X_train_deep, X_train_clin, y_train, pos_w_train, random_seed, epochs=200, patience=35): # patence 25
    es_skf                           = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_seed)
    es_train_indices, es_val_indices = next(es_skf.split(X_train_deep, y_train))

    scaler_deep = StandardScaler().fit(X_train_deep[es_train_indices])
    scaler_clin = StandardScaler().fit(X_train_clin[es_train_indices])

    def to_tensor(X, scaler): 
        return torch.tensor(scaler.transform(X), dtype=torch.float32).to(DEVICE)
    
    y_to_tensor  = lambda y: torch.tensor(y, dtype=torch.long).to(DEVICE)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(to_tensor(X_train_deep[es_train_indices], scaler_deep), 
                                                                              to_tensor(X_train_clin[es_train_indices], scaler_clin), 
                                                                              y_to_tensor(y_train[es_train_indices])), 
                                               batch_size=32, 
                                               shuffle=True)
    
    # model     = GlaucomaChemicalFusionNetwork(num_deep_features=X_train_deep.shape[1], num_classes=2).to(DEVICE)
    model     = GlaucomaChemicalFusionNetwork(num_deep_features=X_train_deep.shape[1], num_clinical_features=X_train_clin.shape[1], num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w_train], dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001) # 0.001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss, epochs_no_improve, best_epoch = float('inf'), 0, 0
    X_val_d, X_val_c, y_val                      = to_tensor(X_train_deep[es_val_indices], scaler_deep), to_tensor(X_train_clin[es_val_indices], scaler_clin), y_to_tensor(y_train[es_val_indices])

    history_train_loss = []
    history_val_loss   = []
    # print("epochs and patience =", epochs, patience)
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for bd, bc, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bd, bc), by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * bd.size(0) # Accumulate batch loss
        scheduler.step()
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        history_train_loss.append(epoch_train_loss)
        
        model.eval()
        with torch.no_grad(): 
            val_loss = criterion(model(X_val_d, X_val_c), y_val).item()
            history_val_loss.append(val_loss) # Save val loss

        if val_loss < best_val_loss:
            best_val_loss, epochs_no_improve, best_epoch = val_loss, 0, epoch
        elif (epochs_no_improve := epochs_no_improve + 1) >= patience: 
            break

    # Retrain on full dataset
    s_deep_fin, s_clin_fin = StandardScaler().fit(X_train_deep), StandardScaler().fit(X_train_clin)
    full_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(to_tensor(X_train_deep, s_deep_fin), 
                                                                             to_tensor(X_train_clin, s_clin_fin), 
                                                                             y_to_tensor(y_train)), 
                                              batch_size = 32, 
                                              shuffle    = True)
    
    # final_model     = GlaucomaChemicalFusionNetwork(num_deep_features=X_train_deep.shape[1], num_classes=2).to(DEVICE)
    final_model     = GlaucomaChemicalFusionNetwork(num_deep_features=X_train_deep.shape[1], num_clinical_features=X_train_clin.shape[1], num_classes=2).to(DEVICE)
    optimizer_final = torch.optim.AdamW(final_model.parameters(), lr=0.001)

    for _ in range(best_epoch + 1):
        final_model.train()
        for bd, bc, by in full_loader:
            optimizer_final.zero_grad()
            criterion(final_model(bd, bc), by).backward()
            optimizer_final.step()

    # return final_model, (s_deep_fin, s_clin_fin), best_epoch + 1
    return final_model, (s_deep_fin, s_clin_fin), best_epoch + 1, history_train_loss, history_val_loss

# ==========================================
# 3. Core Execution Loop
# ==========================================
def run_k_fold_cv(args, df, save_dir):
    set_seed(args.seed)

    is_clinical_only = "none" in [m.lower() for m in args.models]
    if args.exclude_clinical and args.classifier == "gcfn":
        raise ValueError("GCFN requires clinical features to fuse. Use --classifier xgboost for deep-only ablation.")  
    if is_clinical_only and args.classifier == "gcfn":
        raise ValueError("GCFN requires deep image features to fuse. Use --classifier xgboost for clinical-only ablation.")
    
    # Create a directory to save the weights for all 10 folds
    weights_dir = os.path.join(save_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # 1. Feature Extraction
    all_feats, base_idxs, y_final = [], None, None
    
    # Check if the user wants to skip image backbones (Clinical Only)
    if "none" not in [m.lower() for m in args.models]:
        for model_name in args.models:
            # feature, label, index <- In here, we also got the features from the backbone.
            X, y, idxs = utils_dataset.build_dataset(df, 
                                                     model_name, 
                                                     args.root_glauc, 
                                                     args.root_normal, 
                                                     DEVICE, 
                                                     args.weights if model_name == "custom_resnet" else None)
            if base_idxs is None: 
                base_idxs, y_final = np.arange(len(df)), y
            all_feats.append(X)
        X_deep = np.hstack(all_feats)
    else:
        # Clinical-Only Mode: Extract indices and labels directly from the CSV
        base_idxs = np.arange(len(df))
        y_final   = df["Label"].values
        X_deep    = np.zeros((len(df), 0)) # Creates an empty array for deep features

    # Conditionally load clinical features based on the new argument
    if not args.exclude_clinical:
        X_clin = df.loc[base_idxs, args.clinical_feats].values

    # 2. Cross Validation
    k_fold_results = [] 
    log_file_path  = os.path.join(save_dir, "results.txt")
    
    with open(log_file_path, "w") as f:
        f.write(f"=== CONFIG ===\nClassifier: {args.classifier}\nModels: {args.models}\nExclude Clinical: {args.exclude_clinical}\n\n")

        for fold_idx, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed).split(X_deep, y_final)):
            # set_seed(args.seed)
            X_tr_d, X_te_d = X_deep[train_idx], X_deep[test_idx]
            y_tr, y_te     = y_final[train_idx], y_final[test_idx]
            pos_w_train    = (y_tr == 0).sum() / (y_tr == 1).sum()

            if args.classifier == "xgboost":
                # Conditionally stack clinical data for XGBoost
                if args.exclude_clinical:
                    X_tr_xg, X_te_xg = X_tr_d, X_te_d
                else:
                    X_tr_c, X_te_c   = X_clin[train_idx], X_clin[test_idx]
                    X_tr_xg, X_te_xg = np.hstack([X_tr_d, X_tr_c]), np.hstack([X_te_d, X_te_c])

                model, scaler, best_ep = train_xgboost_model_per_fold(X_tr_xg, y_tr, pos_w_train, args.seed + fold_idx, args.max_depth, args.lr)
                probs_test             = model.predict_proba(scaler.transform(X_te_xg))[:, 1]
                
                # --- NOVELTY: SAVE XGBOOST MODEL AND SCALER ---
                fold_save_path = os.path.join(weights_dir, f"xgboost_fold_{fold_idx + 1}.joblib")
                joblib.dump({'model': model, 'scaler': scaler}, fold_save_path)
                
            else:
                # GCFN Logic (requires clinical data)
                X_tr_c, X_te_c = X_clin[train_idx], X_clin[test_idx]
                model, scalers, best_ep, train_losses, val_losses = train_gcfn_model_per_fold(X_tr_d, 
                                                                                              X_tr_c, 
                                                                                              y_tr, 
                                                                                              pos_w_train, 
                                                                                              args.seed + fold_idx)
                # 1. Generate and save the Plot
                plt.figure(figsize=(8, 6))
                plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
                plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
                plt.axvline(x=best_ep - 1, color='red', linestyle='--', label=f'Selected Epoch ({best_ep})')
                plt.title(f'GCFN Training vs. Validation Loss (Fold {fold_idx + 1})')
                plt.xlabel('Epochs')
                plt.ylabel('Cross Entropy Loss')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_path = os.path.join(save_dir, f"gcfn_learning_curve_fold_{fold_idx + 1}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"📈 Learning curve plot saved to: {plot_path}")

                # 2. Generate and save the Excel Log
                df_log = pd.DataFrame({
                    'Epoch': range(1, len(train_losses) + 1),
                    'Training_Loss': train_losses,
                    'Validation_Loss': val_losses
                })
                excel_path = os.path.join(save_dir, f"gcfn_training_log_fold_{fold_idx + 1}.xlsx")
                df_log.to_excel(excel_path, index=False)
                print(f"📊 Training log Excel saved to: {excel_path}")

                model.eval()
                
                with torch.no_grad():
                    probs_test = torch.softmax(model(
                        torch.tensor(scalers[0].transform(X_te_d), dtype=torch.float32).to(DEVICE), 
                        torch.tensor(scalers[1].transform(X_te_c), dtype=torch.float32).to(DEVICE)
                    ), dim=1)[:, 1].cpu().numpy()

                # --- SAVE GCFN MODEL AND SCALERS ---
                fold_save_path  = os.path.join(weights_dir, f"gcfn_fold_{fold_idx + 1}.pth")
                model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    'model_state_dict': model_cpu_state,
                    'scalers': scalers 
                }, fold_save_path)
                # print(f"✅ GCFN Fold {fold_idx + 1} weights saved to: {os.path.abspath(fold_save_path)}")
                
            p, r, t     = precision_recall_curve(y_te, probs_test)
            best_thresh = t[np.argmax(2 * p * r / (p + r + 1e-8))] if len(t) > 0 else 0.5
            preds       = (probs_test >= best_thresh).astype(int)

            metrics = {"accuracy": accuracy_score(y_te, preds), "precision": precision_score(y_te, preds, zero_division=0), "recall": recall_score(y_te, preds), "f1_score": f1_score(y_te, preds), "roc_auc": roc_auc_score(y_te, probs_test)}
            k_fold_results.append(metrics)
            
            msg = f"Fold {fold_idx + 1} | Best Ep/Tree: {best_ep} | AUC: {metrics['roc_auc']:.4f}"
            print(msg); f.write(msg + "\n")

    return k_fold_results, log_file_path, model, scaler if args.classifier == "xgboost" else scalers, X_deep.shape[1]


# ==========================================
# 3.5. Inference Execution Loop (Reproduce CV)
# ==========================================
def run_10_fold_inference(args, df):
    set_seed(args.seed)
    
    if args.exclude_clinical and args.classifier == "gcfn":
        raise ValueError("The GCFN classifier requires clinical features.")

    # 1. Feature Extraction (Now with Timing and Offline Support!)
    all_feats, base_idxs, y_final = [], None, None
    total_backbone_time_per_img = 0.0
    
    is_clinical_only = "none" in [m.lower() for m in args.models]
    
    if not is_clinical_only:
        for model_name in args.models:
            start_ext = time.perf_counter()
            
            # --- NEW: Use precomputed features if requested ---
            if getattr(args, 'use_precomputed', False):
                feat_path = f"data/precomputed_features/{args.dataset_name}_{model_name}.npz"
                if not os.path.exists(feat_path):
                    raise FileNotFoundError(f"Missing precomputed file: {feat_path}")
                print(f"⚡ Loading precomputed {model_name} features from disk...")
                data = np.load(feat_path)
                X, y, idxs = data['X'], data['y'], data['idxs']
            else:
                X, y, idxs = utils_dataset.build_dataset(df, model_name, args.root_glauc, args.root_normal, DEVICE, args.weights if model_name == "custom_resnet" else None)
            
            ext_time = time.perf_counter() - start_ext
            
            if base_idxs is None: 
                base_idxs = np.arange(len(X)) 
                y_final   = np.array(y)
            all_feats.append(X)
            
            if args.measure_time and not getattr(args, 'use_precomputed', False):
                time_per_img                 = ext_time / len(X)
                total_backbone_time_per_img += time_per_img
                print(f"  -> {model_name} processing time: {time_per_img*1000:.2f} ms/image")
        
        X_deep = np.hstack(all_feats)
        
    else:
        # Clinical Only Mode ---> X_deep is empty, but we still need base_idxs and y_final for the clinical features and labels
        base_idxs = np.arange(len(df))
        y_final   = df["Label"].values
        X_deep    = np.zeros((len(df), 0))
    
    # --- Dynamically extract clinical features using args ---
    if not args.exclude_clinical:
        X_clin = df.loc[base_idxs, args.clinical_feats].values # args.clinical_feats = ["CDR", "RDR"].

    # 2. Iterate through Folds and Load Weights
    k_fold_results = []
    
    skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_deep, y_final)):
        
        X_te_d = X_deep[test_idx]
        y_te   = y_final[test_idx]
        num_test_imgs = len(y_te) # Need this to normalize classifier time

        if args.classifier == "xgboost":
            if args.exclude_clinical: 
                X_te_xg = X_te_d
            else:                     
                X_te_c  = X_clin[test_idx]
                X_te_xg = np.hstack([X_te_d, X_te_c])

            weight_path = os.path.join(args.load_weights_dir, f"xgboost_fold_{fold_idx + 1}.joblib")
            loaded_data = joblib.load(weight_path)
            model, scaler = loaded_data['model'], loaded_data['scaler']
            
            X_te_xg_scaled = scaler.transform(X_te_xg)
            probs_test = model.predict_proba(X_te_xg_scaled)[:, 1]
            
            fold_time = 0.0
            if args.measure_time:
                fold_time = measure_inference_time(model, (X_te_xg_scaled,), model_type="xgboost")
                
        else:
            X_te_c = X_clin[test_idx]
            
            weight_path = os.path.join(args.load_weights_dir, f"gcfn_fold_{fold_idx + 1}.pth")
            loaded_data = torch.load(weight_path, map_location=DEVICE, weights_only=False)
            
            # model = GlaucomaChemicalFusionNetwork(num_deep_features=X_deep.shape[1], num_classes=2).to(DEVICE)
            model = GlaucomaChemicalFusionNetwork(num_deep_features=X_deep.shape[1], num_clinical_features=X_te_c.shape[1], num_classes=2).to(DEVICE)
            model.load_state_dict(loaded_data['model_state_dict'])
            model.eval()
            scalers = loaded_data['scalers']
            
            t_X_te_d = torch.tensor(scalers[0].transform(X_te_d), dtype=torch.float32).to(DEVICE)
            t_X_te_c = torch.tensor(scalers[1].transform(X_te_c), dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                probs_test = torch.softmax(model(t_X_te_d, t_X_te_c), dim=1)[:, 1].cpu().numpy()

            fold_time = 0.0
            if args.measure_time:
                fold_time = measure_inference_time(model, (t_X_te_d, t_X_te_c), model_type="pytorch", device=DEVICE)

        # 3. Calculate Total Time per Image
        total_time_per_img = 0.0
        if args.measure_time:
            classifier_time_per_img = fold_time / num_test_imgs
            total_time_per_img = total_backbone_time_per_img + classifier_time_per_img

        # 4. Calculate Metrics
        p, r, t = precision_recall_curve(y_te, probs_test)
        best_thresh = t[np.argmax(2 * p * r / (p + r + 1e-8))] if len(t) > 0 else 0.5
        preds = (probs_test >= best_thresh).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_te, preds), 
            "precision": precision_score(y_te, preds, zero_division=0), 
            "recall": recall_score(y_te, preds), 
            "f1_score": f1_score(y_te, preds), 
            "roc_auc": roc_auc_score(y_te, probs_test),
            "inference_time_per_img": total_time_per_img # Saved as seconds
        }
        k_fold_results.append(metrics)
        
        time_msg = f" | Total Time/Img: {total_time_per_img*1000:.2f} ms" if args.measure_time else ""
        print(f"Loaded Fold {fold_idx + 1} from {weight_path} | AUC: {metrics['roc_auc']:.4f}{time_msg}")

    return k_fold_results

# ==========================================
# 4. Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier",  type=str,   default="gcfn", choices=["xgboost", "gcfn"])
    parser.add_argument("--models",      nargs="+",  default=["custom_resnet", "convnext", "swin"])
    parser.add_argument("--weights",     type=str,   default="glaucolite_imagenet_epoch_75.pth")
    parser.add_argument("--csv_path",    type=str,   default="data/final_combined_results_rim_one.csv")
    parser.add_argument("--root_glauc",  type=str,   default="data/1_rim_one/RIM-ONE_DL_images/train_test_combined/glaucoma", help="Directory containing glaucoma images")
    parser.add_argument("--root_normal", type=str,   default="data/1_rim_one/RIM-ONE_DL_images/train_test_combined/normal",   help="Directory containing normal images")
    parser.add_argument("--k_fold",      type=int,   default=10)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--max_depth",   type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--conf_interval", type=float, default=0.95, help="Confidence level for interval calculation (e.g., 0.95 for 95% CI)")
    parser.add_argument("--exclude_clinical", action="store_true", help="Run using ONLY image features (ignores CDR and RDR)")
    parser.add_argument("--load_weights_dir", type=str, default=None, help="Path to the directory containing the 10 fold weight files. If provided, skips training and runs ensemble inference.")
    parser.add_argument("--measure_time", action="store_true", help="Measure and print inference time for each fold during ensemble inference.")
    parser.add_argument("--clinical_feats", nargs="+", default=["CDR", "RDR"], help="List of clinical features to include (e.g., CDR RDR)")
    parser.add_argument("--use_precomputed", action="store_true", help="Load features from disk instead of running PyTorch models.")
    parser.add_argument("--dataset_name", type=str, default="rim_one", choices=["rim_one", "origa", "refuge"], help="Used to find the correct precomputed files.")
    
    args = parser.parse_args()
    df   = load_dataframe(args.csv_path)

    # --- 1. RUN TRAINING OR INFERENCE ---
    if args.load_weights_dir is None:
        print("=== Starting K-Fold Cross Validation Training ===")
        save_dir = get_next_run_dir()
        results, log_path, raw_model, raw_scaler, num_deep = run_k_fold_cv(args, df, save_dir)
        # --- 2. RUN TIMING TEST (Abstracted cleanly into utils.py) ---
        if "none" not in [m.lower() for m in args.models]:
            execute_timing_test(args, df, raw_model, raw_scaler, num_deep, DEVICE)
        else:
            print("\n⏭️ Skipping Inference Timing Test (Clinical-Only Mode).")
    else:
        print(f"=== RUNNING ENSEMBLE INFERENCE FROM: {args.load_weights_dir} ===")
        results  = run_10_fold_inference(args, df)
        log_path = None

    # --- 3. CALCULATE AND PRINT FINAL STATS ---
    log_final_results(results, args, log_path)