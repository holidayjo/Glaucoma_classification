import torch
import torch.nn as nn

class GlaucomaChemicalFusionNetwork(nn.Module):
    """
    Advanced GCFM Architecture (V3)
    Incorporates LayerNorm, deep clinical embedding, residual modulation,
    and a direct Clinical Skip Connection to the final classifier.
    """
    def __init__(self, num_deep_features=6400, num_clinical_features=2, embed_dim=512, num_classes=2):
        super().__init__()
        
        # 1. Visual Feature Compression
        self.visual_reduction = nn.Sequential(nn.Linear(num_deep_features, embed_dim*3),
                                              nn.LayerNorm(embed_dim*3), # Increased dimension for better gradient flow\
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.1),
                                              nn.Linear(embed_dim*3, embed_dim*2),
                                              nn.LayerNorm(embed_dim*2),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.2),
                                              nn.Linear(embed_dim*2, embed_dim),
                                              nn.LayerNorm(embed_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.4))
        
        # 2. Deep Clinical Embedding (Biomarker Embedding)
        self.clinical_embedding = nn.Sequential(nn.Linear(num_clinical_features, 32),
                                                nn.LayerNorm(32),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(32, 128),
                                                nn.LayerNorm(128),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.4))
         
        # 3. Modulation Generators (Excitation and Pathology Prior)
        self.gain_generator = nn.Linear(128, embed_dim)
        self.disp_generator = nn.Linear(128, embed_dim)
        
        # # ZERO-INITIALIZATION TRICK
        # nn.init.zeros_(self.gain_generator[1].weight)
        # nn.init.zeros_(self.gain_generator[1].bias)
        # nn.init.zeros_(self.disp_generator[1].weight)
        # nn.init.zeros_(self.disp_generator[1].bias)
        
        # 4. Multi-Stage Classification Head
        # INPUT CHANGE: 512 (Modulated Visual) + 128 (Direct Clinical Embedding) = 640
        self.classifier = nn.Sequential(nn.Linear(embed_dim + 128, 256), 
                                        nn.LayerNorm(256),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.4),
                                        nn.Linear(256, 64),
                                        nn.LayerNorm(64),
                                        nn.ReLU(inplace=True),            
                                        nn.Linear(64, num_classes))

    def forward(self, deep_features, clinical_data):
        # Step 1: Compress visual features (V)
        v_features = self.visual_reduction(deep_features)
        
        # Step 2: Build rich clinical context (C)
        c_embed = self.clinical_embedding(clinical_data)
        
        # Step 3: Generate clinical conditioning filters (E and P)
        gain = self.gain_generator(c_embed) # Excitation Vector
        disp = self.disp_generator(c_embed)   # Pathology Prior
        
        # Step 4: THE CHEMICAL FUSION (Modulation)
        modulated_features = v_features + (v_features * gain) + disp
        
        # Step 5: CLINICAL SKIP CONNECTION
        # Concatenate the modulated visual features with the direct clinical embedding
        final_fused_features = torch.cat((modulated_features, c_embed), dim=1)
        
        # Step 6: Final Prediction
        output = self.classifier(final_fused_features)
        return output
