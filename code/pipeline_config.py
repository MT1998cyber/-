# -*- coding: utf-8 -*-
CENTERS = {
    "CENTER_1":    {"image": r"IMAGE_DIR",    "mask": r"MASK_DIR"},
    "CENTER_2":  {"image": r"IMAGE_DIR",  "mask": r"MASK_DIR"},
    "CENTER_3": {"image": r"IMAGE_DIR", "mask": r"MASK_DIR"},
    "CENTER_4":    {"image": r"IMAGE_DIR",    "mask": r"MASK_DIR"},
}

CLINICAL_CSV = r"TABLE_CSV_DIR" 
ID_COL = "ID"
SPLIT_COL = "group"                
SPLIT_VALUES = ["train", "test"]    
TRAIN_VALUES = ["train"]            
LABEL_COL = "label"                 

CLINICAL_FEATURES = [
    "first", "sex", "hemato_location", "stroke", "diabetes", "smoking", "SBP", "DBP", "GCS_score",
    "IVH", "Blend_sign", "Black_Hole_Sign", "Swirl_Sign", "Satellite_Sign", "island_sign",
    "Irregular_Shape", "Barras_Density", "V_Baseline_Hematoma_mL",
]
LOG_FEATURES = ["first", "V_Baseline_Hematoma_mL"]  
SIGNS = ["Blend_sign", "Black_Hole_Sign", "Swirl_Sign", "Satellite_Sign", "island_sign",
         "Irregular_Shape", "Barras_Density"]

WORK_ROOT = r"WORK_ROOT"
RUN_NAME = "RUN_NAME"        
VOXEL_YAML = None
REGIONAL_YAML = None


EXPECTED_SPACING_MM = (0.5, 0.5, 5.0)   
PARTITION_MODE = "none"      
RIM_THICKNESS_MM = 3.0        
RIM_HU_RANGE = (-15.0, 100.0) 
CORE_HU_RANGE = None          
VOXEL_FEATURE_SET = "hu_entropy"  
MAX_CORE_NAN_FRACTION = 0.005 
MAX_RIM_NAN_FRACTION = 0.05
N_WORKERS = 10                 
SKIP_EXISTING = True         
MAX_TRAIN_VOXELS_PER_PATIENT = None  
PCA_VARIANCE = None           
K_RANGE = list(range(2, 11))   
N_CLUSTERS = 2                 
WRITE_EXCEL = True             

SELECTION = {
    "MAX_MISSING_FRAC": 0.10,       
    "HARMONIZE": True,             
    "UNSEEN_CENTER_MODE": "self",   
    "P_THRESHOLD": 0.05,            
    "CORR_THRESHOLD": 0.9,         
    "N_FOLDS": 10,                 
    "N_C": 60,                      
    "LASSO_RULE": "1se",            
    "MAX_FEATURES": None,          
    "SEED": 0,
}

E3 = {
    "POOL_PATTERNS": [r"^(Hab\d+|Rim)_(original|log-sigma-[0-9-]+mm-3D)_firstorder_",
                      r"^(Hab\d+|Rim)_original_shape_", r"^FRAC_"],
    "HARMONIZE_PATTERN": r"_firstorder_",  
    "MODELS": {
        "C_full": ["C_full"],
        "C_noSigns": ["C_noSigns"],
        "Signs": ["Signs"],
        "H": ["H"],
        "C_noSigns+H": ["C_noSigns", "H"],
        "C_full+H": ["C_full", "H"],
    },
    "COMPARISONS": [
        ("C_noSigns+H", "C_full"),      
        ("C_noSigns+H", "C_noSigns"),   
        ("H", "Signs"),                
        ("C_full+H", "C_full"),         
        ("C_full", "C_noSigns"),        
    ],
    "PLOT_MODELS": ["C_noSigns", "C_full", "C_noSigns+H", "C_full+H"],
    "RUN_DEV_CV": True,                
    "N_SPLITS": 5, "N_REPEATS": 3,
    "N_BOOT": 2000,
    "N_CAL_BINS": 8,                    
    "SEED": 0,
}

E2 = {"N_SPLITS": 5, "N_REPEATS": 3, "LOCO_MODES": ["self", "none"],
      "INCLUDE_EXTERNAL_IN_LOCO": True, "FALLBACK_TOP": 5, "N_BOOT": 1000, "SEED": 0,
      "CLINICAL_SHIFT_VARS": ["V_Baseline_Hematoma_mL", "first", "GCS_score", "SBP"]}

E4 = {
    "ALGORITHMS": ["logistic", "svm", "random_forest", "extra_trees", "gbm", "xgboost", "lightgbm", "catboost"],
    "CLASS_WEIGHT_BALANCED": False,
    "HABITAT_CANDIDATES": "prefilter",  
    "MAX_HABITAT_CANDIDATES": 30,
    "INNER_CV": "stratified",           
    "INNER_SPLITS": 5, "INNER_REPEATS": 1,
    "MIN_FEATURES": 1,                 
    "STEP_FRACTION": 0.10,             
    "STEP_ONE_BELOW": 20,
    "N_FEATURES_RULE": "1se",           
    "AUC_TOLERANCE": 0.01,
    "BEST_ALGORITHM": "inner_cv",      
    "PERM_BACKGROUND": 50, "PERM_MAX_EVALS": 500,  
    "SHAP_SAVE_FOR": "all",            
    "FIT_ALL_CENTERS": True,         
    "RESUME": True,                     
    "SEED": 0,
}
