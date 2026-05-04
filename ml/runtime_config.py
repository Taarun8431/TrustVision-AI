from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "ml" / "models" / "model.pth"
OUTPUT_DIR = ROOT_DIR / "output"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

MODEL_VERSION = "TrustVision DeepfakeResNet v1.2-calibrated"
CORRECTED_CLASS_MAP = {"REAL": 0, "FAKE": 1}

# The newly trained checkpoint has the correct dataset labels (REAL=0, FAKE=1).
# Setting DEPLOYED_CLASS_MAP to None ensures inference uses the checkpoint's native class map.
DEPLOYED_CLASS_MAP = None
LABEL_MAPPING_NOTE = (
    "Inference now natively uses the trained checkpoint's class map "
    "as the dataset labels have been properly corrected during retraining."
)
