"""
Deepfake inference using the Hugging Face SigLip model:
  prithivMLmods/deepfake-detector-model-v1

Label space (from model config):
  id2label = {0: "fake", 1: "real"}

The model is loaded once at startup and reused for every request.
"""

import cv2
import io
import numpy as np
from PIL import Image, ExifTags
from scipy.fftpack import dct
from transformers import AutoImageProcessor, SiglipForImageClassification
import torch

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from .runtime_config import MODEL_NAME, MODEL_VERSION, LABEL_MAPPING_NOTE


class DeepfakeDetector:
    """
    Wraps the Hugging Face SigLip deepfake detection model.

    Inference pipeline per image:
      1. Detect and crop the largest face (OpenCV Haar Cascade).
      2. Run the HF model on both the face crop (60%) and full image (40%).
      3. Blend probabilities and apply forensic signal adjustments.
      4. Return a structured result dict.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_error = None

        # Label indices: class 0 = fake, class 1 = real (per HF model config)
        self.class_map = {"REAL": 1, "FAKE": 0}

        # Load the Hugging Face SigLip model
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = SiglipForImageClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()

            # Confirm label mapping from model config (defensive check)
            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                normalized = {int(k): str(v).strip().upper() for k, v in id2label.items()}
                real_idx = next((i for i, lbl in normalized.items() if lbl == "REAL"), None)
                fake_idx = next((i for i, lbl in normalized.items() if lbl == "FAKE"), None)
                if real_idx is not None and fake_idx is not None:
                    self.class_map = {"REAL": real_idx, "FAKE": fake_idx}

            self.model_loaded = True
            print(
                f"[DeepfakeDetector] Loaded '{self.model_name}' on {self.device} "
                f"| class_map={self.class_map}"
            )
        except Exception as exc:
            self.load_error = str(exc)
            print(f"[DeepfakeDetector] ERROR: Failed to load HF model '{self.model_name}': {exc}")

        # OpenCV face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # MediaPipe face mesh (optional, currently unused in scoring)
        self.mp_face_mesh = (
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, refine_landmarks=True
            )
            if MEDIAPIPE_AVAILABLE
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def metadata(self) -> dict:
        """Returns model metadata for the /health and /model-info endpoints."""
        return {
            "model_loaded": self.model_loaded,
            "model_name": self.model_name,
            "model_version": MODEL_VERSION,
            "device": str(self.device),
            "class_map": self.class_map,
            "label_mapping_note": LABEL_MAPPING_NOTE,
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "load_error": self.load_error,
        }

    def predict_image(self, image_bytes: bytes) -> dict:
        """
        Run deepfake detection on raw image bytes.

        Returns a dict with keys:
          success, prediction, authenticity_score, confidence, risk_level,
          real_probability, fake_probability, neural_real_probability,
          neural_fake_probability, model_version, class_map, heatmap_base64,
          heatmap_error, face_box, forensic_analysis
        """
        try:
            if not self.model_loaded:
                raise RuntimeError(
                    f"Model not loaded ('{self.model_name}'). "
                    f"Error: {self.load_error}"
                )

            # Decode bytes → OpenCV BGR
            np_img = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("Could not decode image. Ensure the file is a valid image.")

            # Face detection and crop
            face_img, face_box = self._detect_face(img_bgr)
            if face_img is None:
                # No face found — use the full image
                face_img = img_bgr
                face_box = (0, 0, img_bgr.shape[1], img_bgr.shape[0])

            # Convert BGR → RGB PIL images
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            full_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            # Run model on face crop and full image, then blend (60/40)
            face_probs = self._run_model(face_pil)
            full_probs = self._run_model(full_pil)
            blended = (0.6 * face_probs) + (0.4 * full_probs)

            real_idx = self.class_map["REAL"]
            fake_idx = self.class_map["FAKE"]
            neural_real = float(blended[real_idx].item())
            neural_fake = float(blended[fake_idx].item())

            # Forensic signal analysis
            metadata_signals = self._read_metadata_signals(image_bytes)
            forensic_signals = self._compute_forensic_signals(face_img)
            fusion = self._fuse_probabilities(neural_real, neural_fake, metadata_signals, forensic_signals)

            real_prob = fusion["real_probability"]
            fake_prob = fusion["fake_probability"]

            prediction = "REAL" if real_prob >= fake_prob else "FAKE"
            authenticity_score = real_prob * 100
            confidence = max(real_prob, fake_prob) * 100
            risk_level = self._risk_level(prediction, confidence, fake_prob, metadata_signals, forensic_signals)

            return {
                "success": True,
                "prediction": prediction,
                "authenticity_score": round(authenticity_score, 3),
                "confidence": round(confidence, 2),
                "risk_level": risk_level,
                "real_probability": round(real_prob * 100, 3),
                "fake_probability": round(fake_prob * 100, 3),
                "neural_real_probability": round(neural_real * 100, 3),
                "neural_fake_probability": round(neural_fake * 100, 3),
                "model_version": MODEL_VERSION,
                "class_map": self.class_map,
                "heatmap_base64": None,
                "heatmap_error": "Heatmap generation is not supported for the SigLip model.",
                "face_box": {
                    "x1": face_box[0],
                    "y1": face_box[1],
                    "x2": face_box[2],
                    "y2": face_box[3],
                },
                "forensic_analysis": {
                    "metadata": metadata_signals,
                    "signals": forensic_signals,
                    "forensic_adjustment": fusion["forensic_adjustment"],
                    "fusion_reasons": fusion["fusion_reasons"],
                },
            }

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def predict_video(self, video_path: str, fps_sample_rate: int = 1) -> dict:
        """
        Sample frames from a video file and aggregate per-frame predictions.

        Args:
            video_path: Path to the video file on disk.
            fps_sample_rate: How many frames per second to sample.

        Returns:
            Aggregated result dict (same shape as predict_image, minus heatmap).
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_step = max(1, int(fps / fps_sample_rate))

            scores, confidences, fake_probs = [], [], []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_step == 0:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        result = self.predict_image(buf.tobytes())
                        if result.get("success"):
                            scores.append(result["authenticity_score"])
                            confidences.append(result["confidence"])
                            fake_probs.append(result["fake_probability"] / 100.0)
                frame_idx += 1

            cap.release()

            if not scores:
                raise ValueError("No frames could be processed from the video.")

            avg_score = sum(scores) / len(scores)
            avg_conf = sum(confidences) / len(confidences)
            avg_fake = sum(fake_probs) / len(fake_probs)
            prediction = "FAKE" if avg_fake > 0.5 else "REAL"

            if prediction == "FAKE" and avg_conf > 80:
                risk = "HIGH"
            elif prediction == "FAKE" or avg_conf < 70:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            return {
                "success": True,
                "prediction": prediction,
                "authenticity_score": round(avg_score, 2),
                "confidence": round(avg_conf, 2),
                "risk_level": risk,
                "total_frames_analyzed": len(scores),
                "heatmap_base64_thumbnail": None,
            }

        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_model(self, pil_img: Image.Image) -> torch.Tensor:
        """Run the HF SigLip model on a PIL image and return softmax probabilities."""
        inputs = self.processor(images=pil_img.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        return probs

    def _detect_face(self, img_bgr: np.ndarray):
        """
        Detect the largest face in a BGR image using Haar Cascade.

        Returns:
            (face_crop_bgr, (x1, y1, x2, y2)) or (None, None) if no face found.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        if len(faces) == 0:
            return None, None

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        mx, my = int(w * 0.2), int(h * 0.2)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_bgr.shape[1], x + w + mx)
        y2 = min(img_bgr.shape[0], y + h + my)
        return img_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _read_metadata_signals(self, image_bytes: bytes) -> dict:
        """
        Inspect image metadata for AI-generation markers and camera EXIF tags.
        """
        signals = {
            "is_ai_metadata": False,
            "has_camera_exif": False,
            "ai_metadata_hits": [],
            "camera_exif_tags": [],
            "editor_metadata_hits": [],
            "metadata_error": None,
        }
        ai_terms = (
            "midjourney", "stable diffusion", "stable-diffusion", "dall-e",
            "dalle", "ai generated", "aigc", "comfyui", "automatic1111",
            "novelai", "invokeai", "sdxl", "flux", "firefly",
            "generative ai", "openai", "chatgpt",
        )
        editor_terms = ("photoshop", "lightroom", "gimp", "snapseed")
        camera_tag_names = {
            "Make", "Model", "LensModel", "LensMake", "ExposureTime",
            "FNumber", "ISOSpeedRatings", "PhotographicSensitivity",
            "FocalLength", "DateTimeOriginal",
        }

        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_raw:
                info_text = " ".join(
                    f"{k}={v}" for k, v in pil_raw.info.items()
                ).lower()
                signals["ai_metadata_hits"] = [t for t in ai_terms if t in info_text]
                signals["editor_metadata_hits"] = [t for t in editor_terms if t in info_text]
                signals["is_ai_metadata"] = bool(signals["ai_metadata_hits"])

                exif = pil_raw.getexif()
                if exif:
                    for key in exif:
                        tag = str(ExifTags.TAGS.get(key, key))
                        if tag in camera_tag_names:
                            signals["camera_exif_tags"].append(tag)
                    signals["has_camera_exif"] = bool(signals["camera_exif_tags"])
        except Exception as exc:
            signals["metadata_error"] = str(exc)

        return signals

    def _compute_forensic_signals(self, face_img: np.ndarray) -> dict:
        """
        Compute texture, frequency, edge, and ELA signals on the face crop.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        hf_ratio = float(self._dct_high_freq_ratio(gray_f))
        edges = cv2.Canny(gray, 100, 200)
        edge_fraction = float(np.mean(edges > 0))
        ela_variance = float(self._ela_variance(face_img))

        strong_flags = {
            "very_low_texture": laplacian_var < 30.0,
            "very_low_edge_density": edge_fraction < 0.0025,
            "very_low_high_frequency_energy": hf_ratio < 5e-6,
            "high_ela_variance": ela_variance > 100.0,
        }
        weak_flags = {
            "low_texture": laplacian_var < 60.0,
            "low_edge_density": edge_fraction < 0.006,
            "low_high_frequency_energy": hf_ratio < 1e-5,
            "medium_ela_variance": ela_variance > 50.0,
        }

        return {
            "laplacian_variance": round(laplacian_var, 6),
            "dct_high_frequency_ratio": round(hf_ratio, 10),
            "edge_fraction": round(edge_fraction, 6),
            "ela_variance": round(ela_variance, 6),
            "strong_flags": strong_flags,
            "weak_flags": weak_flags,
            "strong_signal_count": int(sum(strong_flags.values())),
            "weak_signal_count": int(sum(weak_flags.values())),
        }

    def _dct_high_freq_ratio(self, gray_f: np.ndarray) -> float:
        """
        Compute the ratio of high-frequency DCT energy to total energy.
        Generative models leave checkerboard artifacts in the high-frequency band.
        """
        dct_y = dct(dct(gray_f.T, norm="ortho").T, norm="ortho")
        power = np.abs(dct_y) ** 2
        h, w = power.shape
        high = np.sum(power[int(h * 0.5):, int(w * 0.5):])
        total = np.sum(power)
        return high / total if total > 0 else 0.0

    def _ela_variance(self, face_img: np.ndarray) -> float:
        """
        Error Level Analysis: re-compress to JPEG and measure difference variance.
        AI images often show higher ELA variance due to uniform compression response.
        """
        _, enc = cv2.imencode(".jpg", face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        diff = cv2.absdiff(face_img.astype(np.float32), dec.astype(np.float32))
        return float(np.var(diff))

    def _fuse_probabilities(
        self,
        neural_real: float,
        neural_fake: float,
        metadata_signals: dict,
        forensic_signals: dict,
    ) -> dict:
        """
        Adjust neural probabilities upward for fake based on forensic evidence.
        """
        fake_prob = float(np.clip(neural_fake, 0.001, 0.999))
        adjustment = 0.0
        reasons = []

        strong = forensic_signals["strong_flags"]
        weak = forensic_signals["weak_flags"]
        strong_count = forensic_signals["strong_signal_count"]

        if metadata_signals["is_ai_metadata"]:
            adjustment += 0.75
            reasons.append("explicit AI-generation metadata detected")

        if strong["very_low_texture"]:
            adjustment += 0.10
            reasons.append("very low facial texture variance")
        elif weak["low_texture"]:
            adjustment += 0.02

        if strong["very_low_edge_density"]:
            adjustment += 0.08
            reasons.append("very low edge density")
        elif weak["low_edge_density"]:
            adjustment += 0.02

        if strong["very_low_high_frequency_energy"]:
            adjustment += 0.06
            reasons.append("very low high-frequency DCT energy")
        elif weak["low_high_frequency_energy"]:
            adjustment += 0.01

        if strong["high_ela_variance"]:
            adjustment += 0.08
            reasons.append("high ELA variance (AI compression artifact)")
        elif weak["medium_ela_variance"]:
            adjustment += 0.02

        if not metadata_signals["has_camera_exif"] and strong_count >= 1:
            adjustment += 0.04
            reasons.append("no camera EXIF with synthetic artifact signals")

        fused_fake = min(0.99, fake_prob + adjustment)

        # Floor overrides for high-confidence forensic cases
        if metadata_signals["is_ai_metadata"]:
            fused_fake = max(fused_fake, 0.97)
        elif not metadata_signals["has_camera_exif"] and strong_count >= 3:
            fused_fake = max(fused_fake, 0.72)
        elif not metadata_signals["has_camera_exif"] and strong_count >= 2:
            fused_fake = max(fused_fake, 0.62)

        fused_real = 1.0 - fused_fake
        return {
            "real_probability": fused_real,
            "fake_probability": fused_fake,
            "forensic_adjustment": round(fused_fake - fake_prob, 6),
            "fusion_reasons": reasons,
        }

    def _risk_level(
        self,
        prediction: str,
        confidence: float,
        fake_probability: float,
        metadata_signals: dict,
        forensic_signals: dict,
    ) -> str:
        if prediction == "FAKE":
            if metadata_signals["is_ai_metadata"] or fake_probability >= 0.85:
                return "HIGH"
            return "MEDIUM"
        if confidence < 65 or forensic_signals["strong_signal_count"] >= 2:
            return "MEDIUM"
        return "LOW"
