import torch
import cv2
import numpy as np
import base64
import io
import os
from torchvision import transforms
from PIL import Image, ExifTags
from scipy.fftpack import dct
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from .model import DeepfakeResNetViT as DeepfakeResNet
from .runtime_config import DEPLOYED_CLASS_MAP, LABEL_MAPPING_NOTE, MODEL_VERSION

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

class DeepfakeDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            # Construct the absolute path so we can call inference.py from anywhere (like the Backend root)
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.pth")

        self.model_path = os.path.abspath(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the architecture
        self.model = DeepfakeResNet(pretrained=False).to(self.device)
        self.model.eval()

        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else 'models', exist_ok=True)

        self.class_map = {'REAL': 0, 'FAKE': 1}  # fallback default
        self.class_map_source = "default"
        self.checkpoint_label_source = None
        self.model_loaded = False

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = self._normalize_state_dict(checkpoint['state_dict'])
                self.model.load_state_dict(state_dict, strict=True)
                checkpoint_class_map = self._normalize_class_map(checkpoint.get('class_map', self.class_map))
                if DEPLOYED_CLASS_MAP:
                    self.class_map = self._normalize_class_map(DEPLOYED_CLASS_MAP)
                    self.class_map_source = "runtime_config.DEPLOYED_CLASS_MAP"
                else:
                    self.class_map = checkpoint_class_map
                    self.class_map_source = "checkpoint"
                self.checkpoint_label_source = checkpoint.get("label_source")
                self.model_loaded = True
                print(f"Loaded trained model checkpoint from {model_path} with class_map={self.class_map}")
            else:
                self.model.load_state_dict(self._normalize_state_dict(checkpoint), strict=True)
                self.model_loaded = True
                print(f"Loaded trained model weights from {model_path} (no class_map metadata). Using default class_map={self.class_map}")
        else:
            print(f"Warning: Model weights {model_path} not found. Inference will return a clear error instead of random predictions.")

        # Data transformations required for ResNet
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize Grad-CAM visualization
        self.cam = GradCAM(model=self.model, target_layers=self.model.get_target_layer())
        
        # Initialize OpenCV Face Detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) if MEDIAPIPE_AVAILABLE else None

    def _normalize_state_dict(self, state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        return {
            k.replace("model.", "", 1) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }

    def _normalize_class_map(self, class_map):
        if not isinstance(class_map, dict):
            return {'REAL': 0, 'FAKE': 1}

        label_to_index = {
            str(k).strip().upper(): int(v)
            for k, v in class_map.items()
            if str(k).strip().upper() in {"REAL", "FAKE"}
        }
        if 'REAL' in label_to_index and 'FAKE' in label_to_index:
            return {'REAL': label_to_index['REAL'], 'FAKE': label_to_index['FAKE']}

        index_to_label = {}
        for k, v in class_map.items():
            try:
                index_to_label[int(k)] = str(v).strip().upper()
            except (TypeError, ValueError):
                continue
        normalized = {label: index for index, label in index_to_label.items()}
        if 'REAL' in normalized and 'FAKE' in normalized:
            return {'REAL': normalized['REAL'], 'FAKE': normalized['FAKE']}

        return {'REAL': 0, 'FAKE': 1}

    def metadata(self):
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "model_version": MODEL_VERSION,
            "device": str(self.device),
            "class_map": self.class_map,
            "class_map_source": self.class_map_source,
            "checkpoint_label_source": self.checkpoint_label_source,
            "label_mapping_note": LABEL_MAPPING_NOTE,
            "mediapipe_available": MEDIAPIPE_AVAILABLE,
        }

    def _compute_landmark_score(self, pil_image):
        return 0.5  # Model trained with landmarks disabled
        if self.mp_face_mesh is None:
            return 0.5

        image_np = np.array(pil_image)
        results = self.mp_face_mesh.process(image_np)
        if not results.multi_face_landmarks:
            return 0.5

        face_landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y] for lm in face_landmarks])

        eye_left = coords[33]
        eye_right = coords[263]
        nose = coords[1]
        mouth = coords[0]
        eye_dist = np.linalg.norm(eye_left - eye_right)
        nose_eye = np.linalg.norm(nose - (eye_left + eye_right) / 2)
        mouth_nose = np.linalg.norm(mouth - nose)

        if eye_dist == 0 or nose_eye == 0 or mouth_nose == 0:
            return 0.5

        ratio1 = nose_eye / eye_dist
        ratio2 = mouth_nose / eye_dist
        ideal1, ideal2 = 0.35, 0.45
        score = 1.0 - (abs(ratio1 - ideal1) + abs(ratio2 - ideal2)) / 1.0
        score = np.clip((score + 1) / 2, 0.0, 1.0)
        return float(score)

    def _predict_probs(self, pil_img):
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device, non_blocking=True)
        landmark_score = torch.tensor([self._compute_landmark_score(pil_img)], dtype=torch.float32, device=self.device)

        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == 'cuda')):
            output = self.model(input_tensor, landmark_score)
            probs = torch.nn.functional.softmax(output, dim=1)[0]

        return input_tensor, probs

    def detect_frequency_artifacts(self, gray_img):
        """
        State-of-the-Art Generative Models (Stable Diffusion, Midjourney) 
        upsample images via decoders that leave unnatural repeating frequencies 
        (checkerboard artifacts) in the high-frequency spectrum.
        """
        # Compute 2D Discrete Cosine Transform
        dct_y = dct(dct(gray_img.T, norm='ortho').T, norm='ortho')
        
        # Calculate power spectrum
        power_spectrum = np.abs(dct_y) ** 2
        
        # Isolate High Frequency vs Low Frequency domains
        h, w = power_spectrum.shape
        high_freq_energy = np.sum(power_spectrum[int(h*0.5):, int(w*0.5):])
        total_energy = np.sum(power_spectrum)
        
        if total_energy == 0:
            return 0
            
        ratio = high_freq_energy / total_energy
        return ratio

    def _compute_ela_variance(self, face_img):
        """
        Error Level Analysis: Compress image to JPEG and measure difference variance.
        AI-generated images often have higher ELA variance due to compression artifacts.
        """
        # Encode to JPEG with quality 90
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded_img = cv2.imencode('.jpg', face_img, encode_param)
        
        # Decode back
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        # Compute difference
        diff = cv2.absdiff(face_img.astype(np.float32), decoded_img.astype(np.float32))
        
        # Variance of the difference
        variance = np.var(diff)
        return float(variance)

    def detect_face(self, img_bgr):
        """Detects the largest face in an image and crops it with a margin."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # detectMultiScale(image, scaleFactor, minNeighbors)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None, None
            
        # Select the largest face by bounding box area (w * h)
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        
        # Add a 20% margin to the bounding box
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_bgr.shape[1], x + w + margin_x)
        y2 = min(img_bgr.shape[0], y + h + margin_y)
        
        face_img = img_bgr[y1:y2, x1:x2]
        return face_img, (x1, y1, x2, y2)

    def _read_metadata_signals(self, image_bytes):
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
                    f"{key}={value}" for key, value in pil_raw.info.items()
                ).lower()
                signals["ai_metadata_hits"] = [term for term in ai_terms if term in info_text]
                signals["editor_metadata_hits"] = [term for term in editor_terms if term in info_text]
                signals["is_ai_metadata"] = bool(signals["ai_metadata_hits"])

                exif = pil_raw.getexif()
                if exif:
                    for key in exif:
                        tag_name = str(ExifTags.TAGS.get(key, key))
                        if tag_name in camera_tag_names:
                            signals["camera_exif_tags"].append(tag_name)
                    signals["has_camera_exif"] = bool(signals["camera_exif_tags"])
        except Exception as exc:
            signals["metadata_error"] = str(exc)

        return signals

    def _compute_forensic_signals(self, face_img):
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray_float = gray_face.astype(np.float32)
        laplacian_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
        hf_ratio = float(self.detect_frequency_artifacts(gray_float))
        edges = cv2.Canny(gray_face, 100, 200)
        edge_fraction = float(np.mean(edges > 0))

        # Error Level Analysis (ELA)
        ela_variance = self._compute_ela_variance(face_img)

        strong_flags = {
            "very_low_texture": laplacian_var < 30.0,
            "very_low_edge_density": edge_fraction < 0.0025,
            "very_low_high_frequency_energy": hf_ratio < 5e-6,
            "high_ela_variance": ela_variance > 100.0,  # AI images may have high ELA variance
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

    def _fuse_probabilities(self, neural_real_prob, neural_fake_prob, metadata_signals, forensic_signals):
        fake_prob = float(np.clip(neural_fake_prob, 0.001, 0.999))
        adjustment = 0.0
        reasons = []
        strong_count = forensic_signals["strong_signal_count"]
        weak_flags = forensic_signals["weak_flags"]
        strong_flags = forensic_signals["strong_flags"]

        if metadata_signals["is_ai_metadata"]:
            adjustment += 0.75
            reasons.append("explicit AI-generation metadata")

        if strong_flags["very_low_texture"]:
            adjustment += 0.10
            reasons.append("very low facial texture variance")
        elif weak_flags["low_texture"]:
            adjustment += 0.02

        if strong_flags["very_low_edge_density"]:
            adjustment += 0.08
            reasons.append("very low edge density")
        elif weak_flags["low_edge_density"]:
            adjustment += 0.02

        if strong_flags["very_low_high_frequency_energy"]:
            adjustment += 0.06
            reasons.append("very low high-frequency energy")
        elif weak_flags["low_high_frequency_energy"]:
            adjustment += 0.01

        if strong_flags["high_ela_variance"]:
            adjustment += 0.08
            reasons.append("high ELA variance (potential AI artifact)")
        elif weak_flags["medium_ela_variance"]:
            adjustment += 0.02

        if not metadata_signals["has_camera_exif"] and strong_count >= 1:
            adjustment += 0.04
            reasons.append("no camera EXIF with synthetic artifact signals")

        fused_fake_prob = min(0.99, fake_prob + adjustment)

        if metadata_signals["is_ai_metadata"]:
            fused_fake_prob = max(fused_fake_prob, 0.97)
        elif not metadata_signals["has_camera_exif"] and strong_count >= 3:
            fused_fake_prob = max(fused_fake_prob, 0.72)
        elif not metadata_signals["has_camera_exif"] and strong_count >= 2:
            fused_fake_prob = max(fused_fake_prob, 0.62)

        fused_real_prob = 1.0 - fused_fake_prob
        return {
            "real_probability": fused_real_prob,
            "fake_probability": fused_fake_prob,
            "forensic_adjustment": round(fused_fake_prob - fake_prob, 6),
            "fusion_reasons": reasons,
        }

    def _risk_level(self, prediction, confidence, fake_probability, metadata_signals, forensic_signals):
        if prediction == "FAKE":
            if metadata_signals["is_ai_metadata"] or fake_probability >= 0.85:
                return "HIGH"
            return "MEDIUM"

        if confidence < 65 or forensic_signals["strong_signal_count"] >= 2:
            return "MEDIUM"
        return "LOW"

    def _generate_heatmap(self, input_tensor, face_rgb):
        try:
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)[0, :]
            resized_face = cv2.resize(face_rgb, (224, 224)) / 255.0
            cam_image = show_cam_on_image(resized_face, grayscale_cam, use_rgb=True)
            cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', cam_image_bgr)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{heatmap_base64}", None
        except Exception as exc:
            return None, str(exc)

    def predict_image(self, image_bytes: bytes):
        """
        Accepts raw image bytes, runs face detection, deepfake inference,
        and generates Grad-CAM heatmap overlay.
        """
        try:
            if not self.model_loaded:
                raise RuntimeError(f"Model weights are not loaded. Expected checkpoint at {self.model_path}.")

            # Decode bytes to OpenCV BGR image
            np_img = np.frombuffer(image_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
            
            if img_bgr is None:
                raise ValueError("Could not decode image bytes.")

            # Detect and crop face
            face_img, face_box = self.detect_face(img_bgr)
            
            # If no face is found, process the entire image as fallback
            if face_img is None:
                face_img = img_bgr
                face_box = (0, 0, img_bgr.shape[1], img_bgr.shape[0])

            # Convert BGR (OpenCV) to RGB (Model/Pillow)
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            full_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            full_pil_img = Image.fromarray(full_rgb)

            # Training used uncropped dataset images with landmark scores.
            # Blend full-image and face-crop probabilities at inference to reduce deployment shift.
            input_tensor, face_probs = self._predict_probs(pil_img)
            _, full_probs = self._predict_probs(full_pil_img)
            probs = (0.6 * face_probs) + (0.4 * full_probs)

            # Resolve class indices from class_map (checkpoint metadata) to prevent reversal
            real_index = int(self.class_map.get('REAL', 0))
            fake_index = int(self.class_map.get('FAKE', 1))
            neural_real_prob = float(probs[real_index].item())
            neural_fake_prob = float(probs[fake_index].item())

            metadata_signals = self._read_metadata_signals(image_bytes)
            forensic_signals = self._compute_forensic_signals(face_img)
            fusion = self._fuse_probabilities(
                neural_real_prob,
                neural_fake_prob,
                metadata_signals,
                forensic_signals,
            )
            real_prob = fusion["real_probability"]
            fake_prob = fusion["fake_probability"]

            authenticity_score = real_prob * 100
            pred_index = real_index if real_prob >= fake_prob else fake_index
            prediction = "REAL" if pred_index == real_index else "FAKE"
            confidence = max(real_prob, fake_prob) * 100
            risk_level = self._risk_level(
                prediction,
                confidence,
                fake_prob,
                metadata_signals,
                forensic_signals,
            )

            heatmap_base64, heatmap_error = self._generate_heatmap(input_tensor, face_rgb)
            
            return {
                "success": True,
                "authenticity_score": round(authenticity_score, 3),
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "risk_level": risk_level,
                "real_probability": round(real_prob * 100, 3),
                "fake_probability": round(fake_prob * 100, 3),
                "neural_real_probability": round(neural_real_prob * 100, 3),
                "neural_fake_probability": round(neural_fake_prob * 100, 3),
                "model_version": MODEL_VERSION,
                "class_map": self.class_map,
                "class_map_source": self.class_map_source,
                "heatmap_base64": heatmap_base64,
                "heatmap_error": heatmap_error,
                "face_box": {"x1": face_box[0], "y1": face_box[1], "x2": face_box[2], "y2": face_box[3]},
                "forensic_analysis": {
                    "metadata": metadata_signals,
                    "signals": forensic_signals,
                    "forensic_adjustment": fusion["forensic_adjustment"],
                    "fusion_reasons": fusion["fusion_reasons"],
                },
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def predict_video(self, video_path: str, fps_sample_rate=1):
        """
        Extracts frames from a video path, samples 'fps_sample_rate' frames per second,
        runs image inference on each frame, and aggregates the predictions.
        (Note: Since fastapi handles video uploads by writing them temporarily,
        accepting a file path is usually best for video processing with cv2).
        """
        try:
            # OpenCV VideoCapture handles video files easily
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file.")
                
            # Grab original fps
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Determine step size to get fps_sample_rate frames per second
            if fps <= 0: fps = 30 # fallback
            frame_step = int(max(1, fps / fps_sample_rate))
            
            frame_scores = []
            frame_predictions = []
            frame_confidences = []
            frame_fake_probs = []
            heatmaps = []

            current_frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process only frames based on step logic
                if current_frame_idx % frame_step == 0:
                    # Encode frame back to bytes so we can use existing predict_image logic
                    # (In a highly optimized system we would extract this directly, but for now reuse is best)
                    success, buffer = cv2.imencode('.jpg', frame)
                    if success:
                        img_bytes = buffer.tobytes()
                        result = self.predict_image(img_bytes)
                        
                        if result.get("success"):
                            frame_scores.append(result["authenticity_score"])
                            frame_predictions.append(result["prediction"])
                            frame_confidences.append(result["confidence"])
                            frame_fake_probs.append(max(0.0, 1.0 - (result["authenticity_score"] / 100.0)))
                            # Maybe we just want one representative heatmap or all of them
                            heatmaps.append(result["heatmap_base64"])
                    
                current_frame_idx += 1

            cap.release()
            
            if not frame_scores:
                raise ValueError("Could not process any frames from the video.")

            # Aggregate results using simple averaging
            avg_score = sum(frame_scores) / len(frame_scores)
            avg_confidence = sum(frame_confidences) / len(frame_confidences)
            avg_fake_prob = sum(frame_fake_probs) / len(frame_fake_probs)
            
            final_prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
            
            if final_prediction == "FAKE" and avg_confidence > 80:
                risk_level = "HIGH"
            elif final_prediction == "FAKE" or (final_prediction == "REAL" and avg_confidence < 70):
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            return {
                "success": True,
                "authenticity_score": round(avg_score, 2),
                "prediction": final_prediction,
                "confidence": round(avg_confidence, 2),
                "risk_level": risk_level,
                "total_frames_analyzed": len(frame_scores),
                # Return the very first heatmap as a thumbnail for the report/frontend
                "heatmap_base64_thumbnail": heatmaps[0] if heatmaps else None 
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
