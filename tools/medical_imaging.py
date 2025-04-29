from transformers import pipeline
import torch
from PIL import Image
import numpy as np
from typing import Dict, Optional

class MedicalImaging:
    def __init__(self):
        self.vision_model = pipeline("image-classification",
                                   model="microsoft/beit-base-patch16-224-pt22k-ft22k")
        self.disease_detector = pipeline("image-classification",
                                       model="nvidia/med-seg")
        
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze medical images using multiple specialized models"""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            
            # Get general image classification
            classification = self.vision_model(image)
            
            # Get specialized disease detection
            disease_detection = self.disease_detector(image)
            
            # Combine results
            return {
                "classification": {
                    "label": classification[0]["label"],
                    "confidence": float(classification[0]["score"])
                },
                "disease_detection": {
                    "finding": disease_detection[0]["label"],
                    "probability": float(disease_detection[0]["score"])
                },
                "image_quality": self._assess_image_quality(image)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """Assess the quality of medical images"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Calculate basic quality metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        return {
            "brightness": float(brightness),
            "contrast": float(contrast),
            "resolution": image.size,
            "format": image.format
        }