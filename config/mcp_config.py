from typing import Dict, Any
import os

class MCPConfig:
    # Model configurations
    MODEL_CONFIG = {
        "gemini": {
            "model_name": "gemini-2.0-flash",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        },
        "medical_nlp": {
            "symptom_classifier": "facebook/bart-large-mnli",
            "medical_bert": "pritamdeka/S-PubMedBert-MS-MARCO",
            "max_length": 512
        },
        "medical_imaging": {
            "vision_model": "microsoft/beit-base-patch16-224-pt22k-ft22k",
            "disease_detector": "nvidia/med-seg",
            "image_size": (224, 224)
        }
    }

    # Tool configurations
    TOOL_CONFIG = {
        "symptom_analysis": {
            "urgency_levels": ["emergency", "urgent", "non-urgent", "routine"],
            "confidence_threshold": 0.75
        },
        "treatment_planning": {
            "max_alternatives": 3,
            "follow_up_days": 14,
            "review_required": True
        },
        "health_metrics": {
            "vital_signs_tracking": True,
            "trend_analysis": True,
            "risk_assessment": True
        }
    }

    # API configurations
    API_CONFIG = {
        "rate_limit": 100,  # requests per minute
        "timeout": 30,      # seconds
        "retry_attempts": 3
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return cls.MODEL_CONFIG.get(model_name, {})

    @classmethod
    def get_tool_config(cls, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool"""
        return cls.TOOL_CONFIG.get(tool_name, {})

    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration"""
        return cls.API_CONFIG