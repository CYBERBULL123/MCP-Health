"""
Model Management Utility for Healthcare MCP Server
Helps manage AI model downloads and storage
"""

import os
import shutil
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch

class ModelManager:
    """Manages AI model downloads and storage for the MCP server"""
    
    MODELS = {
        'medical_nlp': {
            'name': 'facebook/bart-large-mnli',
            'size': '1.2GB',
            'required': True
        },
        'vision': {
            'name': 'microsoft/beit-base-patch16-224-pt22k-ft22k',
            'size': '2.0GB',
            'required': True
        },
        'medical_bert': {
            'name': 'pritamdeka/S-PubMedBert-MS-MARCO',
            'size': '1.5GB',
            'required': True
        }
    }

    def __init__(self):
        self.cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        self.downloaded_models = set()
        self._check_downloaded_models()

    def _check_downloaded_models(self):
        """Check which models are already downloaded"""
        for model_id in self.MODELS:
            model_name = self.MODELS[model_id]['name']
            if (self.cache_dir / model_name).exists():
                self.downloaded_models.add(model_id)

    def download_model(self, model_id: str):
        """Download a specific model"""
        if model_id not in self.MODELS:
            raise ValueError(f"Unknown model: {model_id}")

        model_info = self.MODELS[model_id]
        print(f"Downloading {model_id} ({model_info['size']})...")
        
        try:
            if model_id == 'medical_bert':
                SentenceTransformer(model_info['name'])
            else:
                AutoModel.from_pretrained(model_info['name'])
                AutoTokenizer.from_pretrained(model_info['name'])
            
            self.downloaded_models.add(model_id)
            print(f"Successfully downloaded {model_id}")
        except Exception as e:
            print(f"Error downloading {model_id}: {str(e)}")

    def download_all_required(self):
        """Download all required models"""
        for model_id, info in self.MODELS.items():
            if info['required'] and model_id not in self.downloaded_models:
                self.download_model(model_id)

    def clear_cache(self, model_id: str = None):
        """Clear model cache for specific or all models"""
        if model_id:
            if model_id not in self.MODELS:
                raise ValueError(f"Unknown model: {model_id}")
            model_path = self.cache_dir / self.MODELS[model_id]['name']
            if model_path.exists():
                shutil.rmtree(model_path)
                self.downloaded_models.remove(model_id)
                print(f"Cleared cache for {model_id}")
        else:
            shutil.rmtree(self.cache_dir)
            self.downloaded_models.clear()
            print("Cleared all model caches")

    def get_storage_info(self):
        """Get information about model storage usage"""
        total_size = 0
        model_sizes = {}
        
        for model_id in self.downloaded_models:
            model_path = self.cache_dir / self.MODELS[model_id]['name']
            size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
            model_sizes[model_id] = size
            total_size += size
        
        return {
            'total_size_gb': total_size / (1024**3),
            'models': model_sizes,
            'downloaded_models': list(self.downloaded_models),
            'missing_required': [
                m for m, info in self.MODELS.items() 
                if info['required'] and m not in self.downloaded_models
            ]
        }

if __name__ == "__main__":
    manager = ModelManager()
    
    # Example usage
    print("Checking model status...")
    info = manager.get_storage_info()
    
    print("\nDownloaded models:", info['downloaded_models'])
    print("Missing required models:", info['missing_required'])
    print(f"Total storage used: {info['total_size_gb']:.2f} GB")
    
    # Download missing required models
    if info['missing_required']:
        print("\nDownloading missing required models...")
        manager.download_all_required()