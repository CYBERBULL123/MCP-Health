import os
from server import app, mcp
from tools.model_manager import ModelManager
from tools.medical_nlp import MedicalNLP
from tools.medical_imaging import MedicalImaging
from tools.health_insights import HealthInsights
from config.mcp_config import MCPConfig
import json
import google.generativeai as genai
import argparse

def load_resources():
    """Load all necessary resources and configurations"""
    # Load medical ontology
    with open('resources/medical_ontology.json', 'r') as f:
        medical_ontology = json.load(f)
    
    # Load prompt templates
    with open('resources/prompt_templates.json', 'r') as f:
        prompt_templates = json.load(f)
    
    return {
        'medical_ontology': medical_ontology,
        'prompt_templates': prompt_templates
    }

def initialize_models(download_models: bool = True):
    """Initialize all AI models and tools"""
    print("\nChecking model status...")
    model_manager = ModelManager()
    info = model_manager.get_storage_info()
    
    if info['missing_required']:
        if download_models:
            print("\nDownloading required models (this may take 15-30 minutes)...")
            model_manager.download_all_required()
        else:
            print("\nWarning: Some required models are not downloaded.")
            print("Missing models:", info['missing_required'])
            print("Run with --download-models to download them.")
    
    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    print(f"\nTotal model storage used: {info['total_size_gb']:.2f} GB")
    
    # Initialize NLP tools
    nlp_tools = MedicalNLP()
    
    # Initialize imaging tools
    imaging_tools = MedicalImaging()
    
    # Initialize health insights with NLP model
    health_tools = HealthInsights(nlp_tools, nlp_tools.medical_bert)
    
    return {
        'nlp': nlp_tools,
        'imaging': imaging_tools,
        'health': health_tools
    }

def start_server(args):
    """Start the MCP server with all components"""
    print("Loading resources...")
    resources = load_resources()
    
    print("Initializing AI models...")
    models = initialize_models(args.download_models)
    
    print("Configuring MCP server...")
    config = MCPConfig()
    
    if args.manage_models:
        manager = ModelManager()
        if args.clear_cache:
            manager.clear_cache(args.clear_cache)
            return
        elif args.model_info:
            info = manager.get_storage_info()
            print("\nModel Storage Information:")
            print(f"Total storage used: {info['total_size_gb']:.2f} GB")
            print("Downloaded models:", info['downloaded_models'])
            print("Missing required models:", info['missing_required'])
            return
    
    print("\nStarting server...")
    app.run(debug=True, port=5000)
    mcp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Healthcare MCP Server Launcher')
    parser.add_argument('--download-models', action='store_true',
                      help='Download missing required models')
    parser.add_argument('--manage-models', action='store_true',
                      help='Enter model management mode')
    parser.add_argument('--clear-cache', type=str, metavar='MODEL_ID',
                      help='Clear cache for specific model or all models if no ID given')
    parser.add_argument('--model-info', action='store_true',
                      help='Show information about downloaded models and storage usage')
    
    args = parser.parse_args()
    start_server(args)