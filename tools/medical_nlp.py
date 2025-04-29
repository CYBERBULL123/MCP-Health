from transformers import pipeline, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict, Optional

class MedicalNLP:
    def __init__(self):
        self.symptom_classifier = pipeline("zero-shot-classification",
                                         model="facebook/bart-large-mnli")
        self.medical_bert = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        
    def analyze_symptoms(self, symptoms: List[str], patient_context: Optional[Dict] = None) -> Dict:
        """Analyze symptoms using medical NLP models"""
        # Combine symptoms for analysis
        symptom_text = ", ".join(symptoms)
        context = f"Patient presents with: {symptom_text}"
        if patient_context:
            context += f"\nPatient Background: {patient_context.get('history', '')}"
        
        # Classify urgency
        urgency_labels = ["emergency", "urgent", "non-urgent", "routine"]
        urgency_result = self.symptom_classifier(context, candidate_labels=urgency_labels)
        
        # Generate embeddings for symptom similarity
        embeddings = self.medical_bert.encode(symptom_text)
        
        return {
            "urgency": {
                "level": urgency_result["labels"][0],
                "confidence": float(urgency_result["scores"][0])
            },
            "embedding": embeddings.tolist(),
            "context": context
        }
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between medical texts"""
        emb1 = self.medical_bert.encode(text1)
        emb2 = self.medical_bert.encode(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))