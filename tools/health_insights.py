from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class HealthInsights:
    def __init__(self, nlp_model, embedding_model):
        self.nlp_model = nlp_model
        self.embedding_model = embedding_model
        
    def generate_treatment_plan(self, condition: str, patient_data: Dict) -> Dict:
        """Generate personalized treatment plans using medical context"""
        # Create medical context embedding
        context_embedding = self.embedding_model.encode(
            f"{condition} {patient_data.get('medical_history', '')}"
        )
        
        # Generate treatment recommendations
        treatment_prompt = f"""
        Condition: {condition}
        Patient Age: {patient_data.get('age')}
        Medical History: {patient_data.get('medical_history')}
        Current Medications: {patient_data.get('medications')}
        
        Generate a comprehensive treatment plan including:
        1. Primary treatment recommendations
        2. Alternative treatment options
        3. Lifestyle modifications
        4. Follow-up schedule
        5. Warning signs to monitor
        6. Expected outcomes
        """
        
        treatment_response = self.nlp_model.generate_text(treatment_prompt)
        
        return {
            "treatment_plan": treatment_response,
            "context_vector": context_embedding.tolist(),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def analyze_health_metrics(self, metrics: Dict) -> Dict:
        """Analyze patient health metrics and generate insights"""
        df = pd.DataFrame([metrics])
        
        # Calculate basic statistics
        stats = {
            "vital_signs": {
                "blood_pressure": self._analyze_blood_pressure(
                    metrics.get('systolic', 0),
                    metrics.get('diastolic', 0)
                ),
                "heart_rate": self._analyze_heart_rate(
                    metrics.get('heart_rate', 0)
                ),
                "temperature": self._analyze_temperature(
                    metrics.get('temperature', 0)
                )
            },
            "trends": self._calculate_trends(df),
            "risk_factors": self._identify_risk_factors(metrics)
        }
        
        return stats
    
    def _analyze_blood_pressure(self, systolic: float, diastolic: float) -> Dict:
        """Analyze blood pressure readings"""
        categories = {
            "normal": (systolic < 120 and diastolic < 80),
            "elevated": (120 <= systolic <= 129 and diastolic < 80),
            "stage1": (130 <= systolic <= 139 or 80 <= diastolic <= 89),
            "stage2": (systolic >= 140 or diastolic >= 90),
            "crisis": (systolic > 180 or diastolic > 120)
        }
        
        category = next(
            (k for k, v in categories.items() if v),
            "unknown"
        )
        
        return {
            "category": category,
            "systolic": systolic,
            "diastolic": diastolic,
            "needs_attention": category in ["stage2", "crisis"]
        }
    
    def _analyze_heart_rate(self, heart_rate: float) -> Dict:
        """Analyze heart rate measurements"""
        return {
            "value": heart_rate,
            "category": "normal" if 60 <= heart_rate <= 100 else "abnormal",
            "needs_attention": heart_rate < 60 or heart_rate > 100
        }
    
    def _analyze_temperature(self, temperature: float) -> Dict:
        """Analyze body temperature"""
        return {
            "value": temperature,
            "category": "normal" if 36.1 <= temperature <= 37.2 else "abnormal",
            "needs_attention": temperature > 38.0
        }
    
    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        """Calculate trends in health metrics"""
        if len(df) < 2:
            return {"error": "Insufficient data for trend analysis"}
            
        trends = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            if len(df[column].dropna()) >= 2:
                slope = np.polyfit(range(len(df)), df[column].values, 1)[0]
                trends[column] = {
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "slope": float(slope)
                }
        return trends
    
    def _identify_risk_factors(self, metrics: Dict) -> List[Dict]:
        """Identify potential health risk factors"""
        risk_factors = []
        
        # Check BMI
        if 'height' in metrics and 'weight' in metrics:
            bmi = metrics['weight'] / (metrics['height'] ** 2)
            if bmi >= 25:
                risk_factors.append({
                    "factor": "BMI",
                    "value": bmi,
                    "category": "overweight" if bmi < 30 else "obese",
                    "priority": "medium" if bmi < 30 else "high"
                })
        
        # Check blood pressure
        if 'systolic' in metrics and 'diastolic' in metrics:
            bp_analysis = self._analyze_blood_pressure(
                metrics['systolic'],
                metrics['diastolic']
            )
            if bp_analysis['category'] in ['stage1', 'stage2', 'crisis']:
                risk_factors.append({
                    "factor": "Blood Pressure",
                    "value": f"{metrics['systolic']}/{metrics['diastolic']}",
                    "category": bp_analysis['category'],
                    "priority": "high" if bp_analysis['needs_attention'] else "medium"
                })
        
        return risk_factors