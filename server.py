"""
Healthcare MCP Server
===================

This is the main server module for the Healthcare Model Context Protocol (MCP) system.
It provides an intelligent healthcare platform that combines AI-powered analysis with
medical practice management.

Key Features:
------------
* AI-powered medical analysis
* Symptom analysis and diagnosis assistance
* Medical image processing
* Treatment recommendations
* Patient and doctor portals
* Appointment management
* Health insights generation

Components:
----------
* FastMCP Server - Handles AI model integration and medical analysis
* Flask Server - Web interface and API endpoints
* SQLAlchemy ORM - Database management
* Authentication System - User management and security

Usage:
-----
1. Start the server:
   ```
   python server.py
   ```
2. Access the web interface at http://localhost:5000
3. API documentation available at /docs/api_reference.md

Dependencies:
-----------
- Flask
- SQLAlchemy
- Google Gemini AI
- LangChain
- Transformers
- Sentence Transformers
- NumPy/Pandas
- PyTorch

For detailed documentation, see the /docs directory.
"""

from mcp.server.fastmcp import FastMCP, Context
import google.generativeai as genai
import os
from typing import Optional, List, Dict
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import pandas as pd
from pydantic import BaseModel
import torch

# Initialize Flask and core components
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///healthcare.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize the MCP server
mcp = FastMCP("Healthcare MCP Server")

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "your-api-key-here"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Database Models
class User(UserMixin, db.Model):
    """
    User model representing both patients and doctors.
    
    Attributes:
        id (int): Primary key
        username (str): Unique username
        email (str): Unique email address
        password_hash (str): Hashed password
        role (str): Either 'patient' or 'doctor'
    
    Usage:
        user = User(username='john_doe', email='john@example.com', role='patient')
        user.set_password('secure_password')
        db.session.add(user)
        db.session.commit()
    """
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False)  # 'patient' or 'doctor'
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Patient(db.Model):
    """
    Patient model for storing patient-specific information.
    
    Attributes:
        id (int): Primary key
        user_id (int): Foreign key to User model
        name (str): Patient's full name
        dob (Date): Date of birth
        medical_history (Text): Patient's medical history
        appointments (relationship): Related appointments
    
    Relationships:
        - One-to-One with User model
        - One-to-Many with Appointment model
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    medical_history = db.Column(db.Text)
    appointments = db.relationship('Appointment', backref='patient', lazy=True)

class Doctor(db.Model):
    """
    Doctor model for storing doctor-specific information.
    
    Attributes:
        id (int): Primary key
        user_id (int): Foreign key to User model
        name (str): Doctor's full name
        specialization (str): Medical specialization
        appointments (relationship): Related appointments
    
    Relationships:
        - One-to-One with User model
        - One-to-Many with Appointment model
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(100), nullable=False)
    appointments = db.relationship('Appointment', backref='doctor', lazy=True)

class Appointment(db.Model):
    """
    Appointment model for managing medical appointments.
    
    Attributes:
        id (int): Primary key
        patient_id (int): Foreign key to Patient model
        doctor_id (int): Foreign key to Doctor model
        datetime (DateTime): Appointment date and time
        status (str): Current status (scheduled, completed, cancelled)
        notes (Text): Appointment notes
    
    Relationships:
        - Many-to-One with Patient model
        - Many-to-One with Doctor model
    """
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=False)
    datetime = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='scheduled')  # scheduled, completed, cancelled
    notes = db.Column(db.Text)

@login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# MCP Tools
@mcp.tool()
async def generate_text(prompt: str, ctx: Context) -> str:
    """
    Generate medical text using Gemini LLM with enhanced context.
    
    Args:
        prompt (str): The input prompt for text generation
        ctx (Context): MCP context for progress tracking
    
    Returns:
        str: Generated text response
    
    Example:
        response = await generate_text(
            "Analyze symptoms: fever, cough, fatigue",
            context
        )
    """
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    await ctx.report_progress(0, 1)
    response = model.generate_content(
        prompt,
        safety_settings=safety_settings,
        generation_config={
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
        }
    )
    await ctx.report_progress(1, 1)
    return response.text

@mcp.tool()
async def analyze_symptoms(symptoms: list[str], ctx: Context, patient_data: Optional[dict] = None) -> Dict:
    """
    Analyze patient symptoms using advanced medical NLP models.
    
    Args:
        symptoms (list[str]): List of reported symptoms
        ctx (Context): MCP context for progress tracking
        patient_data (Optional[dict]): Additional patient context
    
    Returns:
        Dict: Analysis results containing:
            - analysis: Detailed symptom analysis
            - urgency: Urgency level assessment
            - confidence: Confidence score
    
    Example:
        result = await analyze_symptoms(
            ["fever", "cough", "fatigue"],
            context,
            {"age": 35, "history": "asthma"}
        )
    """
    # Initialize medical NLP model
    medical_classifier = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli")
    
    await ctx.report_progress(0, 3)
    
    # Combine symptoms with patient data for context
    context = f"Symptoms: {', '.join(symptoms)}"
    if patient_data:
        context += f"\nPatient Age: {patient_data.get('age', 'N/A')}"
        context += f"\nMedical History: {patient_data.get('history', 'N/A')}"
    
    await ctx.report_progress(1, 3)
    
    # Generate comprehensive analysis
    response = await generate_text(f"""
    As a medical professional, please analyze:
    {context}
    
    Provide:
    1. Potential diagnoses (with confidence levels)
    2. Recommended tests
    3. Urgency level
    4. Immediate actions needed
    """, ctx)
    
    await ctx.report_progress(2, 3)
    
    # Classify urgency
    urgency_result = medical_classifier(
        context,
        candidate_labels=["emergency", "urgent", "non-urgent", "routine"]
    )
    
    await ctx.report_progress(3, 3)
    
    return {
        "analysis": response,
        "urgency": urgency_result["labels"][0],
        "confidence": float(urgency_result["scores"][0])
    }

@mcp.tool()
async def get_treatment_suggestions(
    condition: str,
    patient_history: str,
    ctx: Context
) -> Dict:
    """
    Generate evidence-based treatment recommendations.
    
    Args:
        condition (str): Diagnosed medical condition
        patient_history (str): Patient's medical history
        ctx (Context): MCP context for progress tracking
    
    Returns:
        Dict: Treatment recommendations containing:
            - treatment_plan: Detailed treatment suggestions
            - condition_vector: Medical embedding
            - generated_at: Timestamp
    
    Example:
        plan = await get_treatment_suggestions(
            "Type 2 Diabetes",
            "History of hypertension, non-smoker",
            context
        )
    """
    await ctx.report_progress(0, 3)
    
    # Create embedding model for medical literature search
    embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    
    prompt_template = PromptTemplate(
        input_variables=["condition", "history"],
        template="""
        Based on evidence-based medicine, analyze:
        Condition: {condition}
        Patient History: {history}
        
        Provide:
        1. First-line treatment options
        2. Alternative treatments
        3. Lifestyle modifications
        4. Monitoring requirements
        5. Potential complications
        6. Prevention strategies
        """
    )
    
    await ctx.report_progress(1, 3)
    
    # Generate comprehensive treatment plan
    response = await generate_text(
        prompt_template.format(condition=condition, history=patient_history),
        ctx
    )
    
    await ctx.report_progress(2, 3)
    
    # Generate medical literature embeddings for relevance matching
    condition_embedding = embedder.encode(condition)
    
    await ctx.report_progress(3, 3)
    
    return {
        "treatment_plan": response,
        "condition_vector": condition_embedding.tolist(),
        "generated_at": datetime.utcnow().isoformat()
    }

@mcp.tool()
async def medical_image_analysis(image_path: str, ctx: Context) -> Dict:
    """
    Analyze medical images using computer vision models.
    
    Args:
        image_path (str): Path to the medical image
        ctx (Context): MCP context for progress tracking
    
    Returns:
        Dict: Analysis results containing:
            - classification: Image classification
            - analysis: Detailed findings
            - confidence: Confidence score
    
    Supported Image Types:
        - X-rays
        - MRI scans
        - CT scans
        - Ultrasound images
    
    Example:
        result = await medical_image_analysis(
            "path/to/xray.jpg",
            context
        )
    """
    medical_vision_model = pipeline("image-classification", 
                                  model="microsoft/beit-base-patch16-224-pt22k-ft22k")
    
    await ctx.report_progress(0, 2)
    
    try:
        result = medical_vision_model(image_path)
        await ctx.report_progress(1, 2)
        
        analysis = await generate_text(
            f"Analyze this medical image classification result: {result}",
            ctx
        )
        
        await ctx.report_progress(2, 2)
        
        return {
            "classification": result,
            "analysis": analysis,
            "confidence": float(result[0]["score"])
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def generate_health_insights(patient_data: Dict, ctx: Context) -> Dict:
    """
    Generate personalized health insights and recommendations.
    
    Args:
        patient_data (Dict): Patient health data including:
            - age
            - gender
            - medical_history
            - medications
            - vitals
        ctx (Context): MCP context for progress tracking
    
    Returns:
        Dict: Health insights containing:
            - insights: Personalized recommendations
            - generated_at: Timestamp
            - data_snapshot: Input data snapshot
    
    Example:
        insights = await generate_health_insights(
            {
                "age": 45,
                "gender": "F",
                "medical_history": "Hypertension",
                "medications": ["lisinopril"],
                "vitals": {"bp": "120/80", "hr": 72}
            },
            context
        )
    """
    await ctx.report_progress(0, 2)
    
    health_metrics = pd.DataFrame([patient_data])
    
    insights_prompt = f"""
    Analyze the following patient data:
    Age: {patient_data.get('age')}
    Gender: {patient_data.get('gender')}
    Medical History: {patient_data.get('medical_history')}
    Current Medications: {patient_data.get('medications')}
    Vital Signs: {patient_data.get('vitals')}
    
    Provide:
    1. Health status overview
    2. Risk factors
    3. Preventive recommendations
    4. Lifestyle optimization suggestions
    5. Medical monitoring recommendations
    """
    
    await ctx.report_progress(1, 2)
    
    response = await generate_text(insights_prompt, ctx)
    
    return {
        "insights": response,
        "generated_at": datetime.utcnow().isoformat(),
        "data_snapshot": health_metrics.to_dict(orient="records")[0]
    }

# Flask Routes
@app.route('/')
def index():
    """Home page route."""
    return render_template('index.html')

@app.route('/add_medical_note/<int:patient_id>', methods=['POST'])
@login_required
def add_medical_note(patient_id):
    """
    Add a new medical note to patient history.
    Only accessible by doctors.
    """
    if current_user.role != 'doctor':
        flash('Unauthorized access', 'danger')
        return redirect(url_for('dashboard'))
    
    patient = Patient.query.get_or_404(patient_id)
    note = request.form.get('note')
    
    if note:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        doctor = Doctor.query.filter_by(user_id=current_user.id).first()
        new_note = f"[{timestamp}] Dr. {doctor.name}: {note}"
        
        if patient.medical_history:
            patient.medical_history = new_note + "\n" + patient.medical_history
        else:
            patient.medical_history = new_note
        
        db.session.commit()
        flash('Medical note added successfully', 'success')
    
    return redirect(url_for('medical_history', patient_id=patient_id))

@app.route('/logout')
@login_required
def logout():
    """
    User logout route.
    Logs out the current user and redirects to login page.
    """
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    User login route.
    
    Methods:
        GET: Display login form
        POST: Process login attempt
    """
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration route.
    
    Methods:
        GET: Display registration form
        POST: Process registration
    """
    if request.method == 'POST':
        user = User(
            username=request.form['username'],
            email=request.form['email'],
            role=request.form['role']
        )
        user.set_password(request.form['password'])
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """
    User dashboard route.
    
    Displays:
        - Patient dashboard for patients
        - Doctor dashboard for doctors
    """
    if current_user.role == 'patient':
        patient = Patient.query.filter_by(user_id=current_user.id).first()
        appointments = patient.appointments if patient else []
        return render_template('patient_dashboard.html', appointments=appointments)
    else:
        doctor = Doctor.query.filter_by(user_id=current_user.id).first()
        appointments = doctor.appointments if doctor else []
        return render_template('doctor_dashboard.html', appointments=appointments)

@app.route('/appointment/new', methods=['GET', 'POST'])
@login_required
def new_appointment():
    """
    Create new appointment route.
    
    Methods:
        GET: Display appointment form
        POST: Schedule new appointment
    """
    if request.method == 'POST':
        appointment = Appointment(
            patient_id=request.form['patient_id'],
            doctor_id=request.form['doctor_id'],
            datetime=datetime.strptime(request.form['datetime'], '%Y-%m-%dT%H:%M'),
            notes=request.form['notes']
        )
        db.session.add(appointment)
        db.session.commit()
        return redirect(url_for('dashboard'))
    doctors = Doctor.query.all()
    return render_template('new_appointment.html', doctors=doctors)

@app.route('/medical_history')
@login_required
def medical_history():
    """
    View medical history route.
    
    Access:
        - Restricted to patients viewing their own history
    """
    if current_user.role != 'patient':
        return redirect(url_for('dashboard'))
    patient = Patient.query.filter_by(user_id=current_user.id).first()
    return render_template('medical_history.html', patient=patient)

if __name__ == "__main__":
    """
    Main entry point for the Healthcare MCP Server.
    
    Initialization:
        1. Creates database tables
        2. Starts Flask development server
        3. Runs MCP server
    """
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    mcp.run()