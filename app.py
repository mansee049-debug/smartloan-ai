# app.py
import os
import datetime
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors 
from flask import jsonify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartloan_ai_ultra_secure_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smartloan.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Database Relational Architecture Maps ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    join_date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class LoanPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    applicant_name = db.Column(db.String(150), nullable=False)
    gender = db.Column(db.String(20))
    married = db.Column(db.String(10))
    dependents = db.Column(db.String(10))
    education = db.Column(db.String(50))
    self_employed = db.Column(db.String(10))
    income = db.Column(db.Float)
    co_income = db.Column(db.Float)
    loan_amt = db.Column(db.Float)
    term = db.Column(db.Float)
    credit_hist = db.Column(db.Float)
    property_area = db.Column(db.String(50))
    age = db.Column(db.Integer)
    
    # AI Output Engine Parameters
    loan_status = db.Column(db.String(10))
    approval_score = db.Column(db.Float)
    risk_level = db.Column(db.String(30))
    emi = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Safely load the pickled Random Forest Classifier Pipeline
try:
    with open('models/loan_model.pkl', 'rb') as f:
        ml_pipeline = pickle.load(f)
except FileNotFoundError:
    ml_pipeline = None

# Initialize persistent tables inside the SQLite schema
with app.app_context():
    db.create_all()

# --- Security Session Guard Decorator ---
def login_required(f):
    import functools
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Access Denied. Please authenticate your terminal to continue.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Application Routing Core Engine ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Security mismatch. Passwords do not align.', 'danger')
            return redirect(url_for('register'))
            
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            flash('This identity endpoint is already registered within our matrix.', 'warning')
            return redirect(url_for('register'))
            
        hashed_password = generate_password_hash(password, method='scrypt')
        new_user = User(name=name, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration completed successfully! Proceed to sign-in terminal.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Admin Backdoor Access Parameter Layer
        if email == 'admin@smartloan.com' and password == 'Admin@2026':
            session['admin_logged_in'] = True
            session['user_name'] = 'System Administrator'
            session['user_id'] = 0
            flash('Administrative Core Environment Established.', 'success')
            return redirect(url_for('admin_dashboard'))
            
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash(f'Authentication verified. Access granted to node: {user.name}', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid identity parameters. Core authorization failed.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Terminal disconnected. Session tokens successfully cleared.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    predictions = LoanPrediction.query.filter_by(user_id=user_id).order_by(LoanPrediction.created_at.desc()).all()
    
    total = len(predictions)
    approved = len([p for p in predictions if p.loan_status == 'Approved'])
    app_rate = (approved / total * 100) if total > 0 else 0
    
    return render_template('dashboard.html', predictions=predictions, total=total, app_rate=app_rate)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        # फॉर्म से नाम उठाना (New Update)
        applicant_name = request.form.get('applicant_name') 
        
        gender = request.form.get('gender')
        married = request.form.get('married')
        dependents = request.form.get('dependents')
        education = request.form.get('education')
        self_employed = request.form.get('self_employed')
        income = float(request.form.get('income'))
        co_income = float(request.form.get('co_income'))
        loan_amt = float(request.form.get('loan_amt'))
        term = float(request.form.get('term'))
        credit_hist = float(request.form.get('credit_hist'))
        property_area = request.form.get('property_area')
        age = int(request.form.get('age'))
        
        # Amortized Standard EMI Equation
        r = 0.095 / 12  
        n = term
        p = loan_amt * 1000
        emi_calc = (p * r * ((1+r)**n)) / (((1+r)**n) - 1) if n > 0 else 0

        # Processing AI Inference Vector
        if ml_pipeline:
            input_df = pd.DataFrame([{
                'Gender': gender, 'Married': married, 'Dependents': dependents,
                'Education': education, 'Self_Employed': self_employed,
                'ApplicantIncome': income, 'CoapplicantIncome': co_income,
                'LoanAmount': loan_amt, 'Loan_Amount_Term': term,
                'Credit_History': credit_hist, 'Property_Area': property_area,
                'Applicant_Age': age
            }])
            
            for col, le in ml_pipeline['encoders'].items():
                input_df[col] = le.transform(input_df[col].astype(str))
                
            proba = ml_pipeline['model'].predict_proba(input_df)[0][1]
            status = 'Approved' if proba >= 0.5 else 'Rejected'
            score = int(proba * 100)
        else:
            score = 80 if credit_hist == 1.0 and (income + co_income) > (loan_amt * 10) else 20
            status = 'Approved' if score >= 50 else 'Rejected'
            
        risk = 'Low Risk' if score >= 75 else ('Medium Risk' if score >= 45 else 'High Risk')
        
        # Saved with the new applicant_name variable
        pred_record = LoanPrediction(
            user_id=session['user_id'], applicant_name=applicant_name,
            gender=gender, married=married, dependents=dependents, education=education,
            self_employed=self_employed, income=income, co_income=co_income,
            loan_amt=loan_amt, term=term, credit_hist=credit_hist,
            property_area=property_area, age=age, loan_status=status,
            approval_score=score, risk_level=risk, emi=emi_calc
        )
        db.session.add(pred_record)
        db.session.commit()
        
        return redirect(url_for('result', id=pred_record.id))
        
    return render_template('predict.html')

@app.route('/result/<int:id>')
@login_required
def result(id):
    res = LoanPrediction.query.get_or_404(id)
    return render_template('result.html', res=res)

@app.route('/analytics')
@login_required
def analytics():
    user_id = session['user_id']
    preds = LoanPrediction.query.filter_by(user_id=user_id).all()
    
    app_count = len([p for p in preds if p.loan_status == 'Approved'])
    rej_count = len([p for p in preds if p.loan_status == 'Rejected'])
    
    return render_template('analytics.html', app_count=app_count, rej_count=rej_count, total=len(preds))

@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        flash('Admin token absent. Re-authenticate credentials.', 'danger')
        return redirect(url_for('login'))
    users = User.query.filter(User.id > 0).all()
    preds = LoanPrediction.query.all()
    return render_template('admin.html', users=users, preds=preds)

@app.route('/delete/prediction/<int:id>')
@login_required
def delete_prediction(id):
    pred = LoanPrediction.query.get_or_404(id)
    # Verify processing boundaries
    if session['user_id'] != pred.user_id and not session.get('admin_logged_in'):
        flash('Authorization violation.', 'danger')
        return redirect(url_for('dashboard'))
        
    db.session.delete(pred)
    db.session.commit()
    flash('Log matrix record removed successfully.', 'success')
    return redirect(url_for('admin_dashboard' if session.get('admin_logged_in') else 'dashboard'))

@app.route('/report/pdf/<int:id>')
@login_required
def download_pdf(id):
    res = LoanPrediction.query.get_or_404(id)
    pdf_path = f"dataset/report_{id}.pdf"
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=22, textColor=colors.HexColor('#4facfe'), spaceAfter=15)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, spaceAfter=8, leading=14)
    
    story.append(Paragraph("SMARTLOAN AI - CREDIT EVALUATION COMPLIANCE REPORT", title_style))
    story.append(Spacer(1, 10))
    
    data = [
        [Paragraph("<b>Primary Account Holder:</b>", body_style), Paragraph(res.applicant_name, body_style)],
        [Paragraph("<b>Monthly Salary Stream ($):</b>", body_style), Paragraph(f"${res.income:,.2f}", body_style)],
        [Paragraph("<b>Principal Funding Requested:</b>", body_style), Paragraph(f"${res.loan_amt * 1000:,.2f}", body_style)],
        [Paragraph("<b>AI Processing Pipeline Output:</b>", body_style), Paragraph(res.loan_status, body_style)],
        [Paragraph("<b>Calculated Accuracy Score:</b>", body_style), Paragraph(f"{res.approval_score:.0f}% Match", body_style)],
        [Paragraph("<b>Categorized Risk Dynamic:</b>", body_style), Paragraph(res.risk_level, body_style)],
        [Paragraph("<b>Calculated Monthly Installment (EMI):</b>", body_style), Paragraph(f"${res.emi:,.2f} / Mo", body_style)]
    ]
    
    t = Table(data, colWidths=[220, 280])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fcfdfe')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 10),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t)
    
    doc.build(story)
    return send_file(pdf_path, as_attachment=True)


    # =========================
# SMARTLOAN AI CHATBOT
# =========================

@app.route("/chatbot", methods=["POST"])
def chatbot():

    if "user_id" not in session:
        return jsonify({"reply": "Please login first."})

    message = request.form.get("message", "").strip().lower()

    if not message:
        return jsonify({"reply": "Please type a message."})

    # Greetings
    if any(word in message for word in ["hi", "hello", "hey", "good morning", "good evening"]):
        reply = """👋 Hello! Welcome to SmartLoan AI.

I'm your Smart Banking Assistant.

I can help you with:

• Loan Eligibility
• EMI Calculator
• Required Documents
• Credit Score
• Risk Analysis
• Approval Score
• Banking Guidance
• Loan Recommendations

How can I assist you today?"""

    # EMI
    elif "emi" in message:
        reply = """💰 EMI (Equated Monthly Installment)

EMI is the fixed amount paid every month until the loan is completely repaid.

EMI depends on:

• Loan Amount
• Interest Rate
• Loan Tenure

Use the SmartLoan AI EMI Calculator available in this system to estimate your monthly EMI instantly."""

    # Eligibility
    elif any(word in message for word in ["eligible", "eligibility", "loan approved", "loan rejected"]):
        reply = """🏦 Loan Eligibility

Our AI model analyzes:

✔ Income
✔ Loan Amount
✔ Credit History
✔ Employment
✔ Education
✔ Property Area

Based on these parameters SmartLoan AI predicts the loan approval possibility."""

    # Documents
    elif any(word in message for word in ["document", "documents", "aadhaar", "pan"]):
        reply = """📄 Required Documents

• Aadhaar Card
• PAN Card
• Passport Photo
• Address Proof
• Income Proof
• Salary Slips
• Bank Statement
• Income Tax Return (if applicable)"""

    # Credit
    elif any(word in message for word in ["credit", "cibil", "credit score", "credit history"]):
        reply = """📈 Credit Score

A CIBIL score above 750 improves approval chances.

Tips:

✔ Pay EMIs on time
✔ Don't miss credit card payments
✔ Avoid multiple loan applications
✔ Keep credit utilization low"""

    # Risk
    elif "risk" in message:
        reply = """⚠ Risk Levels

🟢 Low Risk

🟡 Medium Risk

🔴 High Risk

Risk depends upon income, repayment capacity, loan amount and credit history."""

    # Approval
    elif any(word in message for word in ["approval", "score", "chance"]):
        reply = """✅ Approval Score

Higher score means:

✔ Better financial profile
✔ Better repayment ability
✔ Higher loan approval probability"""

    # Home Loan
    elif "home loan" in message:
        reply = """🏠 Home Loan

Home loans help purchase or build a house.

Benefits:

✔ Long tenure
✔ Lower interest
✔ Tax benefits"""

    # Car Loan
    elif "car loan" in message:
        reply = """🚗 Car Loan

Banks generally finance up to 80–90% of the vehicle value depending on eligibility."""

    # Personal Loan
    elif "personal loan" in message:
        reply = """💳 Personal Loan

Useful for:

• Medical Emergency
• Marriage
• Travel
• Education
• Home Renovation"""

    # Education Loan
    elif "education loan" in message:
        reply = """🎓 Education Loan

Education loans cover:

✔ Tuition Fees
✔ Hostel Fees
✔ Books
✔ Equipment"""

    # Business Loan
    elif "business loan" in message:
        reply = """🏢 Business Loan

Business loans help expand business, buy machinery and manage working capital."""

    # Interest
    elif "interest" in message:
        reply = """📊 Interest Rate

Interest depends on:

• Credit Score
• Income
• Loan Type
• Bank Policies
• Loan Tenure"""

    # Financial Health
    elif any(word in message for word in ["financial", "health", "finance"]):
        reply = """📈 Financial Health Tips

✔ Save regularly
✔ Control expenses
✔ Pay EMIs on time
✔ Avoid unnecessary debt
✔ Maintain an emergency fund"""

    # Recommendation
    elif any(word in message for word in ["recommend", "recommendation", "advice", "tips"]):
        reply = """💡 SmartLoan AI Recommendation

✔ Maintain good credit history
✔ Apply for an affordable loan amount
✔ Increase your monthly income
✔ Add a co-applicant
✔ Reduce existing liabilities"""

    # Thanks
    elif any(word in message for word in ["thanks", "thank you"]):
        reply = """😊 You're Welcome!

Thank you for using SmartLoan AI.

Have a wonderful day!"""

    # Default
    else:
        reply = """🤖 I couldn't understand your question.

You can ask about:

• EMI
• Loan Eligibility
• Credit Score
• CIBIL
• Documents
• Home Loan
• Car Loan
• Personal Loan
• Education Loan
• Business Loan
• Risk Analysis
• Approval Score
• Financial Health"""

    return jsonify({"reply": reply}) 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)