import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def generate_pdf_report(user_data: dict, prediction_result: dict, recommendations: dict, username: str = "User") -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1E3A5F'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#1E3A5F'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=HexColor('#667eea'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#333333'),
        spaceAfter=8
    )
    
    elements = []
    
    elements.append(Paragraph("Diabetes Risk Assessment Report", title_style))
    elements.append(Paragraph(f"Generated for: {username}", styles['Normal']))
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    elements.append(Paragraph("Risk Assessment Summary", heading_style))
    
    risk_level = prediction_result.get('risk_level', 'Unknown')
    risk_probability = prediction_result.get('probability_diabetes', 0) * 100
    
    risk_colors = {
        'Low': '#38ef7d',
        'Moderate': '#F2C94C',
        'High': '#f45c43',
        'Very High': '#8E2DE2'
    }
    risk_color = risk_colors.get(risk_level, '#667eea')
    
    summary_data = [
        ['Risk Level', risk_level],
        ['Risk Probability', f'{risk_probability:.1f}%'],
        ['No Diabetes Probability', f'{(100 - risk_probability):.1f}%']
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#333333')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#dddddd'))
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))
    
    elements.append(Paragraph("Health Parameters", heading_style))
    
    bmi = user_data.get('BMI', 25)
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif bmi < 25:
        bmi_category = "Normal"
    elif bmi < 30:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    glucose = user_data.get('Glucose', 100)
    if glucose < 100:
        glucose_status = "Normal"
    elif glucose < 126:
        glucose_status = "Pre-diabetic Range"
    else:
        glucose_status = "Diabetic Range"
    
    health_data = [
        ['Parameter', 'Your Value', 'Status/Category'],
        ['Age', f"{user_data.get('Age', 'N/A')} years", '-'],
        ['BMI', f"{bmi:.1f}", bmi_category],
        ['Blood Glucose', f"{glucose} mg/dL", glucose_status],
        ['Blood Pressure (Diastolic)', f"{user_data.get('BloodPressure', 'N/A')} mmHg", '-'],
        ['Insulin Level', f"{user_data.get('Insulin', 'N/A')} μU/mL", '-'],
        ['Family History Score', f"{user_data.get('DiabetesPedigreeFunction', 'N/A'):.2f}", '-'],
    ]
    
    health_table = Table(health_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
    health_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1E3A5F')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ffffff')),
        ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#333333')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, HexColor('#dddddd'))
    ]))
    elements.append(health_table)
    elements.append(Spacer(1, 20))
    
    if recommendations:
        elements.append(Paragraph("Personalized Recommendations", heading_style))
        
        if recommendations.get('summary'):
            elements.append(Paragraph(recommendations['summary'], normal_style))
            elements.append(Spacer(1, 10))
        
        if recommendations.get('diet_recommendations'):
            elements.append(Paragraph("Diet Recommendations", subheading_style))
            for rec in recommendations['diet_recommendations'][:3]:
                title = rec.get('title', '')
                desc = rec.get('description', '')
                priority = rec.get('priority', 'medium').upper()
                elements.append(Paragraph(f"<b>[{priority}] {title}</b>: {desc}", normal_style))
        
        if recommendations.get('exercise_recommendations'):
            elements.append(Paragraph("Exercise Recommendations", subheading_style))
            for rec in recommendations['exercise_recommendations'][:3]:
                title = rec.get('title', '')
                desc = rec.get('description', '')
                priority = rec.get('priority', 'medium').upper()
                elements.append(Paragraph(f"<b>[{priority}] {title}</b>: {desc}", normal_style))
        
        if recommendations.get('lifestyle_recommendations'):
            elements.append(Paragraph("Lifestyle Recommendations", subheading_style))
            for rec in recommendations['lifestyle_recommendations'][:3]:
                title = rec.get('title', '')
                desc = rec.get('description', '')
                priority = rec.get('priority', 'medium').upper()
                elements.append(Paragraph(f"<b>[{priority}] {title}</b>: {desc}", normal_style))
        
        if recommendations.get('medical_advice'):
            elements.append(Paragraph("Medical Advice", subheading_style))
            for rec in recommendations['medical_advice'][:3]:
                title = rec.get('title', '')
                desc = rec.get('description', '')
                priority = rec.get('priority', 'medium').upper()
                elements.append(Paragraph(f"<b>[{priority}] {title}</b>: {desc}", normal_style))
        
        if recommendations.get('warning_signs'):
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Warning Signs to Watch", subheading_style))
            for sign in recommendations['warning_signs'][:5]:
                elements.append(Paragraph(f"• {sign}", normal_style))
    
    elements.append(Spacer(1, 30))
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=HexColor('#666666'),
        alignment=TA_CENTER,
        spaceBefore=20
    )
    
    elements.append(Paragraph(
        "<b>IMPORTANT DISCLAIMER</b><br/>"
        "This report is generated by an AI-powered health assessment tool for educational purposes only. "
        "It should not be used as a substitute for professional medical advice, diagnosis, or treatment. "
        "Always seek the advice of your physician or other qualified health provider with any questions "
        "you may have regarding a medical condition.",
        disclaimer_style
    ))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
