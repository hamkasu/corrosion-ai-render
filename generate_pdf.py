# generate_pdf.py
from fpdf import FPDF
from PIL import Image
import os

def create_pdf_report(original_image_path, result_image_path, result_text, pdf_path, original_filename, comments=""):
    # Your PDF generation code here
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Corrosion Inspection Report", ln=True, align='C')
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Image: {original_filename}", ln=True)
    pdf.cell(0, 8, f"Generated on: {os.path.basename(pdf_path)}", ln=True)
    pdf.ln(5)

    # Add detection result
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 8, "Detection Result:", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)
    clean_text = result_text.replace('<br>', '\n').replace('Corrosion Detected: PASS ', '').replace('Severity: ', '')
    pdf.multi_cell(0, 6, clean_text)
    pdf.ln(5)

    # Add Comments
    if comments.strip():
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 8, "Comments & Observations:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 6, comments)
        pdf.ln(5)

    # Add Images
    y = pdf.get_y() + 10
    try:
        pdf.image(original_image_path, x=10, y=y, w=90)
        pdf.image(result_image_path, x=105, y=y, w=90)

        pdf.set_y(y + 85)
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(90, 6, "Original Image", align='C')
        pdf.cell(90, 6, "Detected Corrosion", align='C')
    except Exception as e:
        pdf.cell(0, 10, "Error embedding images.", ln=True)

    pdf.output(pdf_path)