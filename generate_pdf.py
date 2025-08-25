from fpdf import FPDF
from PIL import Image
import os

def create_pdf_report(original_image_path, result_image_path, result_text, pdf_path, original_filename, comments=""):
    pdf = FPDF()
    pdf.add_page()
    
    # Set font with smaller size
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Corrosion Inspection Report", ln=True, align='C')
    pdf.ln(5)

    # Metadata
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, f"Image: {original_filename}", ln=True)
    pdf.cell(0, 6, f"Generated on: {os.path.basename(pdf_path)}", ln=True)
    pdf.ln(5)

    # Result text
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 6, "Detection Result:", ln=True)
    
    pdf.set_font("Arial", size=10)
    # Clean and wrap text
    clean_text = result_text.replace('<br>', '\n').replace('Corrosion Detected: PASS ', '')
    pdf.multi_cell(0, 5, clean_text)
    pdf.ln(3)

    # Comments
    if comments.strip():
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 6, "Comments & Observations:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 5, comments)
        pdf.ln(5)

    # Images (640x640 → fit in 90mm width)
    try:
        x1, y1, w1 = 10, pdf.get_y() + 10, 90
        x2, y2 = 105, pdf.get_y() + 10, 90
        
        # Check if image exists
        if os.path.exists(original_image_path):
            pdf.image(original_image_path, x=x1, y=y1, w=w1)
        else:
            pdf.set_xy(x1, y1)
            pdf.cell(w1, 40, "Original Image Not Found", border=1)

        if os.path.exists(result_image_path):
            pdf.image(result_image_path, x=x2, y=y2, w=w1)
        else:
            pdf.set_xy(x2, y2)
            pdf.cell(w1, 40, "Detected Image Not Found", border=1)

        # Labels
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(100, 100, 100)
        pdf.set_xy(x1, y2 + 85)
        pdf.cell(w1, 5, "Original Image", align='C')
        pdf.set_x(x2)
        pdf.cell(w1, 5, "Detected Corrosion", align='C')

    except Exception as e:
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, "Error embedding images.")
        print("PDF Image Error:", str(e))

    pdf.output(pdf_path)