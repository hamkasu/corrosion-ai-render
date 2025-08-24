# generate_pdf.py

from fpdf import FPDF
from PIL import Image
import os

# generate_pdf.py

from fpdf import FPDF
from PIL import Image
import os

def create_pdf_report(original_image_path, result_image_path, result_text, pdf_path, original_filename, comments=""):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 128)
    pdf.cell(0, 10, "Corrosion Inspection Report", ln=True, align='C')
    pdf.ln(5)

    # Add metadata
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

    # === Image Section ===
    y = pdf.get_y() + 5
    img_width = 90  # PDF image width in mm
    img_height = 80  # Approximate height

    # 1. Original Image
    try:
        pdf.image(original_image_path, x=10, y=y, w=img_width)
    except Exception as e:
        pdf.cell(0, 10, "❌ Error: Original image not found", ln=True)

    # 2. Detected + Markup Image
    try:
        # Check for markup
        markup_dir = os.path.join('static', 'results', 'markup')
        markup_path = os.path.join(markup_dir, f"markup_{os.path.basename(result_image_path)}")

        if os.path.exists(markup_path):
            # Create a temporary composite image
            base = Image.open(result_image_path).convert("RGBA")
            overlay = Image.open(markup_path).convert("RGBA")
            # Resize overlay to match base if needed
            overlay = overlay.resize(base.size, Image.Resampling.LANCZOS)
            # Composite
            combined = Image.alpha_composite(base, overlay)
            # Save to temp
            temp_dir = os.path.join('static', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"marked_{os.path.basename(result_image_path)}")
            combined.convert("RGB").save(temp_path, "JPEG")
            final_image_path = temp_path
        else:
            final_image_path = result_image_path  # No markup

        # Add to PDF
        pdf.image(final_image_path, x=105, y=y, w=img_width)

        # Labels
        pdf.set_y(y + img_height + 5)
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(90, 6, "Original Image", align='C')
        pdf.cell(90, 6, "Detected + Markup", align='C')

    except Exception as e:
        pdf.set_y(y)
        pdf.cell(90, img_height, "", border=1)  # Placeholder
        pdf.cell(90, img_height, "", border=1)
        pdf.set_y(y + img_height + 5)
        pdf.set_font("Arial", 'I', 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(90, 6, "Original", align='C')
        pdf.cell(90, 6, "Detected + Markup", align='C')
        print("❌ PDF Image Error:", str(e))

    pdf.output(pdf_path)