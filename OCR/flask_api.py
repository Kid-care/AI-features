import os
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
import google.generativeai as genai

app = Flask(__name__)

# Configure Google Generative AI
genai.configure(api_key='API_KEY')

def pdf_to_image(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    first_page_image = pages[0]
    image_path = 'page_1.png'
    first_page_image.save(image_path, 'PNG')
    return image_path

def crop_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    height, width, _ = image.shape
    new_width = width
    new_height = int(new_width * 3 / 4)
    if new_height > height:
        new_height = height
        new_width = int(new_height * 4 / 3)
    x1 = int((width - new_width) / 2)
    y1 = int((height - new_height) / 2)
    x2 = x1 + new_width
    y2 = y1 + new_height
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def adjust_image(image):
    image = image.astype(np.float32) / 255.0
    brightness = 1.0
    contrast = 1.1
    saturation = 1.2
    exposure = 1.0
    image = cv2.multiply(image, brightness)
    image = (image - 0.5) * contrast + 0.5
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * saturation
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = np.power(image, exposure)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    return image

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return bw_image

@app.route('/extract_text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        if file_path.lower().endswith('.pdf'):
            image_path = pdf_to_image(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = file_path
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': f'Failed to load image from {image_path}'}), 400

        cropped_image = crop_image(image_path)
        adjusted_image = adjust_image(cropped_image)
        bw_image = process_image(adjusted_image)

        black_list = '-c tessedit_char_blacklist=~,®@;%}'
        text = pytesseract.image_to_string(bw_image, config=black_list)

        output_text_file = 'output.txt'
        with open(output_text_file, 'w', encoding='utf-8') as file:
            file.write(text)

        # Use Google Generative AI to generate a report
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"""Write a report about the extracted text from medical analysis
        images in Arabic and tell me if the analysis is normal or not, based on this medical analysis: {text} give me responce of report in arabiccc but keep text in english """)

        report = response.text

        return jsonify({'report': report})

if __name__ == '__main__':
    app.run(debug=True)
