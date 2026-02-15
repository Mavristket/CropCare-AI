# Crop Detection System

A web application for plant disease detection and fertilizer prediction using AI/ML models.

## Features

- **Plant Disease Detection**: Upload crop images to detect diseases using a trained CNN model
- **Fertilizer Prediction**: Get fertilizer recommendations based on soil and environmental conditions
- **Modern Web Interface**: Responsive design with HTML, CSS, and JavaScript

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your trained model is saved at:
   `C:\Users\gaikw\OneDrive\Desktop\Final_model\plant_disease_model.h5`

3. Run the FastAPI server:
```bash
python main.py
```

4. Open your browser and navigate to:
   `http://localhost:8000`

## Project Structure

- `main.py` - FastAPI backend server
- `index.html` - Home page
- `crop_detection.html` - Disease detection page
- `fertilizer.html` - Fertilizer prediction page
- `contact.html` - Contact page
- `style.css` - Stylesheet
- `requirements.txt` - Python dependencies
- `plant_disease_model.h5` - Trained model (should be in this directory)

## API Endpoints

- `GET /` - Home page
- `GET /crop-detection` - Disease detection page
- `GET /fertilizer` - Fertilizer prediction page
- `GET /contact` - Contact page
- `POST /api/predict` - Predict disease from image
- `POST /api/fertilizer-predict` - Get fertilizer recommendation

## Contact

Email: gaikwadb602@gmail.com

## Notes

- The fertilizer prediction API integration is ready. Update the API endpoint and key in the fertilizer form when you have them.
- Make sure to update the CLASS_NAMES list in `main.py` to match your model's output classes.
