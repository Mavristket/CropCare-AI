from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import io
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import json
import time
import joblib

# Gmail SMTP for contact form (use env vars in production)
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "gaikwadb602@gmail.com")
SMTP_APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD", "arjg mhwh xfue uuaz")

app = FastAPI()

# region agent log
DEBUG_LOG_PATH = r"c:\Users\gaikw\OneDrive\Desktop\Final_model\.cursor\debug.log"

# Ensure .cursor directory exists
log_dir = os.path.dirname(DEBUG_LOG_PATH)

if log_dir:
    os.makedirs(log_dir, exist_ok=True)


def agent_debug_log(
    hypothesis_id: str,
    location: str,
    message: str,
    data: Optional[dict] = None,
    run_id: str = "initial",
) -> None:
    """
    Lightweight debug logger that appends NDJSON lines to the debug log file.
    Used for runtime debugging; do not remove until debugging is complete.
    """
    try:
        entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "runId": run_id,
            "hypothesisId": hypothesis_id,
        }
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Logging must never break the main flow, but print for debugging
        print(f"Debug log failed: {e}")
        pass


# Log module import to confirm we are running the instrumented code.
agent_debug_log(
    hypothesis_id="H0",
    location="main.py:module_import",
    message="main.py imported and FastAPI app created",
    data={"__file__": __file__, "cwd_at_import": os.getcwd()},
)
# endregion agent log


# Use absolute base directory so template paths work regardless of cwd.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files
app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/images', StaticFiles(directory='images'), name='images')

# Load the model
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f'Model loaded successfully from {MODEL_PATH}')
    else:
        print(f'Warning: Model file not found at {MODEL_PATH}')
except Exception as e:
    print(f'Error loading model: {e}')

# Fertilizer ML model and label encoders (trained from furtilizer.csv)
FERTILIZER_MODEL_PATH = os.path.join(BASE_DIR, "fertilizer_model.pkl")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
fertilizer_model = None
label_encoders = None

try:
    if os.path.exists(FERTILIZER_MODEL_PATH) and os.path.exists(LABEL_ENCODERS_PATH):
        fertilizer_model = joblib.load(FERTILIZER_MODEL_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        print(f'Fertilizer model and label encoders loaded from {BASE_DIR}')
    else:
        print(f'Warning: fertilizer_model.pkl or label_encoders.pkl not found in {BASE_DIR}')
except Exception as e:
    print(f'Error loading fertilizer model: {e}')

# Crop recommendation ML model (trained from pik.csv)
CROP_MODEL_PATH = os.path.join(BASE_DIR, "crop_model.pkl")
crop_model = None

# Crop label mapping (same order as used in training)
CROP_LABELS = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton',
    'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
    'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

try:
    if os.path.exists(CROP_MODEL_PATH):
        crop_model = joblib.load(CROP_MODEL_PATH)
        print(f'Crop model loaded from {CROP_MODEL_PATH}')
    else:
        print(f'Warning: crop_model.pkl not found in {BASE_DIR}')
except Exception as e:
    print(f'Error loading crop model: {e}')

# Class names (update these based on your model)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn___Cercospora_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca',
    'Grape___Leaf_blight',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites',
    'Tomato___Target_Spot',
    'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___healthy',
    'Unknown___Class_24',
    'Unknown___Class_25',
    'Unknown___Class_26',
    'Unknown___Class_27',
    'Unknown___Class_28',
    'Unknown___Class_29',
    'Unknown___Class_30',
    'Unknown___Class_31',
    'Unknown___Class_32',
    'Unknown___Class_33'
]

# Detailed class information
CLASS_DETAILS = {
    'Apple___Apple_scab': {
        'plant': 'Apple',
        'disease': 'Apple Scab',
        'description': 'Fungal disease causing dark, scaly lesions on leaves, fruit, and twigs',
        'symptoms': 'Dark brown to black spots on leaves, premature leaf drop, distorted fruit',
        'treatment': 'Apply fungicides, remove infected leaves, improve air circulation'
    },
    'Apple___Black_rot': {
        'plant': 'Apple',
        'disease': 'Black Rot',
        'description': 'Fungal disease affecting all parts of the apple tree',
        'symptoms': 'Brown spots on leaves, rotting fruit, cankers on branches',
        'treatment': 'Prune infected branches, apply fungicides, avoid overhead watering'
    },
    'Apple___Cedar_apple_rust': {
        'plant': 'Apple',
        'disease': 'Cedar Apple Rust',
        'description': 'Fungal disease requiring both apple and cedar trees to complete lifecycle',
        'symptoms': 'Yellow-orange spots on leaves, orange gelatinous horns on undersides',
        'treatment': 'Remove nearby cedar trees, apply fungicides during spring'
    },
    'Apple___healthy': {
        'plant': 'Apple',
        'disease': 'Healthy',
        'description': 'No disease detected - plant appears healthy',
        'symptoms': 'Normal green leaves, healthy fruit development',
        'treatment': 'Continue regular care and maintenance'
    },
    'Corn___Cercospora_leaf_spot': {
        'plant': 'Corn',
        'disease': 'Cercospora Leaf Spot',
        'description': 'Fungal disease causing gray leaf spots',
        'symptoms': 'Gray rectangular spots on leaves, leaf yellowing and death',
        'treatment': 'Use resistant varieties, apply fungicides, crop rotation'
    },
    'Corn___Common_rust': {
        'plant': 'Corn',
        'disease': 'Common Rust',
        'description': 'Fungal disease causing rust-colored pustules on leaves',
        'symptoms': 'Reddish-brown pustules on both leaf surfaces',
        'treatment': 'Use resistant varieties, apply fungicides if needed'
    },
    'Corn___Northern_Leaf_Blight': {
        'plant': 'Corn',
        'disease': 'Northern Leaf Blight',
        'description': 'Fungal disease causing cigar-shaped lesions',
        'symptoms': 'Long, elliptical gray-green lesions on leaves',
        'treatment': 'Use resistant varieties, crop rotation, fungicide application'
    },
    'Corn___healthy': {
        'plant': 'Corn',
        'disease': 'Healthy',
        'description': 'No disease detected - plant appears healthy',
        'symptoms': 'Normal green leaves, healthy growth',
        'treatment': 'Continue regular care and maintenance'
    },
    'Grape___Black_rot': {
        'plant': 'Grape',
        'disease': 'Black Rot',
        'description': 'Fungal disease causing black lesions on all plant parts',
        'symptoms': 'Black spots on leaves, rotting berries, black cankers',
        'treatment': 'Remove infected plant parts, apply fungicides, improve air circulation'
    },
    'Grape___Esca': {
        'plant': 'Grape',
        'disease': 'Esca',
        'description': 'Fungal disease causing internal wood decay',
        'symptoms': 'Leaf discoloration, wilting, white rot in wood',
        'treatment': 'Remove infected vines, avoid stress, proper pruning'
    },
    'Grape___Leaf_blight': {
        'plant': 'Grape',
        'disease': 'Leaf Blight',
        'description': 'Fungal disease causing brown spots on leaves',
        'symptoms': 'Brown circular spots on leaves, defoliation',
        'treatment': 'Apply fungicides, remove infected leaves, improve air circulation'
    },
    'Grape___healthy': {
        'plant': 'Grape',
        'disease': 'Healthy',
        'description': 'No disease detected - plant appears healthy',
        'symptoms': 'Normal green leaves, healthy fruit development',
        'treatment': 'Continue regular care and maintenance'
    },
    'Potato___Early_blight': {
        'plant': 'Potato',
        'disease': 'Early Blight',
        'description': 'Fungal disease causing dark spots on leaves',
        'symptoms': 'Dark brown spots with yellow halos on lower leaves',
        'treatment': 'Apply fungicides, crop rotation, avoid overhead watering'
    },
    'Potato___Late_blight': {
        'plant': 'Potato',
        'disease': 'Late Blight',
        'description': 'Devastating fungal disease causing rapid plant death',
        'symptoms': 'Water-soaked spots on leaves, white fungal growth on undersides',
        'treatment': 'Remove infected plants immediately, apply fungicides, use resistant varieties'
    },
    'Potato___healthy': {
        'plant': 'Potato',
        'disease': 'Healthy',
        'description': 'No disease detected - plant appears healthy',
        'symptoms': 'Normal green leaves, healthy tuber development',
        'treatment': 'Continue regular care and maintenance'
    },
    'Tomato___Bacterial_spot': {
        'plant': 'Tomato',
        'disease': 'Bacterial Spot',
        'description': 'Bacterial disease causing leaf spots and fruit lesions',
        'symptoms': 'Small dark spots on leaves, raised lesions on fruit',
        'treatment': 'Use disease-free seeds, copper-based bactericides, crop rotation'
    },
    'Tomato___Early_blight': {
        'plant': 'Tomato',
        'disease': 'Early Blight',
        'description': 'Fungal disease causing target-like spots on leaves',
        'symptoms': 'Dark spots with concentric rings on lower leaves',
        'treatment': 'Mulch around plants, apply fungicides, improve air circulation'
    },
    'Tomato___Late_blight': {
        'plant': 'Tomato',
        'disease': 'Late Blight',
        'description': 'Destructive fungal disease causing rapid decay',
        'symptoms': 'Irregular water-soaked spots, white fungal growth',
        'treatment': 'Remove infected plants, apply fungicides, avoid wet conditions'
    },
    'Tomato___Leaf_Mold': {
        'plant': 'Tomato',
        'disease': 'Leaf Mold',
        'description': 'Fungal disease causing yellow spots on upper leaves',
        'symptoms': 'Pale yellow spots on upper leaf surfaces, olive-green mold below',
        'treatment': 'Improve ventilation, reduce humidity, apply fungicides'
    },
    'Tomato___Septoria_leaf_spot': {
        'plant': 'Tomato',
        'disease': 'Septoria Leaf Spot',
        'description': 'Fungal disease causing many small spots on leaves',
        'symptoms': 'Small circular spots with dark borders, yellow halos',
        'treatment': 'Remove lower leaves, apply fungicides, crop rotation'
    },
    'Tomato___Spider_mites': {
        'plant': 'Tomato',
        'disease': 'Spider Mites',
        'description': 'Pest infestation causing stippling and webbing',
        'symptoms': 'Tiny yellow spots, fine webbing, leaf yellowing',
        'treatment': 'Spray with water, use insecticidal soap, introduce predators'
    },
    'Tomato___Target_Spot': {
        'plant': 'Tomato',
        'disease': 'Target Spot',
        'description': 'Fungal disease causing target-like lesions',
        'symptoms': 'Dark spots with concentric rings, leaf yellowing',
        'treatment': 'Apply fungicides, improve air circulation, crop rotation'
    },
    'Tomato___Yellow_Leaf_Curl_Virus': {
        'plant': 'Tomato',
        'disease': 'Yellow Leaf Curl Virus',
        'description': 'Viral disease transmitted by whiteflies',
        'symptoms': 'Upward leaf curling, yellowing, stunted growth',
        'treatment': 'Control whiteflies, use resistant varieties, remove infected plants'
    },
    'Tomato___healthy': {
        'plant': 'Tomato',
        'disease': 'Healthy',
        'description': 'No disease detected - plant appears healthy',
        'symptoms': 'Normal green leaves, healthy fruit development',
        'treatment': 'Continue regular care and maintenance'
    },
    'Unknown___Class_24': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_25': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_26': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_27': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_28': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_29': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_30': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_31': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_32': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    },
    'Unknown___Class_33': {
        'plant': 'Unknown',
        'disease': 'Unidentified Disease',
        'description': 'Disease class not recognized in our database',
        'symptoms': 'Please consult agricultural expert for identification',
        'treatment': 'Consult local agricultural extension service'
    }
}

# History storage
prediction_history = []

def preprocess_image(image: Image.Image) -> np.ndarray:
    '''Preprocess image for model prediction'''
    image = image.resize((224, 224))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get('/', response_class=HTMLResponse)
async def home():
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    agent_debug_log(
        hypothesis_id="H1",
        location="main.py:home:before_open",
        message="About to open home template",
        data={"template_path": template_path, "exists": os.path.exists(template_path)},
    )
    with open(template_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get('/crop-detection', response_class=HTMLResponse)
async def crop_detection_page():
    # Hypothesis H1: path resolution to crop_detection.html is wrong when opening the template.
    template_path = os.path.join(BASE_DIR, "templates", "crop_detection.html")
    print(f"[DEBUG] crop_detection_page called. BASE_DIR={BASE_DIR}, template_path={template_path}, exists={os.path.exists(template_path)}")
    agent_debug_log(
        hypothesis_id="H1",
        location="main.py:crop_detection_page:before_open",
        message="About to open crop_detection template",
        data={
            "cwd": os.getcwd(),
            "template_path": template_path,
            "exists": os.path.exists(template_path),
        },
    )
    try:
        if not os.path.exists(template_path):
            error_msg = f"Template file not found at: {template_path}. BASE_DIR={BASE_DIR}, cwd={os.getcwd()}"
            print(f"[ERROR] {error_msg}")
            raise FileNotFoundError(error_msg)
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"[DEBUG] Successfully read crop_detection.html, length={len(content)}")
        agent_debug_log(
            hypothesis_id="H1",
            location="main.py:crop_detection_page:after_open",
            message="Successfully read crop_detection.html",
            data={"length": len(content)},
        )
        return HTMLResponse(content=content)
    except FileNotFoundError as e:
        print(f"[ERROR] FileNotFoundError: {e}")
        agent_debug_log(
            hypothesis_id="H1",
            location="main.py:crop_detection_page:file_not_found",
            message="FileNotFoundError opening crop_detection.html",
            data={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail=f"Template file not found: {str(e)}")

@app.get('/fertilizer', response_class=HTMLResponse)
async def fertilizer_page():
    template_path = os.path.join(BASE_DIR, "templates", "fertilizer.html")
    agent_debug_log(
        hypothesis_id="H2",
        location="main.py:fertilizer_page:before_open",
        message="About to open fertilizer template",
        data={"template_path": template_path, "exists": os.path.exists(template_path)},
    )
    with open(template_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get('/crop-recommendation', response_class=HTMLResponse)
async def crop_recommendation_page():
    template_path = os.path.join(BASE_DIR, "templates", "crop_recommendation.html")
    with open(template_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get('/contact', response_class=HTMLResponse)
async def contact_page():
    template_path = os.path.join(BASE_DIR, "templates", "contact.html")
    agent_debug_log(
        hypothesis_id="H3",
        location="main.py:contact_page:before_open",
        message="About to open contact template",
        data={"template_path": template_path, "exists": os.path.exists(template_path)},
    )
    with open(template_path, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())


def send_contact_email(name: str, email: str, subject: str, message: str) -> bool:
    """Send contact form submission via Gmail SMTP."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_EMAIL
        msg['To'] = SMTP_EMAIL
        msg['Subject'] = f"[Crop Detection Contact] {subject}"
        
        body = f"""
New contact form submission from Crop Detection System
────────────────────────────────────────────────────

Name:    {name}
Email:   {email}
Subject: {subject}

Message:
{message}

────────────────────────────────────────────────────
Sent from Crop Detection System Contact Form
"""
        msg.attach(MIMEText(body.strip(), 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD.replace(" ", ""))
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"SMTP error: {e}")
        return False


@app.post('/api/contact')
async def contact_submit(
    name: str = Form(...),
    email: str = Form(...),
    subject: str = Form(...),
    message: str = Form(...),
):
    """Handle contact form submission and send email via SMTP."""
    if not name.strip() or not email.strip() or not subject.strip() or not message.strip():
        return JSONResponse(
            {'success': False, 'message': 'All fields are required.'},
            status_code=400
        )
    
    if send_contact_email(name, email, subject, message):
        return JSONResponse({
            'success': True,
            'message': 'Thank you! Your message has been sent successfully. We will get back to you soon.'
        })
    else:
        return JSONResponse(
            {
                'success': False,
                'message': 'Sorry, we could not send your message. Please try again later or email us directly.'
            },
            status_code=500
        )


@app.post('/api/predict')
async def predict_disease(file: UploadFile = File(...)):
    '''Predict plant disease from uploaded image'''
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Save the uploaded image
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        image_path = os.path.join(BASE_DIR, "images", filename)
        image.save(image_path)
        
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        if predicted_class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[predicted_class_idx]
            class_details = CLASS_DETAILS.get(predicted_class, {})
        else:
            predicted_class = f'Class_{predicted_class_idx}'
            class_details = {}
        
        # Create prediction record
        prediction_record = {
            'id': timestamp,
            'filename': filename,
            'image_path': f'/images/{filename}',
            'prediction': predicted_class,
            'confidence': round(confidence * 100, 2),
            'timestamp': timestamp,
            'details': class_details,
            'all_predictions': {
                CLASS_NAMES[i]: float(predictions[0][i]) * 100 
                for i in range(min(len(CLASS_NAMES), len(predictions[0])))
            }
        }
        
        # Add to history (keep last 50 records)
        prediction_history.insert(0, prediction_record)
        if len(prediction_history) > 50:
            prediction_history.pop()
        
        return JSONResponse(prediction_record)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error processing image: {str(e)}')

@app.get('/api/history')
async def get_prediction_history():
    '''Get prediction history'''
    return JSONResponse({'history': prediction_history})

# Feature order for fertilizer model (must match training)
FERTILIZER_FEATURE_ORDER = [
    "Temparature", "Humidity", "Moisture", "Soil_Type", "Crop_Type",
    "Nitrogen", "Potassium", "Phosphorous"
]


@app.post('/api/fertilizer-predict')
async def predict_fertilizer(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    moisture: float = Form(...),
    soil_type: str = Form(...),
    crop_type: str = Form(...),
):
    '''Predict fertilizer using local ML model and label_encoders.pkl'''
    if fertilizer_model is None or label_encoders is None:
        return JSONResponse({
            'success': False,
            'error': 'Fertilizer model not loaded',
            'message': 'fertilizer_model.pkl and label_encoders.pkl must be in the project folder.',
        }, status_code=503)

    try:
        # Encode categorical inputs (same as in Model_train.ipynb)
        soil_encoded = label_encoders["Soil_Type"].transform([soil_type.strip()])[0]
        crop_encoded = label_encoders["Crop_Type"].transform([crop_type.strip()])[0]

        # Build feature row in training order
        features = np.array([[
            temperature,
            humidity,
            moisture,
            soil_encoded,
            crop_encoded,
            nitrogen,
            potassium,
            phosphorus,
        ]])

        pred_encoded = fertilizer_model.predict(features)[0]
        fertilizer_name = label_encoders["Fertilizer_Name"].inverse_transform([pred_encoded])[0]

        return JSONResponse({
            'success': True,
            'recommendation': f"Recommended fertilizer: **{fertilizer_name}**. Use as per label and local agricultural guidelines.",
            'prediction': fertilizer_name,
            'input_data': {
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'moisture': moisture,
                'soil_type': soil_type,
                'crop_type': crop_type,
            },
            'timestamp': int(time.time())
        })

    except ValueError as e:
        return JSONResponse({
            'success': False,
            'error': 'Invalid input',
            'message': f'Invalid soil type or crop type. Use values from the dropdowns. ({e})',
            'input_data': {
                'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium,
                'temperature': temperature, 'humidity': humidity, 'moisture': moisture,
                'soil_type': soil_type, 'crop_type': crop_type,
            }
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': 'Fertilizer prediction failed',
            'message': str(e),
            'input_data': {
                'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium,
                'temperature': temperature, 'humidity': humidity, 'moisture': moisture,
                'soil_type': soil_type, 'crop_type': crop_type,
            }
        }, status_code=500)

@app.post('/api/crop-predict')
async def predict_crop(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...),
):
    '''Predict crop recommendation using local ML model'''
    if crop_model is None:
        return JSONResponse({
            'success': False,
            'error': 'Crop model not loaded',
            'message': 'crop_model.pkl must be in the project folder.',
        }, status_code=503)

    try:
        # Build feature row in training order: N, P, K, temperature, humidity, ph, rainfall
        features = np.array([[
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall,
        ]])

        pred_encoded = crop_model.predict(features)[0]
        
        # Convert to int if it's a numpy type
        if hasattr(pred_encoded, 'item'):
            pred_encoded = pred_encoded.item()
        pred_encoded = int(pred_encoded)
        
        # Map encoded prediction to crop name
        if 0 <= pred_encoded < len(CROP_LABELS):
            crop_name = CROP_LABELS[pred_encoded]
        else:
            crop_name = f"Unknown crop ({pred_encoded})"

        return JSONResponse({
            'success': True,
            'recommendation': f"Recommended crop: **{crop_name}**. This crop is suitable for the given conditions.",
            'prediction': crop_name,
            'input_data': {
                'nitrogen': nitrogen,
                'phosphorus': phosphorus,
                'potassium': potassium,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall,
            },
            'timestamp': int(time.time())
        })

    except Exception as e:
        return JSONResponse({
            'success': False,
            'error': 'Crop prediction failed',
            'message': str(e),
            'input_data': {
                'nitrogen': nitrogen, 'phosphorus': phosphorus, 'potassium': potassium,
                'temperature': temperature, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall,
            }
        }, status_code=500)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
