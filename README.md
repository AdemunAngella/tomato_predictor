# Tomato Quality Classifier (Streamlit App)

A simple and intuitive Streamlit-based web application that predicts whether
a tomato is **Fresh** or **Rotten** using a trained TensorFlow/Keras
deep learning model.
The app provides real-time predictions, image preview, history tracking,
and a confidence chart, all running smoothly in the browser.

## Features

-   Upload Images (PNG/JPG/JPEG)
-   Instant Predictions powered by a TensorFlow/Keras model
-   Dynamic Confidence Chart (Chart.js)
-   Recent Predictions History (LocalStorage)

## Project Structure

    tomato-app/
    │
    ├── app_streamlit.py
    ├── requirements.txt
    ├── models/
    │   └── tomato_classifier.h5
    │
    ├── static/
           ├── logo.png
           ├── favicon.png
           ├── fresh.png
           ├── rotten.png
           └── uploads/

## Model Details

    The model is a binary classifier trained to distinguish:

    - Fresh Tomato
    - Rotten Tomato

    Model Output: 0 → Rotten Tomato
                    1 → Fresh Tomato

    A sigmoid output layer is used, with a prediction threshold of 0.5.

## Running Locally

    pip install -r requirements.txt
    python app.py

## Installation & Setup

    1. Clone the project
    git clone https://github.com/yourusername/tomato-predictor.git
    cd tomato-app

    2. Install dependencies
    pip install -r requirements.txt

    3. Run the Streamlit app
    python app_streamlit.py

    The app will run on: http://127.0.0.1:8501/

## Deployment (Digital Ocean)

1.  Upload the entire tomato-predictor/ folder.
2.  Create a virtual environment\ 
        - mkvirtualenv tomato-env --python=3.11
        - pip install -r requirements.txt
3.  Install requirements\
4.  Configure WSGI\
        - from app import app as application
5.  Set static files in the Streamlit Cloud Web panel:
        -  URL	Directory
            /static/home/username/tomato-predictor/static/
6.  Test Your App

        - Reload the web app on Streamlit Cloud.
        - Visit your site URL (yourusername.streamlitcloud.com) and test:
            _ Upload images
            _ Prediction result and icons
            _ History chart

##  Client-Side Prediction History

    No database is used - Nothing is stored on the server.
    All prediction history is saved locally in the browser via localStorage.

    Stored data includes:
        - Timestamp (local date & time)
        - Predicted label (Fresh/Rotten)
        - Confidence (%)
        - Icon path

##  Contributing

    Feel free to open issues or submit pull requests. Feedback and improvements are welcome!

## Requirements

    streamlit==1.26.0
    tensorflow==2.13.0
    numpy==1.26.2
    pillow==10.0.0
    opencv-python-headless==4.8.1.78
    matplotlib==3.8.0
    seaborn==0.12.2
    scikit-learn==1.3.2

## License

    MIT License

##  Authors

    Built for tomato quality prediction.
    1. Proscovia Nabuguzi
    2. Ademun Angella 
    3. Hilda Ruth Akao