# Tomato Quality Classifier (Streamlit App)

    A lightweight, user-friendly Streamlit application that predicts whether a tomato is Fresh or Rotten using a trained TensorFlow/Keras deep-learning model.
    It provides instant predictions, image previews, a visual confidence chart, and a searchable prediction history, all running directly in the browser.


## Features

    - Upload images (JPG, JPEG, PNG)
    - Real-time predictions using a TensorFlow/Keras model
    - Confidence visualization chart
    - Built-in prediction history (Session State)
    - Clean UI with icons and preview images

## Project Structure

tomato-predictor/
│
├── app_streamlit.py
├── requirements.txt
├── models/
│   └── tomato_predictor.keras
│
├── static/
│   ├── logo.png
│   ├── favicon.png
│   ├── fresh.png
│   ├── rotten.png
│   └── uploads/
│── .gitignore
│── README.md

## Model Details

    The classifier predicts two categories:
    - Fresh Tomato
    - Rotten Tomato

    Model Output Values: 
        0 → Rotten Tomato
        1 → Fresh Tomato

    The final layer uses a sigmoid activation function with a prediction threshold of 0.5.

## Running Locally

    Run the following commands:
        pip install -r requirements.txt
        streamlit run app_streamlit.py

    App runs locally at:
        http://127.0.0.1:8501/

## Installation & Setup

    1. Clone the project
    git clone https://github.com/yourusername/tomato-predictor.git
    cd tomato-app

    2. Install dependencies
    pip install -r requirements.txt

    3. Run the Streamlit application
    streamlit run app_streamlit.py

## Deployment (Streamlit Cloud)

    1. Push this project to a new GitHub repository.
    2. Visit share.streamlit.io and create a new deployment.
    3. Select your repository and choose "app_streamlit.py" as the entry file.
    4. Streamlit Cloud will automatically install dependencies from requirements.txt.
    5. Open your app URL and test the features:
        - Upload image
        - View prediction
        - Check prediction history
        - Review confidence plot

##  Prediction History (Client-Side)

    No database is used.
    All prediction history is stored temporarily using Streamlit’s session_state.
    History resets when the session restarts.

    Stored information includes:
        - Timestamp
        - Prediction label
        - Confidence score

##  Contributing

    Suggestions, feature additions, and pull requests are welcome.

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

    This project was developed for tomato quality prediction by:
        1. Proscovia Nabuguzi
        2. Ademun Angella 
        3. Hilda Ruth Akao