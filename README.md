# Telco Customer Churn Prediction App

This project consists of a Machine Learning model trained on the Telco Customer Churn dataset, served via a FastAPI backend, and visualized through a Streamlit frontend web app. It is configured for deployment on Render.

## Project Structure

- `api/index.py`: FastAPI backend entry point.
- `app.py`: Streamlit frontend app that collects customer details and communicates with the FastAPI backend.
- `models/`: Directory containing pickle files for the model (`churn_model.pkl`), preprocessing pipeline (`preprocessor.pkl`), and features lists.
- `render.yaml`: Render Blueprint configuration for both the FastAPI backend and Streamlit frontend.
- `requirements.txt`: Python package requirements.

---

## Local Development

### 1. Install Dependencies

It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI Backend Locally

You can run the FastAPI backend locally using uvicorn:

```bash
python3 -m uvicorn api.index:app --reload
```

The backend API will start at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to see the interactive Swagger API documentation.

### 3. Run the Streamlit Frontend Locally

In a new terminal window, set the environment variable pointing to your backend API, and run the Streamlit app:

```bash
export BACKEND_URL="http://localhost:8000"
streamlit run app.py
```

---

## Deploying to Render

This repo includes a `render.yaml` Blueprint that creates two Render web services:

- `telco-churn-api`: FastAPI backend.
- `telco-churn-streamlit`: Streamlit frontend.

### Steps

1. Push this project to a GitHub repository.
2. In Render, choose **New > Blueprint**.
3. Connect the GitHub repository.
4. Render will read `render.yaml` and create both services.
5. Deploy the Blueprint.

The Streamlit service receives `BACKEND_URL` from the API service automatically.

### Manual Render Setup

If you do not use the Blueprint, create two Render web services manually.

Backend:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api.index:app --host 0.0.0.0 --port $PORT`

Frontend:

- Build command: `pip install -r requirements.txt`
- Start command: `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- Environment variable: `BACKEND_URL=<your Render backend URL>`
