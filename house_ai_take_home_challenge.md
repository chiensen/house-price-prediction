# Take-Home AI Challenge: House Price Estimation System

**Role:** AI Engineer (Fresh Graduate)  
**Duration:** Approx. 2 Days  
**Keyword:** House

## 🧠 Overview

Develop an end-to-end machine learning pipeline that estimates the **price of a house** based on its features. This challenge is designed to assess your ability to work with structured data, train a regression model, and deploy a minimal AI-powered application.

## 📌 Objectives

1. **Data Collection**
   - Use a public housing dataset (e.g., from Kaggle, UCI, or sklearn’s Boston housing).
   - Include features such as: `location`, `square_feet`, `bedrooms`, `bathrooms`, `year_built`, `lot_size`, and `price`.
   - Perform necessary preprocessing (e.g., encoding, scaling, missing value handling).

2. **Model Training**
   - Train a regression model using scikit-learn, XGBoost, LightGBM, or another suitable framework.
   - Evaluate performance using RMSE, MAE, and R² metrics.
   - Save the model for inference using `joblib` or similar tools.

3. **Deployment**
   - Build an interactive app or REST API using tools like Streamlit, FastAPI, Flask, or Gradio.
   - Allow users to enter house features and receive an estimated price prediction.

4. **Bonus (Optional)**
   - Include visualizations of feature importances or prediction error.
   - Deploy the system online (e.g., Hugging Face Spaces, Render, Railway).

## 🧪 Deliverables

- Codebase (GitHub repo or zip).
- A `README.md` containing:
  - Dataset source and cleaning steps.
  - Model training explanation and evaluation results.
  - Setup instructions and demo screenshot or link.
- Short reflection: how would you make this production-ready for a real estate company?

## ✅ Evaluation Criteria

- Quality of end-to-end pipeline: data → model → interface
- Code clarity and modularity
- Explanation of approach and model performance
- UI usability and deployment (if applicable)

---

**Tip:** Don’t focus on perfection—aim to show a clean, understandable pipeline that you could easily extend or explain in a real interview.
