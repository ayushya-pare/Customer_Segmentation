# Project - Telecom Customer Segmentation and Churn Prediction
This project uses Data analysis, clustering, and machine learning for customer segmentation and predict which customers are likely to churn.

### Methods Used
* Exploratory Data Analysis
* Data Visualization
* Machine Learning

### Technologies
1. EDA - Pandas, numpy
2. Visualization - Matplotlib, seaborn 
3. Clustering / Segmentation - K-means
4. Machine learning - Modelling (PyCaret), Hyperparameter Tuning, Evaluation (Shapley), MLFlow
5. Deployment - Docker, Flask, Streamlit


## The features:
- customerID: Customer ID
- gender: Client gender (male / female)
- SeniorCitizen: Is the client retired (1, 0)
- Partner: Is the client married (Yes, No)
- tenure: How many months a person has been a client of the company
- PhoneService: Is the telephone service connected (Yes, No)
- MultipleLines: Are multiple phone lines connected (Yes, No, No phone service)
- InternetService: Client’s Internet service provider (DSL, Fiber optic, No)
- OnlineSecurity: Is the online security service connected (Yes, No, No internet service)
- OnlineBackup: Is the online backup service activated (Yes, No, No internet service)
- DeviceProtection: Does the client have equipment insurance (Yes, No, No internet service)
- TechSupport: Is the technical support service connected (Yes, No, No internet service)
- StreamingTV: Is the streaming TV service connected (Yes, No, No internet service)
- StreamingMovies: Is the streaming cinema service activated (Yes, No, No internet service)
- Contract: Type of customer contract (Month-to-month, One year, Two year)
- PaperlessBilling: Whether the client uses paperless billing (Yes, No)
- PaymentMethod: Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- MonthlyCharges: Current monthly payment
- TotalCharges: The total amount that the client paid for the services for the entire time
- Churn: Whether there was a churn (Yes or No)
