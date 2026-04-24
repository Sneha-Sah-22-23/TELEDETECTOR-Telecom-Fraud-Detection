# TELEDETECTOR: Telecom Fraud Detection

This is a hyrid prediction model that uses ML + Logic Check to detect the Fraudulent Numbers, Using MLflow Devops.

## ML Model (Layer 1):
This model uses a Random Forest Classifier for fraud detection.

Key features:  
 • Learns complex behavior patterns   
 • Works well with large datasets  
 • Handles imbalance using class_weight = balanced 
The model predicts whether a number is fraudulent based on behavior. 

## Sim Farm Detection (Layer 2):   
Fraudsters often use multiple SIM cards under one identity.   
This is known as a SIM Farm.  
Rule applied:   
• If a user owns 10 or more SIM cards, they are flagged as suspicious
This rule works even if the behavior looks normal.

## MLflow DevOps
Used MLflow to manage the machine learning workflow.  
MLflow helps in:  
 • Tracking experiments  
 • Recording accuracy and parameters  
 • Saving trained models  
 • Maintaining version control   
This ensures reproducibility and proper model management.
