# **CNS-ADR-Prediction-TB-Patients**

**Live App:** https://cns-adr-prediction-tb-patients-ydad2zfkvuqvkyta8fwpgp.streamlit.app/

> Predicts neurological adverse drug reactions in TB patients using clinical data, enabling proactive risk stratification during treatment.

## OVERVIEW


This project predicts the risk of neurological adverse drug reactions (CNS ADRs) in tuberculosis (TB) patients using clinical and demographic variables. Prior ML models built using this dataset focused on adverse drug reactions related to the liver (hepatotoxicity), while neurological ADR prediction models traditionally rely on drug-level features. However, this project uses patient-level clinical variables to model risk, making it more applicable for real-world clinical use.


## OBJECTIVES


- Predict CNS-related ADR risk during TB treatment
- Enable early risk stratification of patients
- Support clinical decision-making via machine learning
  

## MODEL PERFORMANCE


### Key Metrics

- **ROC-AUC:** 0.96
    
- **F1 Score:** 0.89
  
- **Recall:** 0.83
   
- **Average Precision:** 0.92
    
- **Brier Score:** 0.13
   
- **Nested CV AUC:** 0.91 ± 0.03  

Despite the smaller dataset size (311 rows and 37 feature columns), the model was able to show strong predictive performance with a high degree of robustness and stability.

## MODEL EVALUATION


The model was evaluated using **rigorous validation techniques to ensure generalizability and prevent overfitting**, especially since the dataset size is small. Some of the model evaluation techniques include,

- Nested Cross Validation
- Stratified K-Fold Cross Validation
- Repeated Stratified K-Fold Cross Validation
- ROC Curve
- Precision-Recall Curve
- SHAP Analysis (model interpretation)
  

## SHAP ANALYSIS INFERENCE


Using SHAP analysis, we were able to identify the critical and influential features that were involved in the prediction of CNS ADRs. Those critical features were,

- Gastrointestinal ADRs
- Genitourinary (Urinary tract + Reproductive Organs) ADRs
- Alcohol consumption
- Treatment duration

These results reinforce that patient-level clinical factors play a critical role in predicting neurological adverse drug reactions, rather than relying solely on drug properties. 


## FEATURES USED


- Gastrointestinal ADRs
- Genitourinary reactions
- Alcohol consumption
- Treatment duration
- HIV status
- Weight Change
- Age
  

## LIVE APPLICATION


The deployed streamlit application allows the user to,

- Input clinical data of patients.
- Get real-time risk predictions along with risk stratification based on probability.
  

## KEY CHALLENGES AND APPROACH


- **Small Dataset (311 samples):** Addressed using stratified, repeated, and nested cross-validation  
- **Feature Selection:** Combined VIF analysis with model-based feature importance  
- **Class Imbalance:** Handled using stratified sampling and performance metrics (F1-score, recall) 


## TECH STACK USED


- Python (Numpy, Pandas)
- Scikit-learn
- AdaBoost (Tuned Hyperparameters)
- SHAP (Feature importance and model interpretation)
- Streamlit


## MODEL SELECTION

- Multiple models like Logistic Regression, Random Forest, XGBoost, and Gradient Boosting were evaluated.

- However, the AdaBoost model with tuned hyperparameters was chosen as the final model as it provided the best balance between recall and precision while maintaining strong generalization performance.  

- Moreover, it demonstrated minimal overfitting and achieved the highest overall performance across evaluation metrics.


## PROJECT STRUCTURE

- `app.py`: Streamlit application  
- `final_model_ab.pkl`: Trained AdaBoost model
- `columns_f.pkl`: Model input features  
- `scaler.pkl`: Feature scaler  



## BUSINESS AND CLINICAL OUTCOMES


- Allows the early identification of high-risk patients for neurological adverse drug reactions.
- Enables proactive patient monitoring and targeted intervention during TB treatment.
- Reduces long-term treatment costs by preventing severe adverse events that may need hospitalization or extended care.
- Improves resource allocation by helping clinicians focus their attention on high-risk patients in resource-limited settings.
- Supports data-driven decision-making by enhancing treatment efficiency and limiting unnecessary interventions.


## LIMITATIONS AND FUTURE WORK


- The dataset is relatively small (311 rows, 37 feature columns), which is sometimes common in healthcare due to privacy issues, data access restrictions, and the cost of clinical data collection.

- However, despite the limited sample size, the model showed strong generalization with consistent performance across various rigorous model evaluation techniques, indicating minimal overfitting and robustness.

- Can improve model performance and validity by incorporating larger external datasets for real-world clinical validation.


## APP PREVIEW


<img width="100%" src="https://github.com/user-attachments/assets/0bf6cbf3-75b3-469e-9700-8756b834840c" />

<img width="100%" src="https://github.com/user-attachments/assets/17aa037b-8970-4169-9ad0-6581f5e2f885" />


## DISCLAIMER


This tool is intended for clinical decision support and educational purposes only and should not replace professional medical judgment.



















