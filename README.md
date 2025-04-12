# Customer Churn Prediction using H2O AutoML and AI Agent
=====================================
Inspired from ai-data-science-team.
## Introduction
---------------

This project aims to predict customer churn using H2O AutoML and an AI-powered ML Agent. The goal is to automate the machine learning workflow while identifying customers who are likely to churn and provide insights to improve customer retention strategies.

## AI Agent and H2O AutoML Integration
---------------
The project utilizes an AI-powered ML Agent that automates the machine learning workflow. The agent is built using:
- LangChain
- Google's Gemini AI Model
- H2O AutoML Framework

### Agent Workflow
![image](https://github.com/user-attachments/assets/80ae9ece-d23a-42af-bcf1-9e3741fcee59)


The ML Agent follows these automated steps:
1. **Start**: Initialize the agent with model parameters and logging settings
2. **Recommend ML Steps**: AI analyzes data and recommends appropriate machine learning steps
3. **Create H2O Code**: Automatically generates H2O AutoML code
4. **Execute H2O Code**: Runs the generated code for model training
5. **Fix Code**: Handles any errors and adjusts code if needed
6. **Report Outputs**: Generates comprehensive model performance reports
7. **End**: Completes the automated workflow

## Dataset
------------

The dataset used for this project is the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset from Kaggle, which contains 7,043 customer records with 20 attributes, including demographic data and account information. The target variable is "Churn," a binary indicator of whether a customer discontinued their service in the previous month.

## Methodology
--------------

1. **AI Agent Setup**: 
   - Configured ML Agent with Google's Gemini model
   - Set up logging and model directory paths
   - Initialized H2O AutoML integration

2. **Automated ML Pipeline**:
   - Data preprocessing through agent recommendations
   - H2O AutoML model training with optimal parameters
   - Automated model selection and evaluation
   - Hyperparameter tuning via grid search

3. **Model Training**: The agent automatically trained multiple models:
   - Stacked Ensemble
   - GBM (Gradient Boosting Machine)
   - XGBoost
   - GLM (Generalized Linear Model)
   - Random Forest

## Results
------------

The AI Agent's automated analysis produced the following results:

* Best Model: Stacked Ensemble with AUC score of 0.8287
* Model Performance Metrics:
  - LogLoss: 0.4428
  - Accuracy: 0.7966
  - Precision: 0.8571
  - Recall: 0.6444
  - F1 Score: 0.7323

## Agent-Generated Insights
--------------------------

The ML Agent identified key patterns:
- High correlation between tenure and churn
- Significant impact of contract type on customer retention
- Internet service quality as a major churn factor

## Conclusion
--------------

The combination of AI Agent and H2O AutoML provides an efficient automated solution for churn prediction. The Stacked Ensemble model selected by the agent demonstrates strong predictive performance, while the automated workflow ensures reproducibility and efficiency.

## Recommendations
------------------

Based on the Agent's analysis:
1. **Targeted Marketing**: Focus on high-risk segments identified by the model
2. **Contract Optimization**: Develop retention strategies for month-to-month contracts
3. **Service Improvement**: Enhance internet service quality based on model insights
4. **Customer Engagement**: Implement proactive engagement for at-risk customers

## Future Work
--------------

1. **Agent Enhancement**: 
   - Expand AI capabilities for deeper feature engineering
   - Implement automated A/B testing
   - Develop real-time prediction capabilities

2. **Model Improvements**:
   - Collect additional customer interaction data
   - Experiment with advanced ensemble techniques
   - Implement automated model retraining

3. **Production Deployment**:
   - Create API endpoints for model serving
   - Develop monitoring systems for model performance
   - Implement automated model updates

## Dependencies
--------------
- Python 3.x
- H2O AutoML
- LangChain
- Google Gemini AI
- Pandas
- Scikit-learn

## Usage
--------
```python
# Initialize ML Agent
ml_agent = H2OMLAgent(
    model=llm,
    log=True,
    log_path=LOG_PATH,
    model_directory=MODEL_PATH,
)

# Run automated ML pipeline
ml_agent.invoke_agent(
    data_raw=df,
    user_instructions="Please do classification on 'Churn'",
    target_variable="Churn"
)
```
