import mindsdb_sdk as mdb
import pandas as pd

# Connect to the MindsDB server
mdb.connect()

# Initialize the MindsDB API
api = mdb.API()

# Load the final merged dataset
df_final = pd.read_csv('final_dataset_with_symptoms.csv')

# Ensure the dataset is ready for training
print(df_final.head())

# Upload the dataset to MindsDB
api.upload_dataset(file_name='final_dataset_with_symptoms.csv', name='disease_symptom_dataset')

# Create and train a predictor for diseases
api.train_predictor(
    name='disease_predictor',
    from_data='disease_symptom_dataset',
    predict='medical_condition',  # Predicting the disease
    join_on=['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6']  # Symptom columns
)


# Create and train a predictor for drug recommendation
api.train_predictor(
    name='drug_recommendation_predictor',
    from_data='disease_symptom_dataset',
    predict='drug_name',  # Predicting the OTC drug
    join_on=['medical_condition']  # Join on predicted disease
)


# Query the disease predictor with new symptoms
disease_query_result = api.query_predictor(
    name='disease_predictor',
    input_data={
        'Symptom_1': 'itching',
        'Symptom_2': 'skin_rash',
        'Symptom_3': '',
        'Symptom_4': '',
        'Symptom_5': '',
        'Symptom_6': ''
    }
)

# Extract the top 3 predicted diseases
top_diseases = disease_query_result['predictions'][:3]
print("Top 3 Predicted Diseases:", top_diseases)

# Function to get drug recommendations for each disease
def get_drug_recommendations(diseases):
    drug_recommendations = []
    for disease in diseases:
        drug_query_result = api.query_predictor(
            name='drug_recommendation_predictor',
            input_data={
                'medical_condition': disease
            }
        )
        drug_recommendations.append(drug_query_result['predictions'][0])
    return drug_recommendations

# Get drug recommendations for the top 3 predicted diseases
recommended_drugs = get_drug_recommendations(top_diseases)
print("Recommended OTC Drugs:", recommended_drugs)
