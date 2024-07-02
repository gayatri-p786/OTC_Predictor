import mindsdb
import pandas as pd
import sqlalchemy

# Load the final merged dataset
df_final = pd.read_csv('final_dataset_with_symptoms.csv')

# Ensure the dataset is ready for training
print(df_final.head())

# Connect to MindsDB MySQL Interface
engine = sqlalchemy.create_engine('mysql+pymysql://mindsdb:mindsdb@127.0.0.1:47335')
connection = engine.connect()

# Save the dataset to a temporary CSV file
df_final.to_csv('final_merged_dataset_temp.csv', index=False)

# Create a table in MindsDB and upload the dataset
connection.execute("""
    CREATE TABLE IF NOT EXISTS disease_symptom_dataset (
        drug_name VARCHAR(255),
        medical_condition VARCHAR(255),
        Symptom_1 VARCHAR(255),
        Symptom_2 VARCHAR(255),
        Symptom_3 VARCHAR(255),
        Symptom_4 VARCHAR(255),
        Symptom_5 VARCHAR(255),
        Symptom_6 VARCHAR(255),
        Symptom_7 VARCHAR(255),
        Symptom_8 VARCHAR(255),
        Symptom_9 VARCHAR(255),
        Symptom_10 VARCHAR(255),
        Symptom_11 VARCHAR(255),
        Symptom_12 VARCHAR(255),
        Symptom_13 VARCHAR(255),
        Symptom_14 VARCHAR(255),
        Symptom_15 VARCHAR(255),
        Symptom_16 VARCHAR(255),
        Symptom_17 VARCHAR(255)
    );
""")

# Upload the CSV file to MindsDB table
connection.execute("""
    LOAD DATA INFILE 'final_merged_dataset_temp.csv'
    INTO TABLE disease_symptom_dataset
    FIELDS TERMINATED BY ',' 
    ENCLOSED BY '"'
    LINES TERMINATED BY '\n'
    IGNORE 1 ROWS;
""")

# Train the predictor for diseases
connection.execute("""
    CREATE PREDICTOR disease_predictor
    FROM disease_symptom_dataset
    (SELECT * FROM disease_symptom_dataset)
    PREDICT medical_condition;
""")

# Train the predictor for drug recommendation
connection.execute("""
    CREATE PREDICTOR drug_recommendation_predictor
    FROM disease_symptom_dataset
    (SELECT * FROM disease_symptom_dataset)
    PREDICT drug_name;
""")

# Query the disease predictor with new symptoms
disease_query_result = connection.execute("""
    SELECT medical_condition, probability
    FROM disease_predictor
    WHERE Symptom_1 = 'itching' AND Symptom_2 = 'skin_rash' AND Symptom_3 = '' AND Symptom_4 = '' AND Symptom_5 = '' AND Symptom_6 = ''
    ORDER BY probability DESC
    LIMIT 3;
""").fetchall()

# Extract the top 3 predicted diseases
top_diseases = [result[0] for result in disease_query_result]
print("Top 3 Predicted Diseases:", top_diseases)

# Function to get drug recommendations for each disease
def get_drug_recommendations(diseases):
    drug_recommendations = []
    for disease in diseases:
        drug_query_result = connection.execute(f"""
            SELECT drug_name, probability
            FROM drug_recommendation_predictor
            WHERE medical_condition = '{disease}'
            ORDER BY probability DESC
            LIMIT 1;
        """).fetchone()
        drug_recommendations.append(drug_query_result[0])
    return drug_recommendations

# Get drug recommendations for the top 3 predicted diseases
recommended_drugs = get_drug_recommendations(top_diseases)
print("Recommended OTC Drugs:", recommended_drugs)

# Close the connection
connection.close()
