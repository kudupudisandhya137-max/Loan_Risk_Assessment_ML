import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_loan_data(file_path):
    print("1. Loading raw dataset from Excel...")
    df = pd.read_excel(file_path, sheet_name="Loan_Risk_Dataset")
    
    print(f"   Raw dataset loaded with shape: {df.shape}")

    # ----------------------------------------------------
    # STEP 2: BINARY RISK TARGET CONVERSION
    # ----------------------------------------------------
    # Simplifying risk levels into binary categories (Low Risk vs High Risk)
    df['Binary_Risk'] = df['Risk_Level'].apply(
        lambda x: 'Low Risk' if x == 'Low Risk' else 'High Risk'
    )

    # ----------------------------------------------------
    # STEP 3: DROP IRRELEVANT & LEAKAGE COLUMNS
    # ----------------------------------------------------
    # - Loan_ID: Unique identifier (has no predictive value)
    # - Risk_Score: Causes data leakage (since Risk_Level was derived from it)
    # - Risk_Level: Replaced by our new Binary_Risk target
    columns_to_drop = ["Loan_ID", "Risk_Score", "Risk_Level"]
    processed_df = df.drop(columns=columns_to_drop, errors="ignore")

    # ----------------------------------------------------
    # STEP 4: CATEGORICAL ENCODING
    # ----------------------------------------------------
    # Convert text-based Employment_Type ('Salaried'/'Self-Employed') into numeric format
    label_encoders = {}
    if "Employment_Type" in processed_df.columns:
        le_emp = LabelEncoder()
        processed_df["Employment_Type"] = le_emp.fit_transform(processed_df["Employment_Type"].astype(str))
        label_encoders["Employment_Type"] = le_emp
        print("   Categorical 'Employment_Type' encoded successfully.")

    # ----------------------------------------------------
    # STEP 5: ENCODE TARGET VARIABLE
    # ----------------------------------------------------
    target_encoder = LabelEncoder()
    processed_df["Binary_Risk_Encoded"] = target_encoder.fit_transform(processed_df["Binary_Risk"])
    processed_df = processed_df.drop(columns=["Binary_Risk"]) # Keep numerical target

    print(f"   Preprocessed dataset shape: {processed_df.shape}")
    return processed_df, target_encoder, label_encoders

if __name__ == "__main__":
    excel_file = "Intelligent_Loan_Risk_Assessment_Dataset.xlsx"
    
    # Run preprocessing pipeline
    clean_df, t_enc, l_enc = preprocess_loan_data(excel_file)
    
    # Export preprocessed dataset to CSV for inspection/submission
    output_filename = "preprocessed_loan_dataset.csv"
    clean_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ Preprocessed dataset successfully exported to '{output_filename}'!")
    print("\nFirst 3 rows of the preprocessed dataset:")
    print(clean_df.head(3))