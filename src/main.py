import pandas as pd
from src.multimodal_model import multimodal_prediction
from src.shap_explanation import explain_predictions

def main():
    # Generate or load data
    notes_df = pd.read_csv('data/clinical_notes.csv')
    patient_text = notes_df.loc[0, 'notes']
   ## patient_image = 'data/medical_images/train/PNEUMONIA/person1_bacteria_1.jpeg'
    patient_image = 'data/medical_images/train/NORMAL/NORMAL2-IM-1350-0001.jpeg'

    # Run prediction
    print("Running multimodal prediction...")
    score, embedding = multimodal_prediction(patient_text, patient_image)
    print(f"Predicted risk score: {score:.4f}")

    # Run SHAP explanation
    print("Generating SHAP explanations...")
    explain_predictions(embedding)

if __name__ == '__main__':
    main()
