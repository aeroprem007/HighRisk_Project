# data/generate_data.py
import os
import pandas as pd
import random
from faker import Faker

os.makedirs('data', exist_ok=True)
fake = Faker()

records = []
for _ in range(500):
    age         = random.randint(18, 90)
    sex         = random.choice(['M', 'F'])
    hr          = random.randint(60, 100)
    systolic    = random.randint(110, 140)
    diastolic   = random.randint(70, 90)
    bp          = f"{systolic}/{diastolic}"
    temp        = round(random.uniform(36.0, 38.5), 1)
    diabetes    = random.choice(['Yes', 'No'])
    smoker      = random.choice(['Yes', 'No'])
    complaint   = fake.sentence(nb_words=6)
    note = (
        f"Patient age: {age}, Gender: {sex}, "
        f"Vitals: HR={hr}, BP={bp}, Temp={temp}°C. "
        f"History: Diabetic={diabetes}, Smoking={smoker}. "
        f"Current complaint: {complaint}"
    )
    records.append({'notes': note})

df = pd.DataFrame(records)
df.to_csv('data/clinical_notes.csv', index=False)
print("clinical_notes.csv generated successfully.")