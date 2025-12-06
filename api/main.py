import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from datetime import datetime
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Build models directory path relative to this file (api/main.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of "api" -> project root
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_pickle(fname):
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# Load models and encoders (will raise clear error if missing)
one_hot_encoder = load_pickle("one_hot_encoder.pkl")
label_encoders = load_pickle("label_encoders.pkl")
scaler = load_pickle("scaler.pkl")
model = load_pickle("model.pkl")

# Define input schema
class CarInput(BaseModel):
    Levy: int
    Manufacturer: str
    Model: str
    Prod_year: int
    Category: str
    Leather_interior: str
    Fuel_type: str
    Engine_volume: float
    Mileage: int
    Cylinders: float
    Gear_box_type: str
    Drive_wheels: str
    Wheel: str
    Color: str
    Airbags: int

@app.post("/predict/")
def predict(car_data: CarInput):
    try:
        print("\n================ NEW REQUEST ================")

        # Convert input to DataFrame
        data = pd.DataFrame([car_data.dict()])
        print("Incoming data:\n", data)

        # Generate `Age` feature and drop Prod_year
        data['Age'] = datetime.now().year - data['Prod_year']
        data.drop(columns=['Prod_year'], errors='ignore', inplace=True)

        # Rename columns to match preprocessing pipeline
        column_rename_map = {
            "Leather_interior": "Leather interior",
            "Gear_box_type": "Gear box type",
            "Drive_wheels": "Drive wheels",
            "Engine_volume": "Engine volume",
            "Fuel_type": "Fuel type"
        }
        data.rename(columns=column_rename_map, inplace=True)
        print("\nAfter rename:\n", data.head())

        # ONE-HOT encode categorical columns (if present)
        one_hot_columns = ['Leather interior', 'Gear box type', 'Drive wheels', 'Wheel']
        present_one_hot_cols = [c for c in one_hot_columns if c in data.columns]
        print("\nOne-hot columns:", present_one_hot_cols)

        if present_one_hot_cols:
            encoded = one_hot_encoder.transform(data[present_one_hot_cols])
            if hasattr(encoded, "toarray"):
                encoded = encoded.toarray()

            encoded_df = pd.DataFrame(
                encoded,
                columns=one_hot_encoder.get_feature_names_out(present_one_hot_cols),
                index=data.index
            )
            print("\nOne-hot encoded df:\n", encoded_df.head())

            data = pd.concat([data, encoded_df], axis=1)
            data.drop(columns=present_one_hot_cols, inplace=True)

        # LABEL encode categorical columns
        label_encode_columns = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Color']
        print("\nChecking label encoders…")
        for column in label_encode_columns:
            if column in data.columns:
                print(f"Column '{column}' incoming value: {data[column].iloc[0]}")
                le = label_encoders.get(column)
                if le is None:
                    print(f"❌ No label encoder found for '{column}'")
                    raise ValueError(f"No label encoder found for column '{column}'")

                # Debug: show available classes
                print(f"Label encoder classes for {column}: {list(le.classes_)}")

                value = data[column].iloc[0]
                if value not in le.classes_:
                    raise ValueError(f"❌ Unknown value '{value}' for column '{column}'")

                data[column] = le.transform([value])

        # SCALE numerical columns
        numerical_columns = ['Levy', 'Engine volume', 'Mileage', 'Age']
        print("\nBefore scaling numerical columns:\n", data[numerical_columns])

        data[numerical_columns] = scaler.transform(data[numerical_columns])

        print("\nAfter scaling:\n", data[numerical_columns])

        print("\nFinal columns before prediction:", data.columns)
        print("Final data row:\n", data.head())

        # Make prediction
        prediction = model.predict(data)
        pred_value = float(prediction[0])
        print("\nPrediction:", pred_value)

        return {"prediction": pred_value}

    except Exception as e:
        print("\n❌ ERROR OCCURRED:")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
