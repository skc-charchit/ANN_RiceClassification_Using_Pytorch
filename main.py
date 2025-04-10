from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Load the model and scaler
model_filename = 'best_model.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file)

# Define the FastAPI app
app = FastAPI()

# Define a request model for input data
class Data(BaseModel):
    id: int
    Area: float
    MajorAxisLength: float
    MinorAxisLength: float
    Eccentricity: float
    ConvexArea: float
    EquivDiameter: float
    Extent: float
    Perimeter: float
    Roundness: float
    AspectRation: float

# Define a route for predictions
@app.post("/predict")
async def predict(data: Data):
    # Convert input data to a numpy array
    input_data = np.array([
        [data.id, data.Area, data.MajorAxisLength, data.MinorAxisLength, data.Eccentricity,
         data.ConvexArea, data.EquivDiameter, data.Extent, data.Perimeter, data.Roundness,
         data.AspectRation]
    ])

    # Scale the input data using the loaded scaler
    scaled_input_data = loaded_scaler.transform(input_data[:, 1:])  # Exclude 'id' column

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(scaled_input_data)

    return {"prediction": int(prediction[0])}

# Define a route for model information
@app.get("/info")
async def info():
    return {"model": "Quadratic Discriminant Analysis (QDA)"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
