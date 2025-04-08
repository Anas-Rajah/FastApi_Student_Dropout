from fastapi import FastAPI, Query
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
import pandas as pd

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = load_model("student_performance_api_End")

@app.get("/predict")
async def predict(
    Age: int = Query(..., description="Student's age (0: Less than 20 years, 1: 20-24 years, 2: 25-29 years, 3: 30 years or older)"),
    Last_GPA: int = Query(..., description="Last GPA (0: 0-49, 1: 50-59, 2: 60-69, 3: 70-84, 4: 85-100)"),
    Turkish_Proficiency: int = Query(..., description="Turkish language proficiency (0: Weak, 1: Average, 2: High)"),
    Major_Satisfaction: int = Query(..., description="Satisfaction with the major (0: Not satisfied, 1: Neutral, 2: Satisfied)"),
    Family_Economy: int = Query(..., description="Family's economic status (0: Low, 1: Medium, 2: High)"),
    Scholarship: int = Query(..., description="Receiving a scholarship (0: No, 1: Yes)"),
    Part_Time_Work: int = Query(..., description="Part-time work (0: No, 1: Yes, part-time, 2: Yes, full-time)"),
    Father_Education_Level: int = Query(..., description="Father's education level (0: No formal education, 1: Basic education, 2: Secondary, 3: Bachelor's, 4: Master's or PhD)"),
    Mother_Education_Level: int = Query(..., description="Mother's education level (0: No formal education, 1: Basic education, 2: Secondary, 3: Bachelor's, 4: Master's or PhD)"),
    Parents_Employment_Status: int = Query(..., description="Parents' employment status (0: Both unemployed, 1: Mother works, father unemployed, 2: Father works, mother unemployed, 3: Both employed)"),
):
    # Collect data in a dictionary
    data = {
        "Age": Age,
        "Last_GPA": Last_GPA,
        "Turkish_Proficiency": Turkish_Proficiency,
        "Major_Satisfaction": Major_Satisfaction,
        "Family_Economy": Family_Economy,
        "Scholarship": Scholarship,
        "Part_Time_Work": Part_Time_Work,
        "Father_Education_Level": Father_Education_Level,
        "Mother_Education_Level": Mother_Education_Level,
        "Parents_Employment_Status": Parents_Employment_Status,
    }

    # Convert data to DataFrame
    df = pd.DataFrame([data])

    # Make prediction
    predictions = predict_model(model, data=df)

    # Extract prediction
    predicted_target = predictions["prediction_label"].iloc[0]
    target_map = {
        0: "Likely to drop out",
        1: "Likely to graduate!",
    }
    target = target_map.get(predicted_target, "Unknown")

    return {"Predicted Result": target}