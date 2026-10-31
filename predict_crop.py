import pickle
import sys
 
import pandas as pd
 
with open("best_model.pkl", "rb") as f:
    bundle = pickle.load(f)
 
FEATURES = bundle["features"]
 
 
def predict_crop(values):
    """Take 7 values [N, P, K, temperature, humidity, ph, rainfall] and return the crop name."""
    row = pd.DataFrame([values], columns=FEATURES)
    if bundle["scaler"] is not None:
        row = bundle["scaler"].transform(row)
    prediction = bundle["model"].predict(row)
    return bundle["label_encoder"].inverse_transform(prediction)[0]
 
 
if __name__ == "__main__":
    if len(sys.argv) == 8:
        # Values given on the command line
        values = [float(v) for v in sys.argv[1:]]
    else:
        # Otherwise ask the user to type each value
        print("Enter the soil and weather values:")
        values = [float(input(f"{name}: ")) for name in FEATURES]
 
    crop = predict_crop(values)
    print(f"\nModel used       : {bundle['model_name']}")
    print(f"Recommended crop : {crop.upper()}")
 