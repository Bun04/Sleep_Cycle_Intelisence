import numpy as np

def predict_sleep_cycles(inputs):
    """
    Predict number of sleep cycles based on inputs using a simple formula.
    In a real model, this would be trained on data.
    """
    base_cycles = inputs['sleep_hours'] / 1.5
    adjustment = (inputs['deep_sleep_minutes'] - 75) * 0.01 + (inputs['rem_sleep_minutes'] - 90) * 0.005 - inputs['awakenings'] * 0.1
    predicted_cycles = base_cycles + adjustment
    return round(max(1, min(6, predicted_cycles)), 1)