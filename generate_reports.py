import json

with open("api_endpoint_validation_report.json", "r") as f:
    api_data = json.load(f)

output_comp = {}
for city, data in api_data.items():
    output_comp[city] = {
        "AQI": data["AQI"],
        "Temperature": data["Weather"]["temperature"],
        "Humidity": data["Weather"]["humidity"]
    }

with open("prediction_output_comparison.json", "w") as f:
    json.dump(output_comp, f, indent=4)

with open("dataset_lookup_report.json", "r") as f:
    lookup_data = json.load(f)

debug_report = {
    "root_cause_analysis": {
        "issue": "The model pipeline is not truly location-dependent for most of India.",
        "evidence": [
            "The KDTree lookup in PredictionService._build_sequence queries aqi_training_dataset.csv.",
            "aqi_training_dataset.csv contains only 12 unique spatial coordinates (stations).",
            "When a user searches for any location (e.g., Kochi), KDTree matches it to the nearest of the 12 stations (e.g., Bengaluru, distance ~370km).",
            "This causes the CNN-LSTM to receive the exact same 7-day historical sequence for entirely different cities (e.g., Bengaluru and Kochi).",
            "Consequently, predictions for large regions collapse to identical outputs, breaking location-dependency."
        ]
    },
    "kdtree_mappings": lookup_data
}

with open("location_prediction_debug_report.json", "w") as f:
    json.dump(debug_report, f, indent=4)

print("Reports generated.")
