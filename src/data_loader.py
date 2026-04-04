# src/data_loader.py
# PURPOSE: Load the clinic JSON file and return it as a Python dictionary.
# This is Stage 1 of our RAG pipeline.

import json
import os

def load_clinic_data(filepath: str) -> dict:
    """
    Loads the clinic data from a JSON file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        A Python dictionary containing all clinic data.
    """
    # Check if the file actually exists before trying to open it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # Open the file and parse the JSON into a Python dictionary
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Successfully loaded data for {len(data['clinics'])} clinics.")
    return data


# --- TEST BLOCK ---
# This code only runs when you execute THIS file directly (not when imported)
if __name__ == "__main__":
    # Build the path to our data file
    # os.path.dirname(__file__) = the folder this script is in (src/)
    # We go one level up (..) to reach the project root, then into data/
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "..", "data", "clinic_data.json")

    clinic_data = load_clinic_data(data_path)

    # Print the name of the first clinic to confirm it worked
    first_clinic = clinic_data["clinics"][0]
    print(f"First clinic loaded: {first_clinic['name']}")
    print(f"Number of services: {len(first_clinic['services'])}")
    print(f"Number of doctors: {len(first_clinic['doctors'])}")