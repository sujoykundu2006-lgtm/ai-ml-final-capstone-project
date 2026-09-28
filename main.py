"""
Module 6: Interactive Terminal Interface for Real Estate Prediction
"""

from model_pipeline import predict_price, load_and_preprocess_data, train_and_benchmark
import os

def run_cli():
    print("************************************************************")
    print("   PROPHETESTATE - AI REAL ESTATE VALUATION SYSTEM (CAPSTONE) ")
    print("************************************************************")

    if not os.path.exists("best_model.pkl"):
        print("Training model and creating initial pipeline artifacts...")
        X, y = load_and_preprocess_data()
        train_and_benchmark(X, y)

    while True:
        print("\nEnter House Attributes for Valuation:")
        try:
            area = float(input(" • Total Area (sq ft, e.g., 2400): "))
            bedrooms = int(input(" • Number of Bedrooms (1-5): "))
            bathrooms = int(input(" • Number of Bathrooms (1-4): "))
            floors = int(input(" • Number of Floors (1-3): "))
            year = int(input(" • Year Built (e.g., 2005): "))
            
            print(" • Locations: [1] Downtown, [2] Suburban, [3] Urban, [4] Rural")
            loc_map = {"1": "Downtown", "2": "Suburban", "3": "Urban", "4": "Rural"}
            loc_choice = input("   Select location (1-4): ").strip()
            location = loc_map.get(loc_choice, "Suburban")

            print(" • Condition: [1] Excellent, [2] Good, [3] Fair, [4] Poor")
            cond_map = {"1": "Excellent", "2": "Good", "3": "Fair", "4": "Poor"}
            cond_choice = input("   Select condition (1-4): ").strip()
            condition = cond_map.get(cond_choice, "Good")

            garage_choice = input(" • Garage included? (y/n): ").strip().lower()
            garage = "Yes" if garage_choice == "y" else "No"

            property_data = {
                "Area": area,
                "Bedrooms": bedrooms,
                "Bathrooms": bathrooms,
                "Floors": floors,
                "YearBuilt": year,
                "Location": location,
                "Condition": condition,
                "Garage": garage
            }

            price = predict_price(property_data)
            print("-" * 50)
            print(f" Predicted Valuation: ${price:,.2f}")
            print(f" Price Per Sq Ft:     ${price / area:,.2f}")
            print("-" * 50)

        except ValueError:
            print("Invalid input value. Please enter proper numbers.")

        cont = input("\nEvaluate another house? (y/n): ").strip().lower()
        if cont != "y":
            print("\nExiting valuation engine. Goodbye!")
            break

if __name__ == "__main__":
    run_cli()