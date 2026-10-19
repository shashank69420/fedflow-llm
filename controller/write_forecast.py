FORECAST_FILE = "forecast_value.txt"

def get_forecast():
    predicted_value = 118.4  # REPLACE with a real number from your Part 1 run
    return predicted_value

if __name__ == "__main__":
    forecast = get_forecast()
    with open(FORECAST_FILE, "w") as f:
        f.write(str(forecast))
    print(f"[FedFlow] Wrote forecast value {forecast} to {FORECAST_FILE}")
