import pandas as pd
import numpy as np

print("--- SYSTEM CHECK ---")

# 1. Test Math
print(f"NumPy Version: {np.__version__}")

# 2. Test Dataframes (The Excel Killer)
data = {'Asset': ['BTC', 'ETH'], 'Price': [95000, 2800]}
df = pd.DataFrame(data)

print("\nYOUR DATAFRAME:")
print(df)

print("\n--- STATUS: READY FOR AUDIT ---")