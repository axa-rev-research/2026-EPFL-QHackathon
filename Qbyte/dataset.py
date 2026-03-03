import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

import numpy as np
import pandas as pd

# ====================================================
# Generate a synthetic catastrophe event dataset
# Each row represents one event with a loss amount.
# Columns: year, event_id (globally unique), insured_loss_million
# ====================================================

# Set random seed for reproducibility
np.random.seed(42)

# Simulation parameters
T = 2000                      # number of simulated years
mu = 2.0                      # mean of log(Loss) (log scale, loss in million EUR)
sigma = 1.5                    # standard deviation of log(Loss)
avg_events_per_year = 3        # average number of events per year (Poisson mean)

# List to store all event records
events = []

# Loop over each year
for year in range(T):
    # Number of events in this year (Poisson distributed)
    n_events = np.random.poisson(avg_events_per_year)
    # Generate loss amounts for all events in this year
    losses_year = np.random.lognormal(mean=mu, sigma=sigma, size=n_events)
    
    # Create a record for each event
    for i, loss in enumerate(losses_year):
        group = np.random.randint(0,3)
        events.append({
            'year': year,
            'event_id': len(events),          # globally unique event ID
            'insured_loss_million': loss,
            'group':group
        })

# Convert to DataFrame
df = pd.DataFrame(events)


# Save to CSV (modify the path as needed)
df.to_csv('catastrophe_events.csv', index=False)
print(f"\nDataset saved with {len(df)} events across {T} years.")

print("Data generated and saved as 'catastrophe_scenarios.csv'")
print(df.head())
df.to_csv('/Users/green/Downloads/传输2/研究生/Quantum hackathon/catastrophe_scene.csv', index=False)