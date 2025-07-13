import numpy as np

# Input two independent demand sequences (same length)
demands_small = [0,1,1,0,0,1,2,0,1,2,2,1,1,2,0,0,1,1,1,0,1,4,1,2,2,0,1,2,3,0,2,2,1,2,3,2,3,1,2]
demands_large = [1,1,1,0,0,4,4,5,2,1,3,4,3,1,3,4,4,1,2,5,9,3,3,1,3,4,1,10,6,6,1,5,3,1,4,8,6,7,2]

# Parameter definitions
C_purchase_small = 35552  # Small vehicle purchase cost
C_purchase_large = 36250  # Large vehicle purchase cost
C_rent_small = 71        # Small vehicle daily rental cost
C_rent_large = 82        # Large vehicle daily rental cost

years = 5
days_total = 365 * years
days_per_cycle = len(demands_small)
cycles = days_total // days_per_cycle

# Exhaustive search of purchase quantities (assume buying up to max demand)
max_small = max(demands_small)
max_large = max(demands_large)

best_combo = None
min_cost = float('inf')

for buy_small in range(max_small + 1):     # Number of small vehicles to purchase
    for buy_large in range(max_large + 1): # Number of large vehicles to purchase
        total_rent_cost = 0

        for daily_small, daily_large in zip(demands_small, demands_large):
            rent_small = max(0, daily_small - buy_small)
            rent_large = max(0, daily_large - buy_large)

            daily_rent_cost = rent_small * C_rent_small + rent_large * C_rent_large
            total_rent_cost += daily_rent_cost * cycles

        purchase_cost = buy_small * C_purchase_small + buy_large * C_purchase_large
        total_cost = total_rent_cost + purchase_cost

        if total_cost < min_cost:
            min_cost = total_cost
            best_combo = (buy_small, buy_large)

print(f"Optimal purchase combination: {best_combo[0]} small vehicles, {best_combo[1]} large vehicles")
print(f"Total 5-year cost: {min_cost:.2f}")