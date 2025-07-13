import numpy as np

# Input data
demand = np.array([1,2,2,0,0,5,6,5,3,3,5,5,4,3,3,4,5,2,3,5,10,7,4,3,5,4,2,12,9,6,3,7,4,3,7,10,9,8,4])
long_term_cost_per_worker = 39 * 130  # €5070 (per worker for 39 days)
short_term_cost_per_day = 150         # €150 per worker per day

# Calculate total cost for each M value
def calculate_cost(M):
    shortage = np.maximum(demand - M, 0)  # Daily worker shortage
    total_short_days = np.sum(shortage)
    total_cost = M * long_term_cost_per_worker + total_short_days * short_term_cost_per_day
    return total_cost, total_short_days

# Evaluate all possible M values (0 to max demand)
max_demand = np.max(demand)
possible_M = range(0, max_demand + 1)
results = []

for M in possible_M:
    total_cost, short_days = calculate_cost(M)
    results.append({
        'M': M,
        'total_cost': total_cost,
        'long_term_cost': M * long_term_cost_per_worker,
        'short_term_cost': short_days * short_term_cost_per_day,
        'short_days': short_days
    })

# Find optimal solution
optimal = min(results, key=lambda x: x['total_cost'])

# Print results
print("39-day total demand:", sum(demand), "worker-days")
print("Optimal long-term drivers (M):", optimal['M'])
print("Total cost: €", optimal['total_cost'])
print("  - Long-term driver cost: €", optimal['long_term_cost'], f"({optimal['M']} workers × €5070)")
print("  - Short-term worker cost: €", optimal['short_term_cost'], f"(covering {optimal['short_days']} worker-days)")
print("\n==== Cost Comparison Table ====")
print("M\tTotal Cost\tLong-term Cost\tShort-term Cost\tShort-term Days")
for r in results:
    print(f"{r['M']}\t€{r['total_cost']}\t€{r['long_term_cost']}\t€{r['short_term_cost']}\t{r['short_days']}")