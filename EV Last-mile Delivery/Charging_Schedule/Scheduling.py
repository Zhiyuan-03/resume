import pulp
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def optimize_multi_day_charging():
    excel_file_path = r"C:\Users\30572\Desktop\Vehicle_Charging_Data.xlsx"

    try:
        # Read data
        price_df = pd.read_excel(excel_file_path, sheet_name="Price")
        demand_df = pd.read_excel(excel_file_path, sheet_name="Mileage_and_Cars")

        # Process price data (convert from €/MWh to €/kWh by dividing by 1000)
        price_df['Time'] = pd.to_datetime(price_df['Time'])
        price_df['Date'] = price_df['Time'].dt.date
        price_df['Hour'] = price_df['Time'].dt.hour
        price_df['price'] = (
                price_df['price']
                .astype(str)
                .str.replace('€', '')
                .str.replace(',', '.')
                .astype(float) / 1000  # Convert from €/MWh to €/kWh
        )
        price_df = price_df.sort_values(['Date', 'Hour'])

        unique_dates = price_df['Date'].unique()
        price_lists = []
        for date in unique_dates:
            daily_prices = price_df[price_df['Date'] == date].sort_values('Hour')['price'].tolist()
            if len(daily_prices) != 24:
                avg_price = sum(daily_prices) / len(daily_prices)
                daily_prices = [avg_price] * 24
                print(f"Warning: Incomplete price data for {date}, filled with average price")
            price_lists.append(daily_prices)

        # Process demand data
        km_totals = demand_df.iloc[:, 1].fillna(0).tolist()
        car_counts = demand_df.iloc[:, 2].fillna(0).astype(int).tolist()

        # Pad data if needed
        days = len(price_lists)
        if len(km_totals) < days:
            km_totals += [0] * (days - len(km_totals))
        if len(car_counts) < days:
            car_counts += [0] * (days - len(car_counts))

    except Exception as e:
        print(f"Error reading or processing Excel file: {e}")
        return None

    # Optimization model
    model = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMinimize)

    N = pulp.LpVariable("N", lowBound=0, cat="Integer")  # Number of chargers
    x = {}  # Charger usage per hour per day
    total_cost = 48.1 * N  # Infrastructure cost

    for d in range(days):
        for h in range(24):
            x[(d, h)] = pulp.LpVariable(f"x_{d}_{h}", lowBound=0)
            # Cost calculation adjusted for €/kWh (already converted)
            total_cost += price_lists[d][h] * x[(d, h)] * 100  # 100 kW power

    model += total_cost

    # Constraints: Charging only between 22:00 - next day 08:00
    for d in range(days):
        km = km_totals[d]
        cars = car_counts[d]
        if cars == 0:
            continue  # No demand
        effective_km = min(cars, 7) / cars * km
        required_hours = 0.0875 * effective_km / 60  # Convert to hours

        valid_hours = []
        if d < days:  # Current day 22-23
            valid_hours += [(d, 22), (d, 23)]
        if d + 1 < days:  # Next day 0-7
            for h in range(8):
                valid_hours.append((d + 1, h))

        model += pulp.lpSum([x[(dd, hh)] for (dd, hh) in valid_hours]) == required_hours

    # Charger capacity constraint
    for d in range(days):
        for h in range(24):
            model += x[(d, h)] <= N

    model.solve()

    # Extract results and calculate costs
    result = {
        "num_chargers": int(pulp.value(N)),
        "total_cost": pulp.value(model.objective),
        "schedule": {},
        "prices": price_lists,
        "dates": unique_dates,
        "km_totals": km_totals,
        "car_counts": car_counts,
        "daily_costs": {},  # Daily electricity costs
        "total_electricity_cost": 0  # Total electricity cost
    }

    total_electricity_cost = 0

    for d in range(days):
        date_str = str(unique_dates[d])
        result["schedule"][date_str] = []
        daily_cost = 0

        for h in range(24):
            val = pulp.value(x[(d, h)]) if pulp.value(x[(d, h)]) else 0.0
            result["schedule"][date_str].append(val)
            # Calculate cost: power(100kW) × time(hours) × price(€/kWh)
            daily_cost += val * 100 * price_lists[d][h]

        result["daily_costs"][date_str] = daily_cost
        total_electricity_cost += daily_cost

    result["total_electricity_cost"] = total_electricity_cost
    result["infrastructure_cost"] = 3000 * result["num_chargers"]
    result["price_unit"] = "€/kWh"  # Indicate the price unit used

    return result



def visualize_results(result):
    plt.figure(figsize=(15, 10))

    # Electricity price visualization
    plt.subplot(2, 1, 1)
    for d in range(min(7, len(result["dates"]))):
        plt.plot(range(24), result["prices"][d], label=result["dates"][d])
    plt.title('Electricity Price Trend (22:00-08:00)')
    plt.xlabel('Hour')
    plt.ylabel('Price (€/kWh)')
    plt.axvspan(22, 24, color='lightgray', alpha=0.3)
    plt.axvspan(0, 8, color='lightgray', alpha=0.3)
    plt.legend()
    plt.grid(True)

    # Charging schedule visualization
    plt.subplot(2, 1, 2)
    for d in range(min(7, len(result["dates"]))):
        date_str = str(result["dates"][d])
        charging_hours = [result["schedule"][date_str][h] for h in range(24)]
        plt.bar(range(24), charging_hours, alpha=0.5, label=date_str)
    plt.title('Charging Schedule Distribution (22:00-08:00)')
    plt.xlabel('Hour')
    plt.ylabel('Charging Time (hours)')
    plt.axvspan(22, 24, color='lightgray', alpha=0.3)
    plt.axvspan(0, 8, color='lightgray', alpha=0.3)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_detailed_schedule(result):
    print("\n" + "=" * 70)
    print(f"Optimal number of chargers: {result['num_chargers']}")
    print(f"Infrastructure cost: {result['infrastructure_cost']:.2f} €")
    print(f"Total electricity cost: {result['total_electricity_cost']:.2f} €")
    print(f"Total estimated cost: {result['total_cost']:.2f} €")
    print("=" * 70 + "\n")

    # Organize by business day
    for d in range(len(result["dates"])):
        business_date = result["dates"][d]
        km = result["km_totals"][d]
        cars = result["car_counts"][d]

        if cars == 0:
            print(f"\nBusiness Date: {business_date} (No charging needed)")
            continue

        print(f"\nBusiness Date: {business_date}")
        print(f"Mileage: {km} km, Vehicles: {cars}")
        print("-" * 80)
        print("Charging Time\tPrice(€/kWh)\tHours\t\tPower(kW)\tCost(€)")

        # Current day 22-23
        daily_electricity_cost = 0
        for h in [22, 23]:
            price = result["prices"][d][h]
            charging = result["schedule"][str(business_date)][h]
            power = 100  # kW
            cost = charging * power * price
            print(f"{business_date} {h:02d}:00\t{price:.2f}\t\t{charging:.2f}\t\t{power}\t\t{cost:.2f}")
            daily_electricity_cost += cost

        # Next day 0-7
        if d + 1 < len(result["dates"]):
            next_date = result["dates"][d + 1]
            for h in range(8):
                price = result["prices"][d + 1][h]
                charging = result["schedule"][str(next_date)][h]
                power = 100  # kW
                cost = charging * power * price
                print(f"{next_date} {h:02d}:00\t{price:.2f}\t\t{charging:.2f}\t\t{power}\t\t{cost:.2f}")
                daily_electricity_cost += cost

        print(
            f"\nTotal charging time: {sum(result['schedule'][str(business_date)][22:24]) + sum(result['schedule'][str(result['dates'][d + 1])][:8]):.2f} hours")
        print(f"Daily electricity cost: {daily_electricity_cost:.2f} €")
        print("-" * 80)


def save_schedule_to_excel(result, output_file):
    output_data = []

    for d in range(len(result["dates"])):
        business_date = result["dates"][d]
        km = result["km_totals"][d]
        cars = result["car_counts"][d]

        if cars == 0:
            output_data.append({
                'Business Date': business_date,
                'Charging Date': business_date,
                'Charging Time': 'No charging needed',
                'Price(€/kWh)': 0,
                'Charging Hours': 0,
                'Power(kW)': 0,
                'Cost(€)': 0,
                'Mileage(km)': km,
                'Vehicles': cars
            })
            continue

        # Current day 22-23
        for h in [22, 23]:
            price = result["prices"][d][h]
            charging = result["schedule"][str(business_date)][h]
            power = 100
            cost = charging * power * price
            output_data.append({
                'Business Date': business_date,
                'Charging Date': business_date,
                'Charging Time': f"{h:02d}:00-{h + 1:02d}:00",
                'Price(€/kWh)': price,
                'Charging Hours': charging,
                'Power(kW)': power,
                'Cost(€)': cost,
                'Mileage(km)': km,
                'Vehicles': cars
            })

        # Next day 0-7
        if d + 1 < len(result["dates"]):
            next_date = result["dates"][d + 1]
            for h in range(8):
                price = result["prices"][d + 1][h]
                charging = result["schedule"][str(next_date)][h]
                power = 100
                cost = charging * power * price
                output_data.append({
                    'Business Date': business_date,
                    'Charging Date': next_date,
                    'Charging Time': f"{h:02d}:00-{h + 1:02d}:00",
                    'Price(€/kWh)': price,
                    'Charging Hours': charging,
                    'Power(kW)': power,
                    'Cost(€)': cost,
                    'Mileage(km)': km,
                    'Vehicles': cars
                })

    # Add summary row
    output_data.append({
        'Business Date': "TOTAL",
        'Charging Date': "",
        'Charging Time': "",
        'Price(€/kWh)': "",
        'Charging Hours': "",
        'Power(kW)': "",
        'Cost(€)': result["total_electricity_cost"],
        'Mileage(km)': "",
        'Vehicles': ""
    })

    output_df = pd.DataFrame(output_data)
    output_df.to_excel(output_file, index=False)
    print(f"\nDetailed charging schedule saved to: {output_file}")


if __name__ == "__main__":
    print("Starting EV charging optimization...")
    result = optimize_multi_day_charging()

    if result:
        print_detailed_schedule(result)
        visualize_results(result)

        output_path = r"C:\Users\30572\Desktop\Charging_Schedule_Results.xlsx"
        save_schedule_to_excel(result, output_path)

        print("\nOptimization completed! Key results:")
        print(f"- Recommended number of chargers: {result['num_chargers']}")
        print(f"- Infrastructure cost: {result['infrastructure_cost']:.2f} €")
        print(f"- Total electricity cost: {result['total_electricity_cost']:.2f} €")
        print(f"- Total estimated cost: {result['total_cost']:.2f} €")