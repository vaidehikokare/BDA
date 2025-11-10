# MapReduce is a programming model used to process large amounts of data in parallel across many computers (or cores).
# Step	Description	Example
# Map	Breaks big data into smaller chunks and applies a function to each piece independently.	Count words in one line of text.
# Reduce	Combines (reduces) the results from the Map step into a single output.	Add up all the word counts from every line.
from multiprocessing import Pool
import pandas as pd
import sqlite3

# phool :running multiple tasks at the same time
def mapper(row):
    return (row["Month"], row["Temperature_Celsius"])


def reducer(mapped_data):
    result = {}
    for month, temp in mapped_data:
        result.setdefault(month, []).append(temp)
    return {m: sum(v) / len(v) for m, v in result.items()}


def run_mapreduce(df):
    with Pool() as p:
        mapped = p.map(mapper, [row for _, row in df.iterrows()])
    reduced = reducer(mapped)

    print("\n🌡️ Average Temperature per Month:")
    for m, t in reduced.items():
        print(f"{m}: {t:.2f}")
    return reduced


def top_fire_months(df, top_n=5):
    top = df.groupby("Month")["Burned_Area_hectares"].mean().sort_values(ascending=False).head(top_n)
    print(f"\n🔥 Top {top_n} Months with Largest Fire Area:\n{top}\n")
    return top


def temperature_area_correlation(df):
    corr = df["Temperature_Celsius"].corr(df["Burned_Area_hectares"])
    print(f"📊 Correlation between Temperature and Fire Area: {corr:.2f}")
    return corr


def query_avg_area_by_month(conn):
    query = """
        SELECT Month, AVG(Burned_Area_hectares) AS avg_area
        FROM forestfires
        GROUP BY Month
        ORDER BY avg_area DESC;
    """
    result = pd.read_sql_query(query, conn)
    print("\n🧾 Average Burned Area by Month (from SQL):")
    print(result)
    return result


def run_pipeline():
    print("=== 🌲 Forest Fire Analysis Pipeline Started ===\n")

    df = pd.read_csv("forestfires.csv")
    print(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    conn = sqlite3.connect("forestfires.db")
    df.to_sql("forestfires", conn, if_exists="replace", index=False)
    print("✅ Data saved to SQLite database.\n")

    run_mapreduce(df)
    top_fire_months(df)
    temperature_area_correlation(df)
    query_avg_area_by_month(conn)

    print("\n=== ✅ Pipeline Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline()
# Load Data – Reads the forestfires.csv file into a Pandas DataFrame.

# Save to Database – Stores the data in an SQLite database for SQL queries.

# MapReduce (Parallel Processing) – Calculates average temperature per month using multiprocessing:

# Map: Extract (Month, Temperature) pairs from each row.

# Reduce: Group by month and compute the average.

# Top Fire Months – Finds the months with the largest average burned area.

# Correlation – Computes the correlation between temperature and burned area.

# SQL Query – Queries the database to get average burned area per month, confirming Pandas results.

# Pipeline Orchestration – Runs all steps sequentially and prints results neatly.

# In short: It’s a forest fire analysis pipeline combining parallel processing, Pandas, and SQL to get insights like average temperature, top fire months, and correlations.