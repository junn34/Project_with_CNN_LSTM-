from fredapi import Fred
import pandas as pd
import os

fred = Fred(api_key='7faa1595daf7911ed2a907d11dc20a2d')

start_date = '2005-01-01'
end_date = '2025-04-30'

fred_ids = [
    "KORCP010000IXOBM", "KORCP020000IXOBM", "KORCP030000IXOBM", "KORCP040000IXOBM",
    "KORCP050000IXOBM", "KORCP060000IXOBM", "KORCP070000IXOBM", "KORCP080000IXOBM", "KORCP090000IXOBM",
    "KORCP100000IXOBM", "KORCP110000IXOBM", "KORCP120000IXOBM", "KORCPICORMINMEI", "KORCPIENGMINMEI",
    "KORCPGRSE01IXOBM", "KORCPGRLH02IXOBM", "KORCPGRHO02IXOBM", "KORCP040100IXOBM", "KORCP040400IXOBM",
    "KORCP040500IXOBM", "KORCP040300IXOBM"
]

column_rename_map = {
    "KORCP010000IXOBM": "Food and non-alcoholic beverages",
    "KORCP020000IXOBM": "Alcoholic beverages, tobacco and narcotics",
    "KORCP030000IXOBM": "Clothing and footwear",
    "KORCP040000IXOBM": "Housing, water, electricity, and fuel",
    "KORCP050000IXOBM": "Household goods and services",
    "KORCP060000IXOBM": "Health",
    "KORCP070000IXOBM": "Transportation",
    "KORCP080000IXOBM": "Communication",
    "KORCP090000IXOBM": "Recreation and culture",
    "KORCP100000IXOBM": "Education",
    "KORCP110000IXOBM": "Restaurants and hotels",
    "KORCP120000IXOBM": "Miscellaneous goods and services",
    "KORCPICORMINMEI": "All items (non-food non-energy)",
    "KORCPIENGMINMEI": "Energy",
    "KORCPGRSE01IXOBM": "Services",
    "KORCPGRLH02IXOBM": "Services less housing",
    "KORCPGRHO02IXOBM": "Housing excluding imputed rentals for housing",
    "KORCP040100IXOBM": "Actual rentals for housing",
    "KORCP040400IXOBM": "Water supply and misc. services relating to dwelling",
    "KORCP040500IXOBM": "Electricity, gas and other fuels",
    "KORCP040300IXOBM": "Maintenance and repair of the dwelling"
}

data = pd.DataFrame()

for fred_id in fred_ids:
    try:
        series = fred.get_series(fred_id, start_date, end_date)
        series.name = column_rename_map[fred_id]
        data = pd.concat([data, series], axis=1)
    except Exception as e:
        print(f"Error fetching {fred_id}: {e}")

data.index = pd.to_datetime(data.index)
data.index.name = 'Date'

cpi_df = pd.read_csv("C:/Users/bjh20/source/repos/딥러닝/딥러닝/CPI.csv", encoding="utf-8-sig")
cpi_df = cpi_df.rename(columns={cpi_df.columns[0]: "Date", cpi_df.columns[1]: "Total CPI"})
cpi_df["Date"] = pd.to_datetime(cpi_df["Date"])
cpi_df.set_index("Date", inplace=True)

data = pd.concat([cpi_df, data], axis=1)

output_dir = 'C:/Users/bjh20/source/repos/딥러닝/딥러닝'
os.makedirs(output_dir, exist_ok=True)

data.to_csv(os.path.join(output_dir, 'merged_data_2025.csv'))

print("저장 완료: merged_data_2025.csv")
