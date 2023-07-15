import os
from batch import read_data, get_output_path

year = 2022
month = 2

print(f'Running batch.py for {year}-{month}')
os.system(f"python batch.py {year} {month}")

output_file = get_output_path(year, month)

data = read_data(output_file)
print(data)
