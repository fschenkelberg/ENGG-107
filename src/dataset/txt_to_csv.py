import csv

def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        if len(lines) == 3:
            for line in lines:
                src, dst, unixts = line.strip().split()
                data.append((src, dst, unixts))
        elif len(lines) == 2:
            for line in lines:
                node_id, department = line.strip().split()
                data.append((node_id, department))
    return data

def write_data_to_csv(data, csv_filename, header):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

filename = "input_file.txt"
csv_filename = "output_data.csv"

try:
    data = read_data_from_file(filename)
    if len(data) == 3:
        header = ['SRC', 'DST', 'UNIXTS']
    elif len(data) == 2:
        header = ['NODEID', 'DEPARTMENT']
    write_data_to_csv(data, csv_filename, header)
    print(f"Data successfully written to '{csv_filename}'.")
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
except Exception as e:
    print("An error occurred:", e)
