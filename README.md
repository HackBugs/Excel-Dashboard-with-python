# Excel-Dashboard-with-python

## This script is a Python program that generates a shift schedule for a team of 5 people from April to December. It creates an Excel file with the schedule. To run this script, you'll need to install several Python packages.

Here's what you need to install and how to do it:

1. First, make sure you have Python installed on your system. The script uses Python's standard libraries and some additional packages.
2. Install the required packages using pip:

```
pip install pandas
```

```
pip install openpyxl
```

## The script uses:

- **calendar** (built-in Python module, no installation needed)
- **pandas** (for data manipulation and creating DataFrames)
- **datetime** (built-in Python module, no installation needed)
- **openpyxl** (for Excel file operations)

## After installing these packages, you should be able to run the script without any errors. The script will:

1. Generate a shift schedule for each day from April to December
2. Rotate team members to ensure fair distribution of shifts
3. Create an Excel file named "shift_schedule_April_December.xlsx" with the schedule

## To run the script, save it as a .py file (e.g., "shift_scheduler.py") and execute it using Python:

```
python shift_scheduler.py
```

