# Data

The dataset used in this work is based on the **NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets**.

## Source
Public dataset website:
- NIST Campus Photovoltaic (PV) Arrays and Weather Station Data Sets

This repository does **not** include the complete raw dataset because of its large size.

## Description
According to the public dataset description, the data contain photovoltaic array measurements and weather station observations collected on the NIST campus. The original public source covers multiple years and includes PV and meteorological measurements.

## Repository Data Policy
To keep this repository lightweight and easy to access:

- The **full dataset** is not uploaded to this repository.
- Only a **small number of sample CSV files** are provided in `data/sample/` for illustrating the data format.
- Users should download the complete dataset from the official public source and place the required files under the `data/` directory before running the code.

## Recommended Directory Structure

```text
data/
├─ README.md
├─ sample/
│  ├─ sample_pv_2016.csv
│  └─ sample_weather_2016.csv
└─ full_dataset/   # user-prepared complete dataset, not included in repo
