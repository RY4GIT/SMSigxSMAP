# This script samples SMAP data using point information (geographic coordinates)
# The Appears' API documentation is here: https://appeears.earthdatacloud.nasa.gov/api/?python#introduction
# The Appears' website is here: https://appeears.earthdatacloud.nasa.gov/

import json
import requests
import os

# Specify current directory create output directory if it does not exist
os.chdir("G:/Shared drives/Ryoko and Hilary/SMSigxSMAP/analysis/0_code")

## Some parameters
params = {'pretty': True}


# The data I am looking at is SMAP Enhanced L3 Radiometer Global and Polar Grid Daily 9 km EASE-Grid Soil Moisture, Version 5 (SPL3SMP_E)
# https://nsidc.org/data/spl3smp_e/versions/5#anchor-2

# A list of product code is here
# https://appeears.earthdatacloud.nasa.gov/products

## Request a sample
product_id = 'SPL3SMP_E.005'
response = requests.get(
    'https://appeears.earthdatacloud.nasa.gov/api/product/{0}'.format(product_id),
    params=params)
product_response = response.text
print(product_response)

# Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag (have different versions)
# Soil_Moisture_Retrieval_Data_AM_soil_moisture
# Soil_Moisture_Retrieval_Data_AM_soil_moisture_error
# Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm
# Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm
# Soil_Moisture_Retrieval_Data_PM_soil_moisture_error_pm

## Prepare tasks
# load the task request from a file
sample_request_path = "./sample_point_request.json"
with open(sample_request_path) as json_file:
    task = json.load(json_file)

## Login to appears
my_credential_path = "./auth.json"
with open(my_credential_path, 'r') as infile:
    my_credentials = json.load(infile)

response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=(my_credentials['username'], my_credentials['password']))
token_response = response.json()
print(token_response)

# submit the task request
token = token_response['token']
response = requests.post(
    'https://appeears.earthdatacloud.nasa.gov/api/task',
    json=task,
    headers={'Authorization': 'Bearer {0}'.format(token)})
task_response = response.json()
print(task_response)

# To check the updates
task_id = task_response['task_id']
response = requests.get(
    'https://appeears.earthdatacloud.nasa.gov/api/task/{0}'.format(task_id),
    headers={'Authorization': 'Bearer {0}'.format(token)}
)
task_response = response.json()
print(task_response)
print(task_response['status'])

# To bundle download
response = requests.get(
    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(task_id),
    headers={'Authorization': 'Bearer {0}'.format(token)}
)
bundle_response = response.json()
print(bundle_response)
len(bundle_response)
for i in range(len(bundle_response)):
    print(bundle_response['files'][i])

    # get a stream to the bundle file
    file_id = bundle_response['files'][i]['file_id']
    filename = file_id
    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(task_id,file_id),
        headers={'Authorization': 'Bearer {0}'.format(token)},
        allow_redirects=True,
        stream=True
    )

    # create a destination directory to store the file in
    dest_dir = "../1_data"
    filepath = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # write the file to the destination directory
    with open(filepath, 'wb') as f:
        for data in response.iter_content(chunk_size=8192):
            f.write(data)

# Point sampling returns the results as csv file
# Area sampling returns the results as nc file