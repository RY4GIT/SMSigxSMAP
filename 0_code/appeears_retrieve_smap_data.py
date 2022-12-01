# ======================= THE SCRIPT ===============================
## This script samples SMAP data using point information (geographic coordinates)
# The Appears' API documentation is here: https://appeears.earthdatacloud.nasa.gov/api/?python#introduction
# The Appears' website is here: https://appeears.earthdatacloud.nasa.gov/

# ======================= THE DATA ===============================
# This script downloads SMAP Enhanced L3 Radiometer Global and Polar Grid Daily 9 km EASE-Grid Soil Moisture, Version 5 (SPL3SMP_E)
# https://nsidc.org/data/spl3smp_e/versions/5#anchor-2
# TODO: which product to use, L2 or L3?

# Layers downloaded are as follows:
# Soil_Moisture_Retrieval_Data_AM_retrieval_qual_flag
# Soil_Moisture_Retrieval_Data_AM_soil_moisture
# Soil_Moisture_Retrieval_Data_AM_soil_moisture_error
# Soil_Moisture_Retrieval_Data_PM_retrieval_qual_flag_pm
# Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm
# Soil_Moisture_Retrieval_Data_PM_soil_moisture_error_pm
# MOD15A2H.061 LAI 500m

# The product includes includes soil moisture retrievals from three algorithms:
# the ones I picked (the ones without notation) are from Dual Channel Algorithm (DCA)

# A list of product code is here
# https://appeears.earthdatacloud.nasa.gov/products

# ======================= START OF THE CODE ===============================

# Import libraries
import json
import requests
import os

# ============ User-defined parameters ==========# 
# Specify current directory and create output directory if it does not exist
os.chdir("G:/Shared drives/Ryoko and Hilary/SMSigxSMAP/analysis/0_code")

# Get the target point locations 
network_name = 'KENYA'

# File path to the sample request json
sample_request_path = "./appeears_request_jsons/point_request_Kenya.json"

# Specify the output directory
dest_dir = os.path.join("../1_data/APPEEARS_subsetting", network_name)

# How often would you want to check the results in seconds
check_request_interval_sec = 60*10

# ===================================================# 

# Parameters
params = {'pretty': True}

## Login to appears
my_credential_path = "./auth.json"
with open(my_credential_path, 'r') as infile:
    my_credentials = json.load(infile)

response = requests.post('https://appeears.earthdatacloud.nasa.gov/api/login', auth=(my_credentials['username'], my_credentials['password']))
token_response = response.json()
print(token_response)

# Get a request from json file
with open(sample_request_path) as json_file:
    task = json.load(json_file)
    
# Submit the task request
token = token_response['token']
response = requests.post(
    'https://appeears.earthdatacloud.nasa.gov/api/task',
    json=task,
    headers={'Authorization': 'Bearer {0}'.format(token)}
)
task_response = response.json()
print(task_response)

# To check the updates every x seconds
task_id = task_response['task_id']

import time
import datetime
starttime = time.time()
while task_response['status'] =='pending' or task_response['status']=='processing':
    print("Still processing the request at %s" % {datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')})
    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/task/{0}'.format(task_id),
        headers={'Authorization': 'Bearer {0}'.format(token)}
    )
    task_response = response.json()
    time.sleep(check_request_interval_sec - ((time.time() - starttime) % check_request_interval_sec)) # check the request every 300 sec
print("Done processing on Appears' side")

# To bundle download
# TODO: Avoid overwriting the previous data
response = requests.get(
    'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}'.format(task_id),
    headers={'Authorization': 'Bearer {0}'.format(token)}
)
bundle_response = response.json()
print(bundle_response)
len(bundle_response)
for i in range(len(bundle_response)):

    # get a stream to the bundle file
    file_id = bundle_response['files'][i]['file_id']
    filename = bundle_response['files'][i]['file_name']
    print("Download a file %s" % filename)

    response = requests.get(
        'https://appeears.earthdatacloud.nasa.gov/api/bundle/{0}/{1}'.format(task_id,file_id),
        headers={'Authorization': 'Bearer {0}'.format(token)},
        allow_redirects=True,
        stream=True
    )

    # create a destination directory to store the file in
    filepath = os.path.join(dest_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # write the file to the destination directory
    with open(filepath, 'wb') as f:
        for data in response.iter_content(chunk_size=8192):
            f.write(data)


# ======================= END OF THE CODE ===============================
