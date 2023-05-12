import pandas as pd
import numpy as np

def piece_wise(X,thresholds):
    # thresholds is a list
    thresholds = sorted(thresholds)
    X = np.array(X)
    result = []
    for i in range(len(thresholds)-1):
        result.append(np.piecewise(X, [(X < thresholds[i+1]) & (X >= thresholds[i]),\
                                       (X >= thresholds[i+1])| (X < thresholds[i])], [1, 0]))
    return result
data = pd.read_csv('London_dataset/lpmc.dat',sep = '	')
# trip purpose
temp = pd.get_dummies(data['purpose'])
data = pd.concat([data, temp], axis=1)
data = data.rename(columns={1: 'purpose_HBW', 2: 'purpose_HBE', 3: 'purpose_HBO',4: 'purpose_EB', 5: 'purpose_NHBO'})
list_var = ['purpose_HBW','purpose_HBE','purpose_HBO','purpose_EB','purpose_NHBO']
for key in list_var:
    print(key,sum(data[key]))

#fueltype
temp = pd.get_dummies(data['fueltype'])
data = pd.concat([data, temp], axis=1)
data = data.rename(columns={1: 'fuel_petrol', 2: 'fuel_diesel', 3: 'fuel_hybrid',4: 'fuel_petrolLGV', 5: 'fuel_dieselLGV',6:'fuel_avg'})
list_var = ['fuel_petrol','fuel_diesel','fuel_hybrid','fuel_petrolLGV','fuel_dieselLGV','fuel_avg']

for key in list_var:
    print(key, sum(data[key]))

#faretype
temp = pd.get_dummies(data['faretype'])
data = pd.concat([data, temp], axis=1)
data = data.rename(columns={1: 'fare_full', 2: 'fare_16plus', 3: 'fare_child',4: 'fare_disabled', 5: 'fare_free'})
list_var = ['fare_full','fare_16plus','fare_child','fare_disabled','fare_free']
for key in list_var:
    print(key,sum(data[key]))

#survey_year
temp = pd.get_dummies(data['survey_year'])
data = pd.concat([data, temp], axis=1)
data = data.rename(columns={1: 'year_1213', 2: 'year_1314', 3: 'year_1415'})
list_var = ['year_1213','year_1314','year_1415']
for key in list_var:
    print(key,sum(data[key]))


# trip time
data['travel_SPRING'] = 0
data['travel_SUMMER'] = 0
data['travel_AUTUMN'] = 0
data['travel_WINTER'] = 0
data.loc[(data['travel_month'] == 3) |
         (data['travel_month'] == 4) |
         (data['travel_month'] == 5) , 'travel_SPRING'] = 1
data.loc[(data['travel_month'] == 6) |
         (data['travel_month'] == 7) |
         (data['travel_month'] == 8), 'travel_SUMMER'] = 1
data.loc[(data['travel_month'] == 9) |
         (data['travel_month'] == 10) |
         (data['travel_month'] == 11), 'travel_AUTUMN'] = 1
data.loc[(data['travel_month'] == 12) |
         (data['travel_month'] == 1) |
         (data['travel_month'] == 2), 'travel_WINTER'] = 1
list_var = ['travel_SPRING','travel_SUMMER','travel_AUTUMN','travel_WINTER']
for key in list_var:
    print(key,sum(data[key]))



# trip year
data['travel_2012'] = 0
data['travel_2013'] = 0
data['travel_2014'] = 0
data['travel_2015'] = 0
data.loc[(data['travel_year'] == 2012) , 'travel_2012'] = 1
data.loc[(data['travel_year'] == 2013) , 'travel_2013'] = 1
data.loc[(data['travel_year'] == 2014) , 'travel_2014'] = 1
data.loc[(data['travel_year'] == 2015) , 'travel_2015'] = 1
list_var = ['travel_2012','travel_2013','travel_2014','travel_2015']
for key in list_var:
    print(key,sum(data[key]))


#day_of_week
data['travel_Mon'] = 0
data['travel_Tue2Thu'] = 0
data['travel_Fri'] = 0
data['travel_Sat'] = 0
data['travel_Sun'] = 0
data.loc[(data['day_of_week'] == 1) , 'travel_Mon'] = 1
data.loc[(data['day_of_week'] == 5) , 'travel_Fri'] = 1
data.loc[(data['day_of_week'] == 6) , 'travel_Sat'] = 1
data.loc[(data['day_of_week'] == 7) , 'travel_Sun'] = 1
data.loc[(data['day_of_week'] == 2) |
         (data['day_of_week'] == 3) |
         (data['day_of_week'] == 4), 'travel_Tue2Thu'] = 1
list_var = ['travel_Mon','travel_Tue2Thu','travel_Fri','travel_Sat','travel_Sun']
for key in list_var:
    print(key,sum(data[key]))


# Trip trip time
thresholds = [0, 6, 10, 14, 17, 20, 24]
result = piece_wise(data['start_time'], thresholds)
data['start_time_morning'] = result[0]
data['start_time_morningpeak'] = result[1]
data['start_time_noon'] = result[2]
data['start_time_afternoon'] = result[3]
data['start_time_evening_peak'] = result[4]
data['start_time_evening'] = result[5]
list_var = ['start_time_morning','start_time_morningpeak','start_time_noon','start_time_afternoon','start_time_evening_peak','start_time_evening']
for key in list_var:
    print(key,sum(data[key]))


all_columns = list(data.columns)
data_input_var = pd.DataFrame({'Input_variables':all_columns,'Mode_choice_input':[0]*len(all_columns),'Normalize':[0]*len(all_columns)})
data_input_var.to_csv('London_dataset/Input_variables_London.csv',index=False)

data.to_csv('London_dataset/data_input_London.csv',index=False)