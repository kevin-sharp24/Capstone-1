# -*- coding: utf-8 -*-
#%%
import imblearn

#%% Read in data

import pandas as pd
import numpy as np

brfss = pd.read_csv('2014 BRFSS.csv')
print(brfss.head())

'''DtypeWarning: Columns (9,10,12,13,15,16,17,18,19,20,21,22,24,25,26,27,28,29,
31,32,33,34,35,36,37,38,39,40,42,43,44,45,46,47,48,50,51,53,54,55,56,58,59,60,
61,123,127,128,129,130,196,202,203,204,205,207,211,228,232,233,235,236,237,238) 
have mixed types.Specify dtype option on import or set low_memory=False.'''
#%% Set up diabetes features

df_diabetes = brfss[['GENHLTH','@_AGEG5YR','@_BMI5CAT','CHECKUP1',
                     'INCOME2','@_RACE','MSCODE','FLUSHOT6','EMPLOY1','SEX',
                     'MARITAL','@_EDUCAG','SLEPTIM1','CVDCRHD4','HLTHCVR1',
                     'MENTHLTH','CHCKIDNY','ADDEPEV2', 'USEEQUIP',
                     'RENTHOM1','EXERANY2','BLIND','DECIDE',
                     'DIABETE3', '@_SMOKER3']]

print(df_diabetes.info())
#%% Convert null response types to NaN
'''
NANs
general health: 7, 9, blank
age: 14
bmi category: blank
last checkup: 7, 9, blank
income: 77, 99, blank
race: 9, blank
MSCODE: ???
flu shot: 7, 9, blank
employed: 9, blank
sex: 7, 9
marital: 9, blank
education: 9
sleep per day: 77, 99, blank
CVDCRHD4: 7, 9, blank
health coverage: 77, 99, blank
mental health: 77, 99, blank (note: convert 88 to 0)
chckdny1: 7, 9, blank
TOTINDA (physical activity): 9
depressive disorder: 7, 9, blank
own or rent home: 7, 9, blank
any exercise: 7, 9, blank (note: same data as totinda, will be highly correlated)
use equip: 7, 9, blank
blind: 7, 9, blank
decide (confusion): 7, 9, blank
any health plan: 7, 9, blank (note: will be highly correlated to health coverage)
diabetes: 7, 9, blank
age of diabetes onset: 98, 99, blank
smoker: 9
'''

df_diabetes.loc[:,'GENHLTH'] = pd.to_numeric(df_diabetes.GENHLTH, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'@_AGEG5YR'] = df_diabetes['@_AGEG5YR'].replace(14, np.nan)
df_diabetes.loc[:,'@_BMI5CAT'] = pd.to_numeric(df_diabetes['@_BMI5CAT'], errors='coerce')
df_diabetes.loc[:,'CHECKUP1'] = pd.to_numeric(df_diabetes.CHECKUP1, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'INCOME2'] = pd.to_numeric(df_diabetes.INCOME2, errors='coerce').replace([77, 99], np.nan)
df_diabetes.loc[:,'@_RACE'] = pd.to_numeric(df_diabetes['@_RACE'], errors='coerce').replace(9, np.nan)
df_diabetes.loc[:,'MSCODE'] = pd.to_numeric(df_diabetes['MSCODE'], errors='coerce')
df_diabetes.loc[:,'FLUSHOT6'] = pd.to_numeric(df_diabetes.FLUSHOT6, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'EMPLOY1'] = pd.to_numeric(df_diabetes.EMPLOY1, errors='coerce').replace(9, np.nan)
df_diabetes.loc[:,'SEX'] = df_diabetes.SEX.replace([7, 9], np.nan)
df_diabetes.loc[:,'MARITAL'] = pd.to_numeric(df_diabetes.MARITAL, errors='coerce').replace(9, np.nan)
df_diabetes.loc[:,'@_EDUCAG'] = df_diabetes['@_EDUCAG'].replace(9, np.nan)
df_diabetes.loc[:,'SLEPTIM1'] = pd.to_numeric(df_diabetes.SLEPTIM1, errors='coerce').replace([77, 99], np.nan)
df_diabetes.loc[:,'CVDCRHD4'] = pd.to_numeric(df_diabetes.CVDCRHD4, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'HLTHCVR1'] = pd.to_numeric(df_diabetes.HLTHCVR1, errors='coerce').replace([77, 99], np.nan)
df_diabetes.loc[:,'MENTHLTH'] = pd.to_numeric(df_diabetes.MENTHLTH, errors='coerce').replace([77, 99], np.nan)\
    .replace(88, 0)
df_diabetes.loc[:,'CHCKIDNY'] = pd.to_numeric(df_diabetes.CHCKIDNY, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'ADDEPEV2'] = pd.to_numeric(df_diabetes.ADDEPEV2, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'RENTHOM1'] = pd.to_numeric(df_diabetes.RENTHOM1, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'EXERANY2'] = pd.to_numeric(df_diabetes.EXERANY2, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'USEEQUIP'] = pd.to_numeric(df_diabetes.USEEQUIP, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'BLIND'] = pd.to_numeric(df_diabetes.BLIND, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'DECIDE'] = pd.to_numeric(df_diabetes.DECIDE, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'DIABETE3'] = pd.to_numeric(df_diabetes.DIABETE3, errors='coerce').replace([7, 9], np.nan)
df_diabetes.loc[:,'@_SMOKER3'] = df_diabetes['@_SMOKER3'].replace(9, np.nan)

print(df_diabetes.info())

#for 2018, HLTHCVR1 is missing about 9/10 of its data, so rather than drop its NaNs, the row itself might be dropped.

#%% Drop rows according to study specifications

#keep respondants whose age is at least 30
df_diabetes = df_diabetes.loc[df_diabetes['@_AGEG5YR'] > 2]

#exclude those who were prediabetic or only diabetic when pregnant (strict 'yes' and 'no' responses kept)
df_diabetes = df_diabetes.loc[(df_diabetes['DIABETE3'] == 1) | (df_diabetes['DIABETE3'] == 3)]

#drop entries with empty values
df_diabetes = df_diabetes.dropna(axis=0, how='any')

print(df_diabetes.info())

#%% Make new categorical variables

df_diabetes.loc[:,'@_AGEG5YR'] = pd.cut(df_diabetes['@_AGEG5YR'], 
                                      bins=[2,4,6,8,10,12,14], 
                                      labels=[1,2,3,4,5,6])

df_diabetes.loc[:,'MENTHLTH'] = pd.cut(df_diabetes.MENTHLTH,
                                       bins=[-1,0,5,30],
                                       labels=[1, 2, 3])

df_diabetes.loc[:,'SLEPTIM1'] = pd.cut(df_diabetes.SLEPTIM1,
                                       bins=[0,6,8,24],
                                       labels=[1, 2, 3])
#%% Apply meaningful labels to data
'''
df_diabetes.loc[:,'GENHLTH'] = df_diabetes.loc[:,'GENHLTH'].\
    replace([1,2,3,4,5], ['1-Excellent','2-Very good', '3-Good', '4-Fair', '5-Poor'])

df_diabetes.loc[:,'@_BMI5CAT'] = df_diabetes.loc[:,'@_BMI5CAT'].replace([1,2,3,4],\
    ['18.5 or less (Underweight)','18.5-25 (Normal)','25-30 (Overweight)','30+ (Obese)'])

df_diabetes.loc[:,'CHECKUP1'] = df_diabetes.loc[:,'CHECKUP1'].replace([1,2,3,4,8],\
    ['12 or fewer months','between 1 and 2 years', 'between 2 and 5 years','more than 5 years','never'])

df_diabetes.loc[:,'INCOME2'] = df_diabetes.loc[:,'INCOME2'].replace([1,2,3,4,5,6,7,8],\
    ['$\$$10,000 or less','$\$$10,001 - $\$$15,000','$\$$15,001 - $\$$20,000','$\$$20,001 - $\$$25,000',
     '$\$$25,001 - $\$$35,000','$\$$35,001 - $\$$50,000','$\$$50,001 - $\$$75,000','$\$$75,000+'])

df_diabetes.loc[:,'@_RACE'] = df_diabetes.loc[:,'@_RACE'].replace([1,2,3,4,5,6,7,8],\
    ['White','Black','American Indian/Alaska Native','Asian',
     'Hawaiian/Pacific Islander','Other','Multiracial','Hispanic'])

df_diabetes.loc[:,'MSCODE'] = df_diabetes.loc[:,'MSCODE'].\
    replace([1,2,3,5],['In the center city of an MSA',
                     'Outside the center city of an MSA but inside the county containing the center city',
                     'Inside a suburban county of the MSA',
                     'Not in an MSA'])

df_diabetes.loc[:,'FLUSHOT6'] = df_diabetes.loc[:,'FLUSHOT6'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'EMPLOY1'] = df_diabetes.loc[:,'EMPLOY1'].\
    replace([1,2,3,4,5,6,7,8], ['Employed for wages','Self-employed',
                                'Out of work for 1 year or more',
                                'Out of work for less than 1 year',
                                'Homemaker', 'Student', 'Retired',
                                'Unable to work'])
    
df_diabetes.loc[:,'SEX'] = df_diabetes.loc[:,'SEX'].\
    replace([1,2], ['Male', 'Female'])
    
df_diabetes.loc[:,'MARITAL'] = df_diabetes.loc[:,'MARITAL'].\
    replace([1,2,3,4,5,6], ['Married', 'Divorced', 'Widowed', 'Separated',
                            'Never married', 'Unmarried couple'])
    
df_diabetes.loc[:,'@_EDUCAG'] = df_diabetes.loc[:,'@_EDUCAG'].\
    replace([1,2,3,4], ['1-Less than high school',
                        '2-Graduated high school',
                        '3-Some college/technical school', 
                        '4-Graduated college/technical school',])
    
df_diabetes.loc[:,'CVDCRHD4'] = df_diabetes.loc[:,'CVDCRHD4'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'HLTHCVR1'] = df_diabetes.loc[:,'HLTHCVR1'].\
    replace([1,2,3,4,5,6,7,8], ['Plan from employer/union','Plan from marketplace',
                                'Medicare','Medicaid/state program',
                                'TRICARE (formerly CHAMPUS), VA, or Military ',
                                'Alaska Native, Indian Health Service, Tribal Health Services',
                                'Other', 'None'])
    
df_diabetes.loc[:,'CHCKIDNY'] = df_diabetes.loc[:,'CHCKIDNY'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'ADDEPEV2'] = df_diabetes.loc[:,'ADDEPEV2'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'RENTHOM1'] = df_diabetes.loc[:,'RENTHOM1'].\
    replace([1,2,3], ['Own', 'Rent', 'Other'])
    
df_diabetes.loc[:,'EXERANY2'] = df_diabetes.loc[:,'EXERANY2'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'USEEQUIP'] = df_diabetes.loc[:,'USEEQUIP'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'BLIND'] = df_diabetes.loc[:,'BLIND'].\
    replace([1,2], ['Yes', 'No'])
    
df_diabetes.loc[:,'DECIDE'] = df_diabetes.loc[:,'DECIDE'].\
    replace([1,2], ['Yes', 'No'])   
    
df_diabetes.loc[:,'DIABETE3'] = df_diabetes.loc[:,'DIABETE3'].\
    replace([1,3], ['Yes', 'No'])
    
df_diabetes.loc[:,'@_SMOKER3'] = df_diabetes.loc[:,'@_SMOKER3'].\
    replace([1,2,3,4], ['Smokes every day', 'Smokes some days',
                        'Former smoker', 'Never smoked'])
'''
#%% rename columns    
df_diabetes.columns=['general_health','10yr_age_group','bmi_category','last_checkup',
                     'income','race','mscode','flushot','employed','sex','marital','education',
                     'sleep_time','angina_coronary_heart_disease','health_coverage',
                     'mental_health_days_per_month','kidney_disease',
                     'depressive_disorder','uses_medical_equipment','rent_or_own_home',
                     'any_exercise','blindness','trouble_concentrating','diabetes', 'smoker']
    
print(df_diabetes.info())

#%% Save to .csv
df_diabetes.to_csv('2014_BRFSS_encoded.csv', index=False)