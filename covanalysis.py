import pandas as pd
import numpy as np
import nltk
import re

import math
import os

# nltk_path=os.path.abspath('./nltk_data')
# #print('My Path: ', path)
# nltk.data.path.append('./nltk_data')


class Covid():

    def __init__(self,data=None):
        self.data=data

    def symptom_fequency(self,data, feature, n):

        """
        This function will collect all of the symptoms present
        and return a list of all the symptoms

        """
        self.feature=feature
        self.n=n
        self.data=data

        symptoms = []

        replace_symp = {'chills': ['chill'],
                        'cough': ['coughing', 'mild cough'],
                        'fever': ['high fever', 'feaver', 'feve\\', 'mild fever'],
                        'sore throat': ['throat discomfort', 'throat pain'],
                        'difficulty breathing': ['difficulty in breathing', 'difficult in breathing', 'breathlessness'],
                        'muscle ache': ['aching muscles', 'muscle pain', 'muscle aches', 'sore body', 'myalgia', 'myalgias',
                                        'muscle cramps'],
                        'sputum': ['cough with sputum'],
                        'fatigue': ['tired'],
                        'nasal discharge': ['runy nose'],
                        'flu symptoms': ['flu']}

        for item in self.data[self.data[self.feature].isnull() == False][self.feature]:
            symptoms.append(item)

        symptoms = [item.split(',') for item in symptoms]

        symptoms = [x.strip() for item in symptoms for x in item]

        symp_fd = nltk.FreqDist(symptoms)

        top_sym = [x[0] for x in symp_fd.most_common(n)]

        symp_nan = self.data[self.data[self.feature].isnull() == True][self.feature].index

        for item in self.data.loc[symp_nan, 'summary']:

            if (type(item) == str):
                extra_symptoms = [x.lower() for x in nltk.word_tokenize(item) if x.isalpha()]
                for word in extra_symptoms:
                    if word in top_sym:
                        symptoms.append(word)

        for i, item in enumerate(symptoms):
            for key in replace_symp.keys():
                if item in replace_symp[key]:
                    symptoms[i] = key

        return symptoms


    def extract_death(self,data):

        """
        This function will look into the <<summary>> column and attempt to extract information/dates
        when the infected DIED. Once it does so, it will record the date under the <<death>>
        column. Then, it will create a new column <<death_status>> where
        1=the infected person DIED
        0=status unknown or person recovered

        """

        self.data=data

        pattern = r'(death on|death|died|died on)\s*(\d+[\/\\]*\d+[\/\\]*\d+)'
        # death_date=[]

        for i, item in zip(self.data['summary'].index, self.data['summary']):
            if (type(item) == str) and re.search(pattern, item):
                death_date = re.findall(pattern, item)[0][1]
                data.loc[i, 'death'] = death_date

        for i, item in zip(self.data['death'].index, self.data['death']):
            if (item == '0') or (item == 0):
                self.data.loc[i, 'death_status'] = 0
            else:
                self.data.loc[i, 'death_status'] = 1
        return self.data


    def extract_recovered(self,data):

        """
        This function will look into the <<summary>> column and attempt to extract information/dates
        when the infected recovered. Once it does so, it will record the date under the <<recovered>>
        column. Then, it will create a new column <<recovered_status>> where
        1=the infected person was RECOVERED
        0=the infected person was not recovered or status unknown
        """
        self.data=data

        pattern = r'(recovered on|discharged on|discharged|recovered)\s*(\d+[\/\\]*\d+[\/\\]*\d+)'

        for i, item in zip(self.data['recovered'].index, self.data['recovered']):
            if item == '12/30/1899':
                self.data.loc[i, 'recovered'] = str(0)

        for i, item in zip(self.data['summary'].index, self.data['summary']):
            if (type(item) == str) and re.search(pattern, item):
                recovered_date = re.findall(pattern, item)[0][1]
                data.loc[i, 'recovered'] = recovered_date

        for i, item in zip(self.data['recovered'].index, self.data['recovered']):
            if (item == '0') or (item == np.nan) or (item == 0):
                self.data.loc[i, 'recovered_status'] = 0
            else:
                self.data.loc[i, 'recovered_status'] = 1
        return self.data


    def extract_symptoms(self,data):

        """
        This function will extract ALL symptoms, form the <<summary>> column as well as from the
        <<symptom>> column and it will create a new column in the data frame for each symptom
        where they will all be recorded with a 1 if present and 0 if not present
        """
        self.data=data

        for i, item in zip(self.data[self.data['symptom'].isnull() == False].index,
                           self.data[self.data['symptom'].isnull() == False]['symptom']):

            # print(item)
            symptoms = [x.lower().strip() for x in item.split(',')]

            replace_symp = {'chills': ['chill'],
                            'cough': ['coughing', 'mild cough'],
                            'fever': ['high fever', 'feaver', 'feve\\', 'mild fever'],
                            'sore throat': ['throat discomfort', 'throat pain'],
                            'difficulty breathing': ['difficulty in breathing', 'difficult in breathing',
                                                     'breathlessness'],
                            'muscle ache': ['aching muscles', 'muscle pain', 'muscle aches', 'sore body', 'myalgia',
                                            'myalgias', 'muscle cramps'],
                            'sputum': ['cough with sputum'],
                            'fatigue': ['tired'],
                            'nasal discharge': ['runy nose'],
                            'flu symptoms': ['flu']}

            for index, item in enumerate(symptoms):
                for key in replace_symp.keys():
                    if item in replace_symp[key]:
                        symptoms[index] = key

            for sym in symptoms:
                self.data.loc[i, sym] = int(1)

        symp_nan = self.data[self.data['symptom'].isnull() == True].index

        sym = self.symptom_fequency(self.data, 'symptom', 34)
        #     print(len(set(sym)))

        sym_lst = list(set(sym))
        #     print(sym_lst)

        for i, item in zip(symp_nan, self.data.loc[symp_nan, 'summary']):

            if (type(item) == str):
                extra_symptoms = [x.lower().strip() for x in item.split(' ') if x.isalpha()]
            #                 print(extra_symptoms)

            for sym in extra_symptoms:
                if sym in sym_lst:
                    if type(self.data.loc[i, 'symptom']) == str:

                        self.data.loc[i, 'symptom'] = self.data.loc[i, 'symptom'] + ',' + sym
                        self.data.loc[i, sym] = 1
                    else:

                        data.loc[i, 'symptom'] = sym
                        data.loc[i, sym] = 1

        pattern = r'no symptoms|no symptom'

        for i, item in zip(self.data['summary'].index, self.data['summary']):
            if (type(item) == str) and re.search(pattern, item):
                self.data.loc[i, 'symptom'] = 'no symptoms'
                self.data.loc[i, sym_lst] = 0

        self.data[self.data.columns[-32:]] = self.data[self.data.columns[-32:]].fillna(int(0))

        symp_nan_data = self.data[self.data['symptom'].isnull() == True].index

        self.data.loc[symp_nan_data, sym_lst] = np.nan

        return self.data

    def final_status(self,data):

        """
        This function will create a new column <<final_status>>
        1=the infected person DIED
        0=the infected person RECOVERED
        """
        self.data=data

        for i in self.data.index:

            if self.data.loc[i, 'death_status'] == float(1):
                self.data.loc[i, 'final_status'] = [1]
            elif self.data.loc[i, 'recovered_status'] == float(1):
                self.data.loc[i, 'final_status'] = [0]

            elif (self.data.loc[i, 'death_status'] == float(0)) and (self.data.loc[i, 'recovered_status'] == float(0)):
                self.data.loc[i, 'death_status'] = np.nan
                self.data.loc[i, 'recovered_status'] = np.nan
                self.data.loc[i, 'final_status'] = np.nan

        return self.data

    def unknown_symp_death_rec_status(self,data):


        """
        This function will drop all individuals that have an unknown status (death vs recovered)
        AND is unknown whether they displayed any symptoms or not

        """
        self.data=data

        sym_nan = self.data[self.data['symptom'].isnull() == True].index

        for i in sym_nan:
            # print(type(data.loc[i,'death_status']))
            if math.isnan(self.data.loc[i, 'death_status']) and math.isnan(self.data.loc[i, 'recovered_status']):
                self.data.drop(i, inplace=True)
        return self.data



    def symptom_to_death(self,data):

        from datetime import timedelta

        """
        This function will create a dataframe that contains only the info from the time
        an individual displayed symptoms to the time of death
        """

        self.data=data

        sym=self.symptom_fequency(self.data,'symptom',34)
        sym_freq = nltk.FreqDist(sym)
        most_common = sym_freq.most_common(5)
        symp_lst = [i[0] for i in most_common]

        symp_to_death = self.data[(self.data['symptom_onset'].isnull() == False) & (self.data['death'].isnull() == False)
                             & (self.data['death'] != '0') & (self.data['death'] != str(1))][
            ['symptom_onset', 'hosp_visit_date', 'death','age', 'gender','symptom']+symp_lst]

        #symp_to_death.loc[[263, 390, 676], 'death'] = ['2/13/2020', '2/25/2020', '2/15/2020']

        symp_to_death[['symptom_onset', 'hosp_visit_date', 'death']] = symp_to_death.iloc[:, :3].apply(pd.to_datetime,
                                                                                                       errors='coerce')

        new_cols = ['symptom_to_death', 'symptom_to_hosp', 'hosp_to_death']

        cols1 = ['death', 'hosp_visit_date', 'death']
        cols2 = ['symptom_onset', 'symptom_onset', 'hosp_visit_date']

        for col_new, c1, c2 in zip(new_cols, cols1, cols2):
            symp_to_death[col_new] = symp_to_death[c1] - symp_to_death[c2]

            symp_to_death[col_new] = pd.to_timedelta(symp_to_death[col_new], errors='coerce').dt.days

        return symp_to_death.dropna()


    def symptom_to_recovered(self,data):

        """
        This function will create a dataframe that contains only the info from the time
        an individual displayed symptoms to the time of death
        """

        self.data=data

        sym=self.symptom_fequency(self.data,'symptom',34)
        sym_freq = nltk.FreqDist(sym)
        most_common = sym_freq.most_common(5)
        symp_lst = [i[0] for i in most_common]

        symp_to_recovered = self.data[(self.data['symptom_onset'].isnull() == False) & (self.data['recovered'].isnull() == False)
                                 & (self.data['recovered'] != '0') & (self.data['recovered'] != str(1))][
            ['symptom_onset', 'hosp_visit_date', 'recovered', 'age','gender','symptom']+symp_lst]

        drop_rows = symp_to_recovered[symp_to_recovered['recovered'] == '12/30/1899'].index
        symp_to_recovered.drop(drop_rows, inplace=True)

        symp_to_recovered[['symptom_onset', 'hosp_visit_date', 'recovered']] = symp_to_recovered.iloc[:, :3].apply(
            pd.to_datetime, errors='coerce')

        new_cols = ['symptom_to_recovered', 'symptom_to_hosp', 'hosp_to_recovered']

        cols1 = ['recovered', 'hosp_visit_date', 'recovered']
        cols2 = ['symptom_onset', 'symptom_onset', 'hosp_visit_date']



        for col_new, c1, c2 in zip(new_cols, cols1, cols2):
            symp_to_recovered[col_new] = symp_to_recovered[c1] - symp_to_recovered[c2]

            symp_to_recovered[col_new] = pd.to_timedelta(symp_to_recovered[col_new], errors='coerce').dt.days

        return symp_to_recovered.dropna()

    def perc_symptoms(self,symptoms):
        self.symptoms=symptoms

        freq_sym = nltk.FreqDist(self.symptoms)
        perc_freq = {}
        for i in freq_sym.most_common():
            perc_freq[i[0]] = i[1]
        df_perc = pd.DataFrame(perc_freq, index=['count'])
        df = df_perc.T.copy()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'symptom'}, inplace=True)
        return df

    def group_perc_symptoms(self,freq_sym):
        self.freq_sym=freq_sym
        perc_freq={}
        for i in self.freq_sym.most_common():
            perc_freq[i[0]] = i[1]
        df_perc = pd.DataFrame(perc_freq, index=['count'])
        df = df_perc.T.copy()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'symptom'}, inplace=True)
        return df



    def create_master_frame(self,data):
        self.data=data

        self.data = self.extract_symptoms(self.data)

        self.data = self.extract_death(self.data)

        self.data = self.extract_recovered(self.data)

        self.data = self.final_status(self.data)

        # data=unknown_symp_death_rec_status(data)

        self.data.drop(['case_in_country', 'link', 'Unnamed: 3', 'source'], axis=1, inplace=True)

        for country in self.data['country'].unique():
            pop = self.data.groupby('country').count()['visiting Wuhan'][country]
            self.data.loc[self.data[self.data['country'] == country].index, 'pop'] = [pop]

        return self.data



# df=pd.read_csv('COVID19_data.csv')
#
# cov=Covid(df)
#
# df=cov.create_master_frame(df)
#
# print(df.head())








