import os
import numpy as np
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta
from covid19_DTM.models.utils import is_Belgian_primary_secundary_school_holiday

#########################
## Compliance function ##
#########################

def ramp_fun(Nc_old, Nc_new, t, t_start, l):
    """
    t : timestamp
        current date
    l : int
        number of additional days after the time delay until full compliance is reached
    """
    
    return np.array(Nc_old + ((Nc_new-Nc_old)/l)*((t-t_start)/timedelta(days=1)), np.float64)

###############################
## Mobility update functions ##
###############################

def load_all_mobility_data(agg, dtype='fractional', beyond_borders=False):
    """
    Function that fetches all available mobility data and adds it to a DataFrame with dates as indices and numpy matrices as values. Make sure to regularly update the mobility data with the notebook notebooks/preprocessing/Quick-update_mobility-matrices.ipynb to get the data for the most recent days. Also returns the average mobility over all available data, which might NOT always be desirable as a back-up mobility.

    Input
    -----
    agg : str
        Denotes the spatial aggregation at hand. Either 'prov', 'arr' or 'mun'
    dtype : str
        Choose the type of mobility data to return. Either 'fractional' (default), staytime (all available hours for region g spent in h), or visits (all unique visits from region g to h)
    beyond_borders : boolean
        If true, also include mobility abroad and mobility from foreigners

    Returns
    -------
    all_mobility_data : pd.DataFrame
        DataFrame with datetime objects as indices ('DATE') and np.arrays ('place') as value column
    average_mobility_data : np.array
        average mobility matrix over all available dates
    """

    ### Validate input ###

    if agg not in ['mun', 'arr', 'prov']:
        raise ValueError(
                    "spatial stratification '{0}' is not legitimate. Possible spatial "
                    "stratifications are 'mun', 'arr', or 'prov'".format(agg)
                )
    if dtype not in ['fractional', 'staytime', 'visits']:
        raise ValueError(
                    "data type '{0}' is not legitimate. Possible mobility matrix "
                    "data types are 'fractional', 'staytime', or 'visits'".format(dtype)
                )

    ### Load all available data ###

    # Define absolute location of this file
    abs_dir = os.path.dirname(__file__)
    # Define data location for this particular aggregation level
    data_location = f'../../../data/covid19_DTM/interim/mobility/{agg}/{dtype}'

    # Iterate over all available interim mobility data
    all_available_dates=[]
    all_available_places=[]
    directory=os.path.join(abs_dir, f'{data_location}')
    for csv in os.listdir(directory):
        # take YYYYMMDD information from processed CSVs. NOTE: this supposes a particular data name format!
        datum = csv[-12:-4]
        # Create list of datetime objects
        all_available_dates.append(pd.to_datetime(datum, format="%Y%m%d"))
        # Load the CSV as a np.array
        if beyond_borders:
            place = pd.read_csv(f'{directory}/{csv}', index_col='mllp_postalcode').values
        else:
            place = pd.read_csv(f'{directory}/{csv}', index_col='mllp_postalcode').drop(index='Foreigner', columns='ABROAD').values
            if dtype=='fractional':
                # make sure the rows sum up to 1 nicely again after dropping a row and a column
                place = place / place.sum(axis=1)
        # Create list of places
        all_available_places.append(place)
    # Create new empty dataframe with available dates. Load mobility later
    df = pd.DataFrame({'DATE' : all_available_dates, 'place' : all_available_places}).set_index('DATE')
    all_mobility_data = df.copy()

    # Take average of all available mobility data
    average_mobility_data = df['place'].values.mean()

    return all_mobility_data, average_mobility_data

class make_mobility_update_function():
    """
    Output the time-dependent mobility function with the data loaded in cache

    Input
    -----
    proximus_mobility_data : DataFrame
        Pandas DataFrame with dates as indices and matrices as values. Output of mobility.get_proximus_mobility_data.
    proximus_mobility_data_avg : np.array
        Average mobility matrix over all matrices
    """
    def __init__(self, proximus_mobility_data):
        self.proximus_mobility_data = proximus_mobility_data.sort_index()

    @lru_cache()
    def __call__(self, t):
        """
        time-dependent function which has a mobility matrix of type dtype for every date.
        Note: only works with datetime input (no integer time steps). This

        Input
        -----
        t : timestamp
            current date as datetime object

        Returns
        -------
        place : np.array
            square matrix with mobility of type dtype (fractional, staytime or visits), dimension depending on agg
        """

        t = pd.Timestamp(t.date())

        try: # simplest case: there is data available for this date (the key exists)
            place = self.proximus_mobility_data['place'][t]
        except:
            if t < pd.Timestamp(2020, 2, 10):
                # prepandemic default mobility. Take average of first 20-ish days
                if t.dayofweek < 5:
                    # business days
                    place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek < 5]['place'][:pd.Timestamp(2020, 3, 1)].mean()
                elif t.dayofweek >= 5:
                    # weekend
                    place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek >= 5]['place'][:pd.Timestamp(2020, 3, 1)].mean()
            elif t == pd.Timestamp(2020, 2, 21):
                # first missing date in Proximus data. Just take average
                place = (self.proximus_mobility_data['place'][pd.Timestamp(2020, 2, 20)] + 
                          self.proximus_mobility_data['place'][pd.Timestamp(2020, 2, 22)])/2
            elif t == pd.Timestamp(2020, 12, 18):
                # second missing date in Proximus data. Just take average
                place = (self.proximus_mobility_data['place'][pd.Timestamp(2020, 12, 17)] + 
                          self.proximus_mobility_data['place'][pd.Timestamp(2020, 12, 19)])/2
            elif t > pd.Timestamp(2021, 8, 31):
                # beyond Proximus service. Make a distinction between holiday/non-holiday and weekend/business day
                holiday = is_Belgian_primary_secundary_school_holiday(t)
                if holiday:
                    # it's a holiday. Take average of summer vacation behaviour
                    if t.dayofweek < 5:
                        # non-weekend holiday
                        place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek < 5]['place'][pd.Timestamp(2021, 7, 1):pd.Timestamp(2021, 8, 31)].mean()
                    elif t.dayofweek >= 5:
                        # weekend holiday
                        place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek >= 5]['place'][pd.Timestamp(2021, 7, 1):pd.Timestamp(2021, 8, 31)].mean()
                if not holiday:
                    # it's not a holiday. Take average of two months before summer vacation
                    if t.dayofweek < 5:
                        # business day
                        place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek < 5]['place'][pd.Timestamp(2021, 6, 1):pd.Timestamp(2021, 6, 30)].mean()
                    elif t.dayofweek >= 5:
                        # regular weekend
                        place = self.proximus_mobility_data[self.proximus_mobility_data.index.dayofweek >= 5]['place'][pd.Timestamp(2021, 6, 1):pd.Timestamp(2021, 6, 30)].mean()
                
        return place

    def mobility_wrapper_func(self, t, states, param):
        return self.__call__(t)

###################
## VOC functions ##
###################

class make_VOC_function():
    """
    Class that returns a time-dependant parameter function for COVID-19 SEIQRD model parameter f_VOC (variant fraction).
    Logistic parameters were manually fitted to the VOC effalence data and are hardcoded in the init function of this module
    """

    def __init__(self, VOC_logistic_growth_parameters):
        self.logistic_parameters=VOC_logistic_growth_parameters
    
    @staticmethod
    def logistic_growth(t, t_sig, k):
        return 1/(1+np.exp(-k*(t-t_sig)/timedelta(days=1)))

    def __call__(self, t, states, param):
        """

        Input
        -----

        t: datetime.datetime
            Current date in simulation

        Output
        ------

        f_VOC: np.ndarray
            Fraction of each VOC (sums to one)        
        """

        # Pre-allocate alpha
        alpha = np.zeros(len(self.logistic_parameters.index), np.float64)
        # Before introduction of first variant, return all zeros
        if t <= min(pd.to_datetime(self.logistic_parameters['t_introduction'].values)):
            return alpha
        else:
            # Retrieve correct index of variant that is currently "growing"
            try:
                idx = [index for index,value in enumerate(self.logistic_parameters['t_introduction'].values) if pd.Timestamp(value) >= t][0]-1
            except:
                idx = len(self.logistic_parameters['t_introduction'].values) - 1
            # Perform computation of currently growing VOC fraction and its derivative
            f = self.logistic_growth(t, self.logistic_parameters.iloc[idx]['t_sigmoid'], self.logistic_parameters.iloc[idx]['k'])
            # Decision logic
            if idx == 0:
                alpha[idx] = f
            else:
                alpha[idx-1] = 1-f
                alpha[idx] = f
            return alpha    

###########################
## Vaccination functions ##
###########################

class make_N_vacc_function():
    """A time-dependent parameter function to return the vaccine incidence at each timestep of the simulation
       Includes an example of a "hypothetical" extension of an ongoing booster campaign
    """
    
    def __init__(self,  df_incidences, agg=None, age_classes=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'),
                 hypothetical_function=False):
        
        #####################
        ## Input variables ##
        #####################
        
        # Window length of exponential moving average to smooth vaccine incidence
        filter_length = 31
        # Parameters of hypothetical booster campaign
        weekly_doses = 7*30000
        vacc_order = list(range(len(age_classes)))[::-1]
        stop_idx = len(age_classes)
        refusal = 0.1*np.ones(len(age_classes))
        extend_weeks = 26
        
        ############################################################
        ## Check if Sciensano changed the definition of the doses ##
        ############################################################
        
        df = df_incidences
        if not (df.index.get_level_values('dose').unique() == ['A', 'B', 'C', 'E']).all():
            # This code was built at a point in time when the following doses were in the raw dataset provided by Sciensano:
            # A: 0 --> P, B: P --> F, C: 0 --> F, E: F --> B
            # If you get this error it means Sciensano has changed the definition of these doses
            raise ValueError("The names of the 'doses' in the vaccination dataset have been changed from those the code was designed for ({0})".format(df.index.get_level_values('dose').unique().values))
        
        ####################################
        ## Step 1: Perform age conversion ##
        ####################################

        # Only perform age conversion if necessary
        if len(age_classes) != len(df.index.get_level_values('age').unique()):
            df = self.age_conversion(df, age_classes, agg)
        elif (age_classes != df.index.get_level_values('age').unique()).any():
            df = self.age_conversion(df, age_classes, agg)

        ###############################################
        ## Step 2: Hypothetical vaccination campaign ##
        ###############################################
        
        if hypothetical_function:
            # REMARK: It would be more elegant to have the user supply a "hypothetical function" together with
            # a dictionary of arguments to the __init__ function.
            # However, since I don't really need hypothetical functions at the moment (2022-05), I'm going to park this here without much further ado
            df = self.append_booster_campaign(df, extend_weeks, weekly_doses, vacc_order, stop_idx, refusal)
        
        #####################################################################
        ## Step 3: Convert weekly to daily incidences and smooth using EMA ##
        #####################################################################

        df = self.smooth_incidences(df, filter_length, agg)
        
        #########################
        ## Step 4: Save a copy ##
        ######################### 

        if agg:
            df.to_pickle(os.path.join(os.path.dirname(__file__), f"../../../data/covid19_DTM/interim/sciensano/vacc_incidence_{agg}.pkl"))
        else:
            df.to_pickle(os.path.join(os.path.dirname(__file__), f"../../../data/covid19_DTM/interim/sciensano/vacc_incidence_national.pkl"))

        ##################################################
        ## Step 5: Assign relevant (meta)data to object ##
        ##################################################        

        # Dataframe 
        self.df = df
        # Number of spatial patches
        try:
            self.G = len(self.df.index.get_level_values('NIS').unique().values)
        except:
            pass
        # Aggregation    
        self.agg = agg
        # Number of age groups
        self.N = len(age_classes)
        # Number of vaccine doses
        self.D = len(self.df.index.get_level_values('dose').unique().values)
        
        return None
        
    @lru_cache()
    def get_data(self,t):
        """
        Function to extract the vaccination incidence at date 't' from the pd.Dataframe of incidences and format it into the right size
        """


        if self.agg:
            try:
                return np.array(self.df.loc[pd.Timestamp(t.date()),:,:,:].values, np.float64).reshape( (self.G, self.N, self.D) )
            except:
                return np.zeros([self.G, self.N, self.D], np.float64)
        else:
            try:
                return np.array(self.df.loc[pd.Timestamp(t.date()),:,:].values, np.float64).reshape( (self.N, self.D) )
            except:
                return np.zeros([self.N, self.D], np.float64)
    
    def __call__(self, t, states, param):
        """
        Time-dependent parameter function compatible wrapper for cached function `get_data`
        Returns the vaccination incidence at time "t"
        
        Input
        -----
        
        t : datetime.datetime
            Current date in simulation
        
        Returns
        -------
        
        N_vacc : np.array
            Number of individuals to be vaccinated at simulation time "t" per [age, (space), dose]
        """

        return self.get_data(t)
    
    ######################
    ## Helper functions ##
    ######################
    
    @staticmethod
    def smooth_incidences(df, filter_length, agg):
        """
        A function to convert the vaccine incidences dataframe from weekly to daily data and smooth the incidences with an exponential moving average
        
        Inputs
        ------
        df: pd.Series
            Pandas series containing the weekly vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19_DTM.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
            Must contain 'date', 'NIS', 'age', 'dose' as indices for the spatial model
        
        filter_length: int
            Window length of the exponential moving average
            
        agg: str or None
            Spatial aggregation level. `None` for the national model, 'prov' for the provincial level, 'arr' for the arrondissement level
        
        Output
        ------
        
        df: pd.Series
            Pandas series containing smoothed daily vaccination incidences, indexed using the same pd.Multiindex as the input
        
        """
        
        # Start- and enddate
        df_start = pd.Timestamp(df.index.get_level_values('date').min())
        df_end = pd.Timestamp(df.index.get_level_values('date').max())
    
        # Make a dataframe with the desired format
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'date':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [pd.date_range(start=df_start, end=df_end, freq='D'),]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        df_new = pd.Series(index=index, name = df.name, dtype=float)

        # Loop over (NIS),age,dose
        for age in df.index.get_level_values('age').unique():
            for dose in df.index.get_level_values('dose').unique():
                if agg:
                    for NIS in df.index.get_level_values('NIS').unique():
                         # Convert weekly to daily data
                        daily_data = df.loc[slice(None), NIS, age, dose].resample('D').bfill().apply(lambda x : x/7)
                        # Apply an exponential moving average
                        daily_data_EMA = daily_data.ewm(span=filter_length, adjust=False).mean()
                        # Assign to new dataframe
                        df_new.loc[slice(None), NIS, age, dose] = daily_data_EMA.values
                else:
                    # Convert weekly to daily data
                    daily_data = df.loc[slice(None), age, dose].resample('D').bfill().apply(lambda x : x/7)
                    # Apply an exponential moving average
                    daily_data_EMA = daily_data.ewm(span=filter_length, adjust=False).mean()
                    # Assign to new dataframe
                    df_new.loc[slice(None), age, dose] = daily_data_EMA.values
                    
        return df_new
        
    @staticmethod
    def age_conversion(df, desired_age_classes, agg):
        """
        A function to convert a dataframe of vaccine incidences to another set of age groups `desired_age_classes` using demographic weighing
        
        Inputs
        ------
        
         df: pd.Series
            Pandas series containing the vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19_DTM.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
            Must contain 'date', 'NIS', 'age', 'dose' as indices for the spatial model  
            
        desired_age_classes: pd.IntervalIndex
            Age groups you want the vaccination incidence to be formatted in
            Example: pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
        
        agg: str or None
            Spatial aggregation level. `None` for the national model, 'prov' for the provincial level, 'arr' for the arrondissement level
           
        Returns
        -------
        
        df_new: pd.Series
            Same pd.Series containing the vaccination incidences, but uses the age groups in `desired_age_classes`
        
        """
        
        
        from covid19_DTM.data.utils import convert_age_stratified_quantity
        print('\nConverting stored vaccine incidence from {0} to {1} age groups, this may take some time\n'.format(len(df.index.get_level_values('age').unique()), len(desired_age_classes)))

        # Define a new dataframe with the desired age groups
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'age':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [desired_age_classes]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        df_new = pd.Series(index=index, name=df.name, dtype=float)

        # Loop to the dataseries level and perform demographic aggregation
        if agg:
            with tqdm(total=len(df.index.get_level_values('date').unique())*len(df.index.get_level_values('NIS').unique())) as pbar:
                for date in df.index.get_level_values('date').unique():
                    for NIS in df.index.get_level_values('NIS').unique():
                        for dose in df.index.get_level_values('dose').unique():
                            data = df.loc[date, NIS, slice(None), dose].reset_index().set_index('age')
                            df_new.loc[date, NIS, slice(None), dose] = convert_age_stratified_quantity(data, desired_age_classes, agg=agg, NIS=NIS).values
                        pbar.update(1)
        else:
            with tqdm(total=len(df.index.get_level_values('date').unique())*len(df.index.get_level_values('NIS').unique())) as pbar:
                for date in df.index.get_level_values('date').unique():
                    for dose in df.index.get_level_values('dose').unique():
                        data = df.loc[date, slice(None), dose].reset_index().set_index('age')
                        df_new.loc[date, slice(None), dose] = convert_age_stratified_quantity(data, desired_age_classes).values
                    pbar.update(1)

        return df_new
    
    @staticmethod
    def append_booster_campaign(df, extend_weeks, weekly_doses, vacc_order, stop_idx, refusal):
        """
        A function that appends `extend_weeks` of a hypothetical booster campaign to the dataframe containing the weekly vaccination incidences `df`
        Does not work for the spatial model!
        
        Inputs
        ------
        
        df: pd.Series
            Pandas series containing the weekly vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19_DTM.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
        
        extend_weeks: int
            Number of weeks the incidences dataframe should be extended
            
        weekly_doses: int/float
            Number of weekly doses to be administered
        
        vacc_order: list/np.array
            A list containing the age-specific order in which the vaccines should be administered
            Must be the same length as the number of age groups in the model
            f.e. [9 8 .. 1 0] means elderly are prioritized
            
        stop_idx: int
            Number of age groups the algorithm should loop over
            f.e. len(age_classes) means all age groups are vaccine eligible
            f.e. len(age_classes)-1 means the algorithm stops at the last index of vacc_order
        
        refusal: list/np.array
            Vaccine refusal rate in age group i 
            Must be the same length as the number of age groups in the model
        
        Outputs
        -------
        
        df: pd.Series
            Pandas series containing the weekly vaccination indices, extended with a hypothetical booster campaign
            Multiindex from the input series is retained
        """
        
        
        # Index of desired output dose
        dose_idx = 3 # Administering boosters
        
        # Start from the supplied data
        df_out = df  
        
        # Metadata
        N = len(df.index.get_level_values('age').unique())
        D = len(df.index.get_level_values('dose').unique())
        
        # Generate desired daterange
        df_end = pd.Timestamp(df.index.get_level_values('date').max())
        date_range = pd.date_range(start=df_end+pd.Timedelta(days=7), end=df_end+pd.Timedelta(days=7*extend_weeks), freq='W-MON')
        
        # Compute the number of fully vaccinated individuals in every age group
        cumulative_F=[]
        for age_class in df.index.get_level_values('age').unique():
            cumulative_F.append((df.loc[slice(None), age_class, 'B'] + df.loc[slice(None), age_class, 'C']).cumsum().iloc[-1])
        
        # Loop over dates
        weekly_doses0 = weekly_doses
        for date in date_range:
            # Reset the weekly doses
            weekly_doses = weekly_doses0
            # Compute the cumulative number of boosted individuals
            cumulative_B=[]
            for age_class in df_out.index.get_level_values('age').unique():
                cumulative_B.append(df_out.loc[slice(None), age_class, 'E'].cumsum().iloc[-1])
            # Initialize N_vacc
            N_vacc = np.zeros([N,D])
            # Booster vaccination loop
            idx = 0
            while weekly_doses > 0:
                if idx == stop_idx:
                    weekly_doses= 0
                else:
                    doses_needed = (1-refusal[vacc_order[idx]])*cumulative_F[vacc_order[idx]] - cumulative_B[vacc_order[idx]]
                    if doses_needed > weekly_doses:
                        N_vacc[vacc_order[idx],dose_idx] = weekly_doses
                        weekly_doses = 0
                    elif doses_needed >= 0:
                        N_vacc[vacc_order[idx],dose_idx] = doses_needed
                        weekly_doses -= doses_needed
                    else:
                        N_vacc[vacc_order[idx],dose_idx] = 0
                    idx+=1
                    
            # Generate series on date with same multiindex as output
            iterables=[]
            for index_name in df.index.names:
                if index_name != 'date':
                    iterables += [df.index.get_level_values(index_name).unique()]
                else:
                    iterables += [[date],]
            index = pd.MultiIndex.from_product(iterables, names=df.index.names)
            df_new = pd.Series(index=index, name=df.name, dtype=float)
            # Put the result in
            df_new.loc[date, slice(None), slice(None)] = N_vacc.flatten()
            # Concatenate
            df_out = pd.concat([df_out, df_new])
        
        return df_out

from tqdm import tqdm
class make_vaccination_efficacy_function():
    
    ################################################
    ## Generate or load dataframe with efficacies ##
    ################################################
    
    def __init__(self, update=False, agg=None, age_classes=pd.IntervalIndex.from_tuples([(0,12),(12,18),(18,25),(25,35),(35,45),(45,55),(55,65),(65,75),(75,85),(85,120)], closed='left'),
                    df_incidences=None, vaccine_params=None, VOCs=['WT', 'abc', 'delta']):
    
        if update==False:
            # Simply load data
            from covid19_DTM.data.utils import to_pd_interval
            if agg:
                path = os.path.join(os.path.dirname(__file__), f"../../../data/covid19_DTM/interim/sciensano/vacc_rescaling_values_{agg}.csv")
                df_efficacies = pd.read_csv(path,  index_col=['date','NIS','age', 'dose', 'VOC'], converters={'date': pd.to_datetime, 'age': to_pd_interval})
            else:
                path = os.path.join(os.path.dirname(__file__), "../../../data/covid19_DTM/interim/sciensano/vacc_rescaling_values_national.csv")
                df_efficacies = pd.read_csv(path,  index_col=['date','age', 'dose', 'VOC'], converters={'date': pd.to_datetime, 'age': to_pd_interval})
        else:
            # Warn user this may take some time
            print('\nThe vaccination rescaling parameters must be updated because a change was made to the vaccination parameters, or the dataframe with incidences was changed/updated. This may take some time.\n')
            # Downsample the incidences dataframe to weekly frequency, if weekly incidences are provided this does nothing. Not tested if the data frequency is higher than one week.
            df_incidences = self.convert_frequency_WMON(df_incidences)
            # Add the one-shot J&J and the second doses together
            df_incidences = self.sum_oneshot_second(df_incidences)
            # Construct dataframe with cumulative sums
            df_cumsum = self.incidences_to_cumulative(df_incidences)
            # Format vaccination parameters 
            vaccine_params, onset, waning = self.format_vaccine_params(vaccine_params)
            # Compute weighted efficacies
            df_efficacies = self.compute_weighted_efficacy(df_incidences, df_cumsum, vaccine_params, waning)
            # Add dummy efficacy for zero doses
            df_efficacies = self.add_efficacy_no_vaccine(df_efficacies)
            # Save result
            dir = os.path.join(os.path.dirname(__file__), "../../../data/covid19_DTM/interim/sciensano/")
            if agg:
                df_efficacies.to_csv(os.path.join(dir, f'vacc_rescaling_values_{agg}.csv'))
                df_efficacies.to_pickle(os.path.join(dir, f'vacc_rescaling_values_{agg}.pkl'))
            else:
                df_efficacies.to_csv(os.path.join(dir, f'vacc_rescaling_values_national.csv'))
                df_efficacies.to_pickle(os.path.join(dir, f'vacc_rescaling_values_national.pkl'))
            
        # Throw out unwanted VOCs
        df_efficacies = df_efficacies.iloc[df_efficacies.index.get_level_values('VOC').isin(VOCs)]
        
        # Check if an age conversion is necessary
        age_conv=False
        if len(age_classes) != len(df_efficacies.index.get_level_values('age').unique()):
            df_efficacies = self.age_conversion(df_efficacies, age_classes, agg)
        elif (age_classes != df_efficacies.index.get_level_values('age').unique()).any():
            df_efficacies = self.age_conversion(df_efficacies, age_classes, agg)

        # Assign efficacy dataset
        self.df_efficacies = df_efficacies

        # Compute some other relevant attributes
        self.available_dates = df_efficacies.index.get_level_values('date').unique()
        self.n_VOCs = len(df_efficacies.index.get_level_values('VOC').unique())
        self.n_doses = len(df_efficacies.index.get_level_values('dose').unique())
        self.N = len(age_classes)
        self.agg = agg
        try:
            self.G = len(df_efficacies.index.get_level_values('NIS').unique())
        except:
            self.G = 0
        
        return None
    
    ################################################
    ## Convert data to a format model understands ##
    ################################################
    
    @lru_cache()
    def __call__(self, t, efficacy):
        """
        Returns vaccine efficacy matrix of size [N, n_doses, n_VOCs] or [G, N, n_doses, n_VOCs] for the requested time t.

        Input
        -----
        t : datetime.datetime
            Time at which you want to know the rescaling parameter matrix
        rescaling_type : str
            Either 'e_s', 'e_i' or 'e_h'
            
        Output
        ------
        E : np.array
            Vaccine efficacy matrix at time t.
        """

        if efficacy not in self.df_efficacies.columns:
            raise ValueError(
                "valid vaccine efficacies are 'e_s', 'e_i', or 'e_h'.")

        if t < min(self.available_dates):
            # Take unity matrix
            if self.agg:
                E = np.zeros([self.G, self.N, self.n_doses, self.n_VOCs], np.float64)
            else:
                E = np.zeros([self.N, self.n_doses, self.n_VOCs, ], np.float64)

        elif t <= max(self.available_dates):
            # Take interpolation between dates for which data is available
            t_data_first = pd.Timestamp(self.available_dates[np.argmax(self.available_dates >=t)-1])
            t_data_second = pd.Timestamp(self.available_dates[np.argmax(self.available_dates >=t)])
            if self.agg:
                E_values_first = self.df_efficacies.loc[t_data_first, :, :, :,:][efficacy].to_numpy()
                E_first = np.reshape(E_values_first, (self.G,self.N,self.n_doses,self.n_VOCs))
                E_values_second = self.df_efficacies.loc[t_data_second, :, :, :, :][efficacy].to_numpy()
                E_second = np.reshape(E_values_second, (self.G,self.N,self.n_doses,self.n_VOCs))
            else:
                E_values_first = self.df_efficacies.loc[t_data_first, :, :, :][efficacy].to_numpy()
                E_first = np.reshape(E_values_first, (self.N,self.n_doses,self.n_VOCs))
                E_values_second = self.df_efficacies.loc[t_data_second, :, :, :][efficacy].to_numpy()
                E_second = np.reshape(E_values_second, (self.N,self.n_doses,self.n_VOCs))

            # linear interpolation
            E = E_first + (E_second - E_first) * (t - t_data_first).total_seconds() / (t_data_second - t_data_first).total_seconds()
            
        elif t > max(self.available_dates):
            # Take last available data point
            t_data = pd.Timestamp(max(self.available_dates))
            if self.agg:
                E_values = self.df_efficacies.loc[t_data, :, :, :, :][efficacy].to_numpy()
                E = np.reshape(E_values, (self.G,self.N,self.n_doses,self.n_VOCs))
            else:
                E_values = self.df_efficacies.loc[t_data, :, :, :][efficacy].to_numpy()
                E = np.reshape(E_values, (self.N,self.n_doses,self.n_VOCs))

        return np.array((1-E), np.float64)
    
    def e_s(self, t, states, param):
        return self.__call__(t, 'e_s')
    
    def e_i(self, t, states, param):
        return self.__call__(t, 'e_i')
    
    def e_h(self, t, states, param):
        return self.__call__(t, 'e_h')

    ######################
    ## Helper functions ##
    ######################
    
    @staticmethod
    def convert_frequency_WMON(df):
        """Converts the frequency of the incidences dataframe to 'W-MON'. This is done to avoid using excessive computational resources."""
        groupers=[]
        for index_name in df.index.names:
            if index_name != 'date':
                groupers += [pd.Grouper(level=index_name)]
            else:
                groupers += [pd.Grouper(level='date', freq='W-MON')]
        df = df.groupby(groupers).sum()
        return df

    def compute_weighted_efficacy(self, df, df_cumsum, vaccine_params, waning):
        """ Computes the average protection of the vaccines for every dose and every VOC, based on the vaccination incidence in every age group (and spatial patch) and subject to exponential vaccine waning
        """ 

        VOCs = vaccine_params.index.get_level_values('VOC').unique()
        efficacies=vaccine_params.index.get_level_values('efficacy').unique()
    
        # Define output dataframe
        iterables=[]
        for index_name in df.index.names:
            iterables += [df.index.get_level_values(index_name).unique()]
        iterables.append(VOCs)
        names=list(df.index.names)
        names.append('VOC')
        index = pd.MultiIndex.from_product(iterables, names=names)
        df_efficacies = pd.DataFrame(index=index, columns=efficacies, dtype=float)
        
        # Computational loop
        dates=df.index.get_level_values('date').unique()
        age_classes=df.index.get_level_values('age').unique()
        doses=df.index.get_level_values('dose').unique() 
        
        if not 'NIS' in df.index.names:
            with tqdm(total=len(age_classes)*len(doses)) as pbar:
                for age_class in age_classes:
                    for dose in doses:
                        for date in dates:
                            cumsum = df_cumsum.loc[date, age_class, dose]
                            delta_t=[]
                            f=[]
                            w=pd.DataFrame(index=pd.MultiIndex.from_product([dates[dates<=date], VOCs]), columns=efficacies, dtype=float)
                            for inner_date in dates[dates<=date]:
                                # Compute fraction versus delta_t
                                delta_t.append((date-inner_date)/pd.Timedelta(days=1))
                                if cumsum != 0.0:
                                    f.append(df.loc[inner_date, age_class, dose]/cumsum)
                                else:
                                    if (date-inner_date)/pd.Timedelta(days=1) == 0:
                                        f.append(1.0)
                                    else:
                                        f.append(0)
                                # Compute weight
                                for VOC in VOCs:
                                    for efficacy in efficacies:
                                        waning_days = waning[dose]
                                        E_best = vaccine_params.loc[(VOC, dose, efficacy), 'best']
                                        E_waned = vaccine_params.loc[(VOC, dose, efficacy), 'waned']
                                        weight = self.exponential_waning((date-inner_date)/pd.Timedelta(days=1), waning_days, E_best, E_waned)
                                        w.loc[(inner_date, VOC), efficacy] = weight
                            for VOC in VOCs:
                                for efficacy in efficacies:
                                    # Compute efficacy
                                    df_efficacies.loc[(date, age_class, dose, VOC),efficacy] = np.dot(f,w.loc[(slice(None), VOC), efficacy])
                        pbar.update(1)
        else:
            NISs=df.index.get_level_values('NIS').unique()
            with tqdm(total=len(NISs)*len(age_classes)*len(doses)) as pbar:
                for age_class in age_classes:
                    for NIS in NISs:
                        for dose in doses:
                            for date in dates:
                                cumsum = df_cumsum.loc[date, NIS, age_class, dose]
                                delta_t=[]
                                f=[]
                                w=pd.DataFrame(index=pd.MultiIndex.from_product([dates[dates<=date], VOCs]), columns=efficacies, dtype=float)
                                for inner_date in dates[dates<=date]:
                                    # Compute fraction versus delta_t
                                    delta_t.append((date-inner_date)/pd.Timedelta(days=1))
                                    if cumsum != 0.0:
                                        f.append(df.loc[inner_date, NIS, age_class, dose]/cumsum)
                                    else:
                                        if (date-inner_date)/pd.Timedelta(days=1) == 0:
                                            f.append(1.0)
                                        else:
                                            f.append(0)
                                    # Compute weight
                                    for VOC in VOCs:
                                        for efficacy in efficacies:
                                            waning_days = waning[dose]
                                            E_best = vaccine_params.loc[(VOC, dose, efficacy), 'best']
                                            E_waned = vaccine_params.loc[(VOC, dose, efficacy), 'waned']
                                            weight = self.exponential_waning((date-inner_date)/pd.Timedelta(days=1), waning_days, E_best, E_waned)
                                            w.loc[(inner_date, VOC), efficacy] = weight
                                for VOC in VOCs:
                                    for efficacy in efficacies:
                                        # Compute efficacy
                                        df_efficacies.loc[(date, NIS, age_class, dose, VOC),efficacy] = np.dot(f,w.loc[(slice(None), VOC), efficacy])
                            pbar.update(1)
                                
        return df_efficacies

    @staticmethod
    def add_efficacy_no_vaccine(df):
        """Adds a dummy vaccine dose 'none' to the efficacy dataframe with zero efficacy
        """

        # Define a new dataframe with the desired age groups
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'dose':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [['none',] + list(df.index.get_level_values(index_name).unique())]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        new_df = pd.DataFrame(index=index, columns=df.columns, dtype=float)
        # Perform a join operation, remove the left columns and fillna with zero
        merged_df = new_df.join(df, how='left', lsuffix='_left', rsuffix='')
        merged_df = merged_df.drop(columns=['e_s_left', 'e_i_left', 'e_h_left']).fillna(0)

        return merged_df

    def age_conversion(self, df, desired_age_classes, agg):
        """
        A function to convert a dataframe of vaccine efficacies to another set of age groups `desired_age_classes` using demographic weighing
        
        Inputs
        ------
        
         df: pd.Series
            Pandas series containing the vaccination incidences, indexed using a pd.Multiindex
            Obtained using the function `covid19_DTM.data.sciensano.get_public_spatial_vaccination_data()`
            Must contain 'date', 'age' and 'dose' as indices for the national model
            Must contain 'date', 'NIS', 'age', 'dose' as indices for the spatial model  
            
        desired_age_classes: pd.IntervalIndex
            Age groups you want the vaccination incidence to be formatted in
            Example: pd.IntervalIndex.from_tuples([(0,20),(20,60),(60,120)], closed='left')
        
        agg: str or None
            Spatial aggregation level. `None` for the national model, 'prov' for the provincial level, 'arr' for the arrondissement level
           
        Returns
        -------
        
        df_new: pd.DataFrame
            Same pd.DataFrame containing the vaccination incidences, but uses the age groups in `desired_age_classes`
        
        """

        from covid19_DTM.data.utils import convert_age_stratified_property

        print('\nConverting stored vaccine efficacies from {0} to {1} age groups, this may take some time\n'.format(len(df.index.get_level_values('age').unique()), len(desired_age_classes)))

        # Define a new dataframe with the desired age groups
        iterables=[]
        for index_name in df.index.names:
            if index_name != 'age':
                iterables += [df.index.get_level_values(index_name).unique()]
            else:
                iterables += [desired_age_classes]
        index = pd.MultiIndex.from_product(iterables, names=df.index.names)
        new_df = pd.DataFrame(index=index, columns=df.columns, dtype=float)

        # Loop to the dataseries level and perform demographic weighing
        if agg:
            with tqdm(total=len(df.index.get_level_values('date').unique())*len(df.index.get_level_values('NIS').unique())) as pbar:
                for date in df.index.get_level_values('date').unique():
                    for NIS in df.index.get_level_values('NIS').unique():
                        for dose in df.index.get_level_values('dose').unique():
                            for VOC in df.index.get_level_values('VOC').unique():
                                new_df.loc[date, NIS, slice(None), dose, VOC] = convert_age_stratified_property(df.loc[date, NIS, slice(None), dose, VOC], desired_age_classes, agg=agg, NIS=NIS).values
                        pbar.update(1)
        else:
            with tqdm(total=len(df.index.get_level_values('date').unique())) as pbar:
                for date in df.index.get_level_values('date').unique():
                    for dose in df.index.get_level_values('dose').unique():
                        for VOC in df.index.get_level_values('VOC').unique():
                            new_df.loc[(date, slice(None), dose, VOC),:] = convert_age_stratified_property(df.loc[date, slice(None), dose, VOC], desired_age_classes).values
                    pbar.update(1)

        return new_df

    @staticmethod
    def exponential_waning(delta_t, waning_days, E_best, E_waned):
        """
        Function to compute the vaccine efficacy after delta_t days, accouting for exponential waning of the vaccine's immunity.

        Input
        -----
        days : float
            number of days after the novel vaccination
        waning_days : float
            number of days for the vaccine to wane (on average)
        E_best : float
            rescaling value related to the best possible protection by the currently injected vaccine
        E_waned : float
            rescaling value related to the vaccine protection after a waning period.

        Output
        ------
        E_eff : float
            effective rescaling value associated with the newly administered vaccine

        """

        if E_best == E_waned:
            return E_best
        # Exponential waning
        A0 = E_best
        k = -np.log((E_waned)/(E_best))/(waning_days)
        E_eff = A0*np.exp(-k*(delta_t))

        return E_eff
    
    @staticmethod
    def incidences_to_cumulative(df):
        """Computes the cumulative number of vaccines based using the incidences dataframe
        """
        levels = list(df.index.names)
        levels.remove("date")
        df_cumsum = df.groupby(by=levels).cumsum()
        df_cumsum = df_cumsum.reset_index().set_index(list(df.index.names)).squeeze().rename('CUMULATIVE')
        return df_cumsum
    
    @staticmethod
    def sum_oneshot_second(df):
        """ Sums vaccine dose 'B' and 'C' in the Sciensano dataset, corresponding to getting a second dose or getting the one-shot J&J vaccine
        """
        # Add dose 'B' and 'C' together
        df[df.index.get_level_values('dose') == 'B'] = df[df.index.get_level_values('dose') == 'B'].values + df[df.index.get_level_values('dose') == 'C'].values
        # Remove dose 'C'
        df = df.reset_index()
        df = df[df['dose'] != 'C']
        # Reformat dataframe
        df = df.set_index(df.columns[df.columns!='INCIDENCE'].to_list())
        df = df.squeeze()
        # Use more declaritive names for doses
        df.rename(index={'A':'partial', 'B':'full', 'E':'boosted'}, inplace=True)
        
        return df
    
    @staticmethod
    def format_vaccine_params(df):
        """ This function format the vaccine parameters provided by the user in function `covid19_DTM.data.model_parameters.get_COVID19_SEIQRD_VOC_parameters` into a format better suited for the computation in this module."""

        ###################
        ## e_s, e_i, e_h ##
        ###################

        # Define vaccination properties  
        iterables = [df.index.get_level_values('VOC').unique(),['none', 'partial', 'full', 'boosted'], ['e_s', 'e_i', 'e_h']]
        index = pd.MultiIndex.from_product(iterables, names=['VOC', 'dose', 'efficacy'])
        df_new = pd.DataFrame(index=index, columns=['initial', 'best', 'waned'])    

        # No vaccin = zero efficacy
        df_new.loc[(slice(None), 'none', slice(None)),:] = 0

        # Partially vaccinated is initially non-vaccinated + Waning of partially vaccinated to non-vaccinated
        df_new.loc[(slice(None), 'partial', slice(None)),['initial','waned']] = 0

        # Fill in best efficacies using data, waning of full, booster using data
        for VOC in df_new.index.get_level_values('VOC'):
            for dose in df_new.index.get_level_values('dose'):
                for efficacy in df_new.index.get_level_values('efficacy'):
                    df_new.loc[(VOC, dose, efficacy),'best'] = df.loc[(VOC,dose),efficacy]
                    if ((dose == 'full') | (dose =='boosted')):
                        df_new.loc[(VOC, dose, efficacy),'waned'] = df.loc[(VOC,'waned'),efficacy]

        # partial, waned = (1/2) * partial, best (equals assumption that half-life is equal to waning days)
        df_new.loc[(slice(None), 'partial', slice(None)),'waned'] = 0.5*df_new.loc[(slice(None), 'partial', slice(None)),'best'].values

        # full, initial = partial, best
        df_new.loc[(slice(None), 'full', slice(None)),'initial'] = df_new.loc[(slice(None), 'partial', slice(None)),'best'].values

        # boosted, initial = full, waned
        df_new.loc[(slice(None), 'boosted', slice(None)),'initial'] = df_new.loc[(slice(None), 'full', slice(None)),'waned'].values

        ############################
        ## onset_immunity, waning ##
        ############################

        onset={}
        waning={}
        for dose in df_new.index.get_level_values('dose'):
            onset.update({dose: df.loc[(df_new.index.get_level_values('VOC')[0], dose), 'onset_immunity']})
            waning.update({dose: df.loc[(df_new.index.get_level_values('VOC')[0], dose), 'waning']})

        return df_new, onset, waning

###################################
## Google social policy function ##
###################################

class make_contact_matrix_function():
    """
    Class that returns contact matrix based on 4 effention parameters by default, but has other policies defined as well.

    Input
    -----
    df_google : dataframe
        google mobility data

    Nc_all : dictionnary
        contact matrices for home, school, work, transport, leisure and others

    Output
    ------

    __class__ : default function
        Default output function, based on contact_matrix_4eff

    """
    def __init__(self, df_google, Nc_all, G=None):
        self.df_google = df_google.astype(float)
        self.Nc_all = Nc_all
        self.G = G
        # Determine if a distinction between weekdays and weekenddays is made
        if (('weekday' in list(self.Nc_all.keys())) and ('weekendday' in list(self.Nc_all.keys()))):
            self.distinguish_day_type = True
        else:
            self.distinguish_day_type = False
        # Compute start and endtimes of dataframe Google
        self.df_google_start = df_google.index.get_level_values('date')[0]
        self.df_google_end = df_google.index.get_level_values('date')[-1]
        # Make a memory of hospital load
        window_length = 28
        self.I = list(70*np.ones(window_length))
        # Check if provincial GCMR data is provided
        self.provincial = None
        if 'NIS' in self.df_google.index.names:
            self.provincial = True
            if self.G != len(self.df_google.index.get_level_values('NIS').unique().values):
                raise ValueError("Using provincial-level Google Community Mobility data only works with spatial aggregation set to 'prov'")

    @lru_cache() # once the function is run for a set of parameters, it doesn't need to compile again
    def __call__(self, t, eff_home=1, eff_schools=1, eff_work=1, eff_rest = 1, mentality=1,
                       school_primary_secundary=None, school_tertiary=None, work=None, transport=None, leisure=None, others=None, home=None):

        """
        t : datetime.datetime
            current simulation date
        eff_... : float in [0,1]
            effectivity parameter
        mentality : float in [0,1]

        school, work, transport, leisure, others : float [0,1]
            level of opening of these sectors
            if None is provided:
                work, transport, leisure, others are derived from the google community mobility data
                school opening is derived from `is_Belgian_primary_secundary_school_holiday`
        """

        # Check if day is weekendday or weekday
        if self.distinguish_day_type:
            if ((t.weekday() == 5) | (t.weekday() == 6)):
                Nc_all = self.Nc_all['weekendday']
            else:
                Nc_all = self.Nc_all['weekday']
        else:
            Nc_all = self.Nc_all

        # Check if schools open/closed are provided and if not figure it out
        if school_primary_secundary is None:
            school_primary_secundary = not is_Belgian_primary_secundary_school_holiday(t)
        
        if school_tertiary is None:
            if ((datetime(year=t.year, month=5, day=14) <= t <= datetime(year=t.year, month=7, day=1)) | \
                 (datetime(year=t.year, month=12, day=14) <= t <= datetime(year=t.year, month=12, day=31)) | \
                   (datetime(year=t.year, month=1, day=1) <= t <= datetime(year=t.year, month=2, day=14)) | \
                    (datetime(year=t.year, month=9, day=1) <= t <= datetime(year=t.year, month=10, day=1))):
                school_tertiary = False
            else:
                school_tertiary = not is_Belgian_primary_secundary_school_holiday(t)

        places_var = [work, transport, leisure, others]
        places_names = ['work', 'transport', 'leisure', 'others']
        GCMR_names = ['work', 'transport', 'retail_recreation', 'grocery']

        if self.provincial:
            if t < self.df_google_start:
                row = -self.df_google.loc[slice(self.df_google.index.get_level_values('date')[0],self.df_google.index.get_level_values('date')[7]), slice(None)].groupby(by='NIS').mean()/100
            elif self.df_google_start <= t <= self.df_google_end:
                # Extract row at timestep t
                row = -self.df_google.loc[(pd.Timestamp(t.date()), slice(None)),:]/100
            else:
                # Extract last 7 days and take the mean
                row = -self.df_google.loc[(self.df_google_end - pd.Timedelta(days=7)): self.df_google_end, slice(None)].groupby(level='NIS').mean()/100

            # Sort NIS codes from low to high
            row.sort_index(level='NIS', ascending=True, inplace=True)
            # Extract values
            values_dict={}
            for idx,place in enumerate(places_var):
                if place is None:
                    place = 1 - row[GCMR_names[idx]].values
                else:
                    try:
                        test=len(place)
                    except:
                        place = place*np.ones(self.G)     
                values_dict.update({places_names[idx]: place})

            # Primary/Secundary schools
            try:
                test=len(school_primary_secundary)
            except:
                school_primary_secundary = school_primary_secundary*np.ones(self.G)

            # Tertiary
            try:
                test=len(school_tertiary)
            except:
                school_tertiary = school_tertiary*np.ones(self.G)

            # Expand dims on mentality
            if isinstance(mentality,tuple):
                mentality=np.array(mentality)
                mentality = mentality[:, np.newaxis, np.newaxis]

            # Construct contact matrix
            CM = (mentality*(eff_home*np.ones(self.G)[:, np.newaxis,np.newaxis]*Nc_all['home'] +
                    (eff_schools*school_primary_secundary)[:, np.newaxis,np.newaxis]*Nc_all['school_primary_secundary'] +
                    (eff_schools*school_tertiary)[:, np.newaxis,np.newaxis]*Nc_all['school_tertiary'] +
                    (eff_work*values_dict['work'])[:,np.newaxis,np.newaxis]*Nc_all['work'] + 
                    (eff_rest*values_dict['transport'])[:,np.newaxis,np.newaxis]*Nc_all['transport'] + 
                    (eff_rest*values_dict['leisure'])[:,np.newaxis,np.newaxis]*Nc_all['leisure'] +
                    (eff_rest*values_dict['others'])[:,np.newaxis,np.newaxis]*Nc_all['others']) )

        else:
            if t < self.df_google_start:
                row = -self.df_google[0:7].mean()/100
            elif self.df_google_start <= t <= self.df_google_end:
                # Extract row at timestep t
                row = -self.df_google.loc[pd.Timestamp(t.date())]/100
            else:
                # Extract last 7 days and take the mean
                row = -self.df_google[-7:-1].mean()/100
            
            # Extract values
            values_dict={}
            for idx,place in enumerate(places_var):
                if place is None:
                    place = 1 - row[GCMR_names[idx]]
                values_dict.update({places_names[idx]: place})  

            # Construct contact matrix
            CM = (eff_home*Nc_all['home'] + 
                    eff_schools*school_primary_secundary*Nc_all['school_primary_secundary'] +
                    eff_schools*school_tertiary*Nc_all['school_tertiary'] +
                    eff_work*values_dict['work']*Nc_all['work'] +
                    eff_rest*values_dict['transport']*Nc_all['transport'] +
                    eff_rest*values_dict['leisure']*Nc_all['leisure'] +
                    eff_rest*values_dict['others']*Nc_all['others'] )

            # Expand to the correct spatial size
            if self.G:
                # Expand contact matrix and mentality to size of spatial stratification
                CM = np.ones(self.G)[:,np.newaxis,np.newaxis]*CM
                mentality = mentality*np.ones(self.G)
                # Multiply CM with mentality
                CM = mentality[:,np.newaxis,np.newaxis]*CM
            else:
                CM = mentality*CM

        return np.array(CM, np.float64)

    def behavioral_model(self, I_new, T, k):
        """
        Computes the contact reduction due to behavioral changes out of fear for infection

        Input
        =====

        I_new: float
            The most recent datapoint

        T: float
            Total population

        Output
        ======

        behavioral_mentality_change: float
            Reduction (bounded between zero and one) 
        """

        # Update the moving window of infected
        self.I.append(I_new)
        self.I = self.I[1:]
        # Take the mean over the moving window
        # Except the final two datapoints --> data collection takes time irl!
        I = np.mean(self.I[:-2])

        return 1-(1-I/T)**k

    ####################
    ## National model ##
    ####################

    def policies_all(self, t, states, param, l1, l2, eff_schools, eff_work, eff_rest, eff_home, mentality, k):
        '''
        Function that returns the time-dependant social contact matrix Nc for all COVID waves.
        
        Input
        -----
        t : datetime.datetime
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l1 : float
            Compliance parameter for social policies during first lockdown 2020 COVID-19 wave
        l2 : float
            Compliance parameter for social policies during second lockdown 2020 COVID-19 wave        
        eff_{location} : float
            "Effectivity" of contacts at {location}.
        k: float
            Parameter of the behavioral change model, linking number of infections to reduction in contacts.
            https://www.sciencedirect.com/science/article/pii/S1755436518301063 
        mentality: float
            Forced government intervention

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''

        # Behavioral change model
        mentality_behavioral = self.behavioral_model(np.sum(states['H_in']), 11e6, k)

        # Assumption: all effectivities equal to eff_work
        eff_schools=eff_work
        eff_rest=eff_work

        # Convert compliance l to dates
        l1_days = timedelta(days=l1)
        l2_days = timedelta(days=l2)
        # Define key dates of first wave
        t1 = datetime(2020, 3, 16) # Start of lockdown
        t2 = datetime(2020, 5, 15) # Start of lockdown relaxation
        t3 = t2 + timedelta(days=2.5*28) # End of lockdown relaxation
        t4 = datetime(2020, 8, 1) # Start of summer lockdown in Antwerp
        t5 = datetime(2020, 9, 1) # End of summer lockdown in Antwerp
        # Define key dates of second wave/winter 2020-2021
        t6 = datetime(2020, 10, 19) # lockdown (1) --> early schools closure
        t7 = datetime(2021, 2, 15) # Contact increase in children
        t8 = datetime(2021, 3, 26) # Start of Easter holiday
        t9 = datetime(2021, 5, 15) # Start of lockdown relaxation
        t10 = t9 + timedelta(days=2.5*28) # End of relaxation
        # Define key dates of winter 2021-2022
        t11 = datetime(2021, 11, 17) # Overlegcommite 1 out of 3
        t12 = datetime(2021, 12, 3) # Overlegcommite 3 out of 3

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home=eff_home, eff_schools=eff_schools, eff_work=eff_work, eff_rest=eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
        elif t2 < t <= t3:
            l = (t3 - t2)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t2, l)      
        elif t3 < t <= t4:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0) 
        elif t4 < t <= t5:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)                                          
        elif t5 < t <= t6:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6  < t <= t6 + l2_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t6, l2)
        elif t6 + l2_days < t <= t7:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        elif t7 < t <= t8:
            return 1.10*self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        elif t8 < t <= t9:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        elif t9 < t <= t10:
            l = (t10 - t9)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality = 1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t9, l)
        elif t10 < t <= t11:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)        

        ######################
        ## Winter 2021-2022 ##
        ######################

        elif t11 < t <= t12:
            l = (t12 - t11)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t11, l)
        elif t > t12:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

    ###################
    ## Spatial model ##
    ###################

    def get_mentality_summer_2020_lockdown(self, summer_rescaling_F, summer_rescaling_W, summer_rescaling_B):

        if self.G == 11:
            idx_Hainaut = [6,]
            idx_F = [0, 1, 4, 5, 8]
            idx_Bxl = [3,]
            idx_W = [2, 6, 7, 9, 10]
            # Coefficients based on relative second wave peak height (as compared to NIS: 20001) 
            mentality_summer_2020_lockdown = np.array([1, 1.12,         # F
                                                    1.12,               # W
                                                    1.12,               # Bxl
                                                    1, 1.13,            # F: W-Fla: 1.45
                                                    2.15, 1.93,         # W
                                                    0.9,                  # F: Lim: 1.43
                                                    1.16, 1.30])        # W
            # Rescale Flanders and Wallonia/Bxl seperately based on two parameters
            mentality_summer_2020_lockdown[idx_F] *= summer_rescaling_F
            mentality_summer_2020_lockdown[idx_Bxl] *= summer_rescaling_B
            mentality_summer_2020_lockdown[idx_W] *= summer_rescaling_W

        elif self.G == 43:
            idx_Hainaut = [21, 22, 23, 24, 25, 26, 27,]
            idx_F = [0, 1, 2,                          # Antwerpen
                    4, 5,                              # Vlaams-Brabant
                    7, 8, 9, 10, 11, 12, 13, 14,       # West-Vlaanderen
                    15, 16, 17, 18, 19, 20,            # Oost-Vlaanderen
                    32, 33, 34]                        # Limburg      
            idx_Bxl = [3,]                             # Brussel                    
            idx_W = [6,                                # Waals-Brabant
                    21, 22, 23, 24, 25, 26, 27,        # Henegouwen
                    28, 29, 30, 31,                    # Luik
                    35, 36, 37, 38, 39,                # Luxemburg
                    40, 41, 42]                        # Namen

            # Coefficients based on relative second wave peak height (as compared to NIS: 20001) 
            mentality_summer_2020_lockdown = np.ones(43)
            mentality_summer_2020_lockdown[0:3] = 1      # Antwerpen
            mentality_summer_2020_lockdown[4:6] = 1.12      # Vlaams-Brabant
            mentality_summer_2020_lockdown[6] = 1.12       # Waals-Brabant
            mentality_summer_2020_lockdown[3] = 1.12     # Brussel
            mentality_summer_2020_lockdown[7:15] = 1    # West-Vlaanderen
            mentality_summer_2020_lockdown[15:21] = 1.13   # Oost-Vlaanderen
            mentality_summer_2020_lockdown[21:28] = 2.15   # Henegouwen
            mentality_summer_2020_lockdown[28:32] = 1.93   # Luik
            mentality_summer_2020_lockdown[32:35] = 0.9   # Limburg
            mentality_summer_2020_lockdown[35:40] = 1.16  # Luxemburg
            mentality_summer_2020_lockdown[40:43] = 1.30   # Namen
            
            # Rescale Flanders and Wallonia/Bxl seperately based on two parameters
            mentality_summer_2020_lockdown[idx_F] *= summer_rescaling_F
            mentality_summer_2020_lockdown[idx_Bxl] *= summer_rescaling_B
            mentality_summer_2020_lockdown[idx_W] *= summer_rescaling_W

        return mentality_summer_2020_lockdown, idx_Hainaut

    def policies_no_lockdown(self, t, states, param, nc):
        mat = self.__call__(datetime(2020, 1, 1), eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, school=1)
        return nc[:, np.newaxis, np.newaxis]*mat
    
    def policies_no_lockdown_home(self, t, states, param, nc):
        mat = self.__call__(datetime(2020, 1, 1), eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, school=0) 
        return nc[:, np.newaxis, np.newaxis]*mat

    def policies_all_spatial(self, t, states, param, l1, l2, eff_schools, eff_work, eff_rest, eff_home, mentality, k, summer_rescaling_F, summer_rescaling_W, summer_rescaling_B):
        '''
        Function that returns the time-dependant social contact matrix Nc for all COVID waves.
        
        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l1 : float
            Compliance parameter for social policies during first lockdown 2020 COVID-19 wave
        l2 : float
            Compliance parameter for social policies during second lockdown 2020 COVID-19 wave        
        eff_{location} : float
            "Effectivity" of contacts at {location}. Alternatively, degree correlation between Google mobility indicator and SARS-CoV-2 spread at {location}.
        mentality : float
            Lockdown mentality multipier

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''

        # Assumption eff_schools = eff_work
        eff_schools=eff_work
        eff_rest=eff_work

        # Behavioral change model
        mentality_behavioral = self.behavioral_model(np.sum(states['H_in']), 11e6, k)

        # Protection against dipping below zero
        if mentality_behavioral > mentality:
            mentality_behavioral = mentality
        
        # Convert compliance l to dates
        l1_days = timedelta(days=l1)
        l2_days = timedelta(days=l2)
        # Define key dates of first wave
        t1 = datetime(2020, 3, 16) # Start of lockdown
        t2 = datetime(2020, 5, 15) # Start of lockdown relaxation
        t3 = t2 + timedelta(days=2.5*28) # End of lockdown relaxation
        t4 = datetime(2020, 8, 1) # Start of summer lockdown in Antwerp
        t5 = datetime(2020, 9, 1) # End of summer lockdown in Antwerp
        # Define key dates of second wave/winter 2020-2021
        t6 = datetime(2020, 10, 19) # lockdown (1) --> early schools closure
        t7 = datetime(2021, 2, 15) # Contact increase in children
        t8 = datetime(2021, 3, 26) # Start of Easter holiday
        t9 = datetime(2021, 5, 15) # Start of lockdown relaxation
        t10 = t9 + timedelta(days=2.5*28) # End of relaxation
        # Define key dates of winter 2021-2022
        t11 = datetime(2021, 11, 17) # Overlegcommite 1 out of 3
        t12 = datetime(2021, 12, 3) # Overlegcommite 3 out of 3
        # Get summer of 2020 parameters
        mentality_summer_2020_lockdown, idx_Hainaut = self.get_mentality_summer_2020_lockdown(summer_rescaling_F, summer_rescaling_W, summer_rescaling_B)    

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home=eff_home, eff_schools=eff_schools, eff_work=eff_work, eff_rest=eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
        elif t2 < t <= t3:
            l = (t3 - t2)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t2, l)      
        elif t3 < t <= t4:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0) 
        elif t4 < t <= t5:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_summer_2020_lockdown), school_primary_secundary=0, school_tertiary=0)      
        elif t5 < t <= t6:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6  < t <= t6 + l2_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t6, l2)
        elif t6 + l2_days < t <= t7:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        elif t7 < t <= t8:
            mat = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            mat[idx_Hainaut,:,:] *= 1.30
            return mat 
        elif t8 < t <= t9:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        elif t9 < t <= t10:
            l = (t10 - t9)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality = 1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t9, l)
        elif t10 < t <= t11:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)        

        ######################      
        ## Winter 2021-2022 ##
        ######################        

        elif t11 < t <= t12:
            l = (t12 - t11)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t11, l)
        elif t > t12:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

    def policies_all_home_only(self, t, states, param, l1, l2, eff_home, mentality, k, summer_rescaling_F, summer_rescaling_W, summer_rescaling_B):
            '''
            Function that returns the time-dependant social contact matrix of home contacts (Nc_home). 
            
            Input
            -----
            t : Timestamp
                simulation time
            states : xarray
                model states
            param : dict
                model parameter dictionary
            mentality : float
                Lockdown mentality multipier

            Returns
            -------
            CM : np.array
                Effective contact matrix (output of __call__ function)
            '''

            # Behavioral change model
            mentality_behavioral = self.behavioral_model(np.sum(states['H_in']), 11e6, k)

            # Protection against dipping below zero
            if mentality_behavioral > mentality:
                mentality_behavioral = mentality
   
            # Convert compliance l to dates
            l1_days = timedelta(days=l1)
            l2_days = timedelta(days=l2)
            # Define key dates of first wave
            t1 = datetime(2020, 3, 16) # Start of lockdown
            t2 = datetime(2020, 5, 15) # Start of lockdown relaxation
            t3 = t2 + timedelta(days=2.5*28) # End of lockdown relaxation
            t4 = datetime(2020, 8, 1) # Start of summer lockdown in Antwerp
            t5 = datetime(2020, 9, 1) # End of summer lockdown in Antwerp
            # Define key dates of second wave/winter 2020-2021
            t6 = datetime(2020, 10, 19) # lockdown (1) --> early schools closure
            t7 = datetime(2021, 2, 15) # Contact increase in children
            t8 = datetime(2021, 3, 26) # Start of Easter holiday
            t9 = datetime(2021, 5, 15) # Start of lockdown relaxation
            t10 = t9 + timedelta(days=2.5*28) # End of relaxation
            # Define key dates of winter 2021-2022
            t11 = datetime(2021, 11, 17) # Overlegcommite 1 out of 3
            t12 = datetime(2021, 12, 3) # Overlegcommite 3 out of 3
            # Get summer of 2020 parameters
            mentality_summer_2020_lockdown, idx_Hainaut = self.get_mentality_summer_2020_lockdown(summer_rescaling_F, summer_rescaling_W, summer_rescaling_B)    

            ################
            ## First wave ##
            ################

            if t <= t1:
                return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
            elif t1 < t <= t1 + l1_days:
                policy_old = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, t1, l1)
            elif t1 + l1_days < t <= t2:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            elif t2 < t <= t3:
                l = (t3 - t2)/timedelta(days=1)
                policy_old = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0,  mentality=1-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, t2, l)      
            elif t3 < t <= t4:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral) 
            elif t4 < t <= t5:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=tuple(mentality_summer_2020_lockdown))      
            elif t5 < t <= t6:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)

            ######################      
            ## Winter 2020-2021 ##
            ######################

            elif t6  < t <= t6 + l2_days:
                policy_old = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, t6, l2)
            elif t6 + l2_days < t <= t7:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            elif t7 < t <= t8:
                mat = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                mat[idx_Hainaut,:,:] *= 1.30
                return mat
            elif t8 < t <= t9:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            elif t9 < t <= t10:
                l = (t10 - t9)/timedelta(days=1)
                policy_old = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality = 1-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, t9, l)
            elif t10 < t <= t11:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)        
            
            ######################      
            ## Winter 2021-2022 ##
            ######################        

            elif t11 < t <= t12:
                l = (t12 - t11)/timedelta(days=1)
                policy_old = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, t11, l)
            elif t > t12:
                return self.__call__(t, eff_home=eff_home, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)

    def policies_all_manuscript(self, t, states, param, l1, l2, eff_schools, eff_work, eff_rest, eff_home, mentality, k, start_relaxation):
        '''
        Function that returns the time-dependant social contact matrix Nc for all COVID waves.

        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l1 : float
            Compliance parameter for social policies during first lockdown 2020 COVID-19 wave
        l2 : float
            Compliance parameter for social policies during second lockdown 2020 COVID-19 wave        
        eff_{location} : float
            "Effectivity" of contacts at {location}. Alternatively, degree correlation between Google mobility indicator and SARS-CoV-2 spread at {location}.
        k: float
            Parameter of the behavioral change model, linking number of infections to reduction in contacts.
            https://www.sciencedirect.com/science/article/pii/S1755436518301063 
        mentality: float

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''

        # Behavioral change model
        mentality_behavioral = self.behavioral_model(np.sum(states['H_in']), 11e6, k)

        # Assumption eff_schools = eff_work
        eff_schools=eff_work
        eff_rest=eff_work

        # Convert compliance l to dates
        l1_days = timedelta(days=l1)
        l2_days = timedelta(days=l2)
        # Define key dates of first wave
        t1 = datetime(2020, 3, 16) # Start of lockdown
        t2 = datetime(2020, 5, 15) # Start of lockdown relaxation
        t3 = t2 + timedelta(days=2.5*28) # End of lockdown relaxation
        t4 = datetime(2020, 8, 1) # Start of summer lockdown in Antwerp
        t5 = datetime(2020, 9, 1) # End of summer lockdown in Antwerp
        # Define key dates of second wave/winter 2020-2021
        t6 = datetime(2020, 10, 19) # lockdown (1) --> early schools closure
        t7 = datetime(2021, 2, 15) # Contact increase in children
        t8 = datetime(2021, 3, 26) # Start of Easter holiday
        start = datetime.strptime(start_relaxation, '%Y-&m-%d')
        l = 62

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home=eff_home, eff_schools=eff_schools, eff_work=eff_work, eff_rest=eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
        elif t2 < t <= t3:
            l = (t3 - t2)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t2, l)      
        elif t3 < t <= t4:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0) 
        elif t4 < t <= t5:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)                                          
        elif t5 < t <= t6:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6  < t <= t6 + l2_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t6, l2)
        elif t6 + l2_days < t <= t7:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        
        if start_relaxation == '2021-03-01':
            if t7 < t <= start:
                return 1.10*self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            elif start < t <= start+timedelta(days=l):
                policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                return ramp_fun(policy_old, policy_new, t, start, l)  
            else:
                return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        else:
            if t7 < t <= t8:
                return 1.10*self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            elif t8 < t <= start:
                return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            elif start < t <= start+timedelta(days=l):
                policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                return ramp_fun(policy_old, policy_new, t, start, l)  
            else:
                return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

    def policies_all_spatial_manuscript(self, t, states, param, l1, l2, eff_schools, eff_work, eff_rest, eff_home, mentality, k, summer_rescaling_F, summer_rescaling_W, summer_rescaling_B, start_relaxation):
        '''
        Function that returns the time-dependant social contact matrix Nc for all COVID waves.

        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        l1 : float
            Compliance parameter for social policies during first lockdown 2020 COVID-19 wave
        l2 : float
            Compliance parameter for social policies during second lockdown 2020 COVID-19 wave        
        eff_{location} : float
            "Effectivity" of contacts at {location}. Alternatively, degree correlation between Google mobility indicator and SARS-CoV-2 spread at {location}.
        mentality : float
            Lockdown mentality multipier

        Returns
        -------
        CM : np.array (9x9)
            Effective contact matrix (output of __call__ function)
        '''

        # Behavioral change model
        mentality_behavioral = self.behavioral_model(np.sum(states['H_in']), 11e6, k)

        # Assumption eff_schools = eff_work
        eff_schools=eff_work
        eff_rest=eff_work

        # Convert compliance l to dates
        l1_days = timedelta(days=l1)
        l2_days = timedelta(days=l2)
        # Define key dates of first wave
        t1 = datetime(2020, 3, 16) # Start of lockdown
        t2 = datetime(2020, 5, 15) # Start of lockdown relaxation
        t3 = t2 + timedelta(days=2.5*28) # End of lockdown relaxation
        t4 = datetime(2020, 8, 1) # Start of summer lockdown in Antwerp
        t5 = datetime(2020, 9, 1) # End of summer lockdown in Antwerp
        # Define key dates of second wave/winter 2020-2021
        t6 = datetime(2020, 10, 19) # lockdown (1) --> early schools closure
        t7 = datetime(2021, 2, 15) # Contact increase in children
        t8 = datetime(2021, 3, 26) # Start of Easter holiday
        start = datetime.strptime(start_relaxation, '%Y-&m-%d')
        l = 62
        # Get summer of 2020 parameters
        mentality_summer_2020_lockdown, idx_Hainaut = self.get_mentality_summer_2020_lockdown(summer_rescaling_F, summer_rescaling_W, summer_rescaling_B)    

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=1, eff_work=1, eff_rest=1, mentality=1-mentality_behavioral, school_primary_secundary=1, school_tertiary=1)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home=eff_home, eff_schools=eff_schools, eff_work=eff_work, eff_rest=eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
        elif t2 < t <= t3:
            l = (t3 - t2)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0)
            return ramp_fun(policy_old, policy_new, t, t2, l)      
        elif t3 < t <= t4:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=0, school_tertiary=0) 
        elif t4 < t <= t5:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=tuple(mentality_summer_2020_lockdown), school_primary_secundary=None, school_tertiary=None) 
        elif t5 < t <= t6:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6  < t <= t6 + l2_days:
            policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            return ramp_fun(policy_old, policy_new, t, t6, l2)
        elif t6 + l2_days < t <= t7:
            return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        
        if start_relaxation == '2021-03-01':
            if t7 < t <= start:
                mat = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                mat[idx_Hainaut,:,:] *= 1.30
                return mat 
            elif start < t <= start+timedelta(days=l):
                policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                return ramp_fun(policy_old, policy_new, t, start, l)  
            else:
                return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
        else:
            if t7 < t <= t8:
                mat = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                mat[idx_Hainaut,:,:] *= 1.30
                return mat 
            elif t8 < t <= start:
                return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
            elif start < t <= start+timedelta(days=l):
                policy_old = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=mentality-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                policy_new = self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)
                return ramp_fun(policy_old, policy_new, t, start, l)  
            else:
                return self.__call__(t, eff_home, eff_schools, eff_work, eff_rest, mentality=1-mentality_behavioral, school_primary_secundary=None, school_tertiary=None)

    def policies_all_home_only_manuscript(self, t, states, param, l1, l2, mentality, k, summer_rescaling_F, summer_rescaling_W, summer_rescaling_B, start_relaxation):
        '''
        Function that returns the time-dependant social contact matrix of home contacts (Nc_home). 

        Input
        -----
        t : Timestamp
            simulation time
        states : xarray
            model states
        param : dict
            model parameter dictionary
        mentality : float
            Lockdown mentality multipier

        Returns
        -------
        CM : np.array
            Effective contact matrix (output of __call__ function)
        '''

        # Behavioral change model
        mentality_behavioral = self.behavioral_model(np.sum(states['H_in']), 11e6, k)

        # Convert compliance l to dates
        l1_days = timedelta(days=l1)
        l2_days = timedelta(days=l2)
        # Define key dates of first wave
        t1 = datetime(2020, 3, 16) # Start of lockdown
        t2 = datetime(2020, 5, 15) # Start of lockdown relaxation
        t3 = t2 + timedelta(days=2.5*28) # End of lockdown relaxation
        t4 = datetime(2020, 8, 1) # Start of summer lockdown in Antwerp
        t5 = datetime(2020, 9, 1) # End of summer lockdown in Antwerp
        # Define key dates of second wave/winter 2020-2021
        t6 = datetime(2020, 10, 19) # lockdown (1) --> early schools closure
        t7 = datetime(2021, 2, 15) # Contact increase in children
        t8 = datetime(2021, 3, 26) # Start of Easter holiday
        start = datetime.strptime(start_relaxation, '%Y-&m-%d')
        l = 62
        # Get summer of 2020 parameters
        mentality_summer_2020_lockdown, idx_Hainaut = self.get_mentality_summer_2020_lockdown(summer_rescaling_F, summer_rescaling_W, summer_rescaling_B)    

        ################
        ## First wave ##
        ################

        if t <= t1:
            return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
        elif t1 < t <= t1 + l1_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
            policy_new = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            return ramp_fun(policy_old, policy_new, t, t1, l1)
        elif t1 + l1_days < t <= t2:
            return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
        elif t2 < t <= t3:
            l = (t3 - t2)/timedelta(days=1)
            policy_old = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            policy_new = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
            return ramp_fun(policy_old, policy_new, t, t2, l)      
        elif t3 < t <= t4:
            return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral) 
        elif t4 < t <= t5:
            return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=tuple(mentality_summer_2020_lockdown)) 
        elif t5 < t <= t6:
            return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)

        ######################      
        ## Winter 2020-2021 ##
        ######################

        elif t6  < t <= t6 + l2_days:
            policy_old = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
            policy_new = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            return ramp_fun(policy_old, policy_new, t, t6, l2)
        elif t6 + l2_days < t <= t7:
            return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
        if start_relaxation == '2021-03-01':
            if t7 < t <= start:
                mat = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                mat[idx_Hainaut,:,:] *= 1.30
                return mat 
            elif start < t <= start+timedelta(days=l):
                policy_old = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, start, l)  
            else:
                return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
        else:
            if t7 < t <= t8:
                mat = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                mat[idx_Hainaut,:,:] *= 1.30
                return mat 
            elif t8 < t <= start:
                return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
            elif start < t <= start+timedelta(days=l):
                policy_old = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=mentality-mentality_behavioral)
                policy_new = self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)
                return ramp_fun(policy_old, policy_new, t, start, l)  
            else:
                return self.__call__(t, eff_home=1, eff_schools=0, eff_work=0, eff_rest=0, mentality=1-mentality_behavioral)

#########################
## Hospital propensity ##
#########################

def h_func(t, states, param, f_h):
    """
    There is a clear difference in hospitalisation propensity between the first COVID-19 wave and the second COVID-19 wave
    The first COVID-19 wave having a significantly higher hospitalisation propensity than the second COVID-19 wave
    This effect is likely caused by the ill preparation during the first COVID-19 wave (nursing homes, no PPE for hospital staff etc.)
    """

    if t <= datetime(2020, 5, 1):
        return param
    elif datetime(2020, 5, 1) < t <=  datetime(2020, 9, 1):
        return ramp_fun(param, f_h*param, t, datetime(2020, 5, 1),(datetime(2020, 9, 1)-datetime(2020, 5, 1))/timedelta(days=1))
    else:
        return f_h*param

##########################
## Seasonality function ##
##########################

class make_seasonality_function():
    """
    Simple class to create functions that controls the season-dependent value of the transmission coefficients. Currently not based on any data, but e.g. weather patterns could be imported if needed.
    """
    def __call__(self, t, states, param, amplitude, peak_shift):
        """
        Default output function. Returns a sinusoid with average value 1.
        
        t : datetime.datetime
            simulation time
        amplitude : float
            maximum deviation of output with respect to the average (1)
        peak_shift : float
            phase. Number of days after January 1st after which the maximum value of the seasonality rescaling is reached 
        """
        ref_date = datetime(2021,1,1)
        # If peak_shift = 0, the max is on the first of January
        maxdate = ref_date + timedelta(days=peak_shift)
        # One period is one year long (seasonality)
        t = (t - maxdate)/timedelta(days=1)/365
        rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
        return rescaling