import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from ..visualization.data import google_mobility

def get_apple_mobility_data(update=True):
    """Download Apple Mobility Trends

    This function downloads, formats and returns the available Apple Mobility Trends for the Provinces of Belgium.
    A copy of the downloaded dataset is automatically saved in the /data/covid19_DTM/raw folder.

    Parameters
    -----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read only previously saved data

    Returns
    -------
    data : pandas.DataFrame
        DataFrame containing the Apple mobility data on daily basis, starting February 15th 2020.
        The dataframe has two indices: the spatial unit (f.e. 'Belgium' or 'Namur Province') and the mobility type ('walking', 'transit', 'driving').
        Example indexing: df_apple.loc[('Belgium','transit')]

    Notes
    -----
    Transport types (walking, transit, driving) are not available for every province.    
    Data before February 15th 2020 was omitted to match the start date of the Google community reports.
    Baseline was defined by Apple as the mobility on January 13th 2020. Data was rescaled using the mean mobility from February 15th 2020 until March 10th, 2020.
    Project homepage: https://covid19.apple.com/mobility
    Data extracted from: https://covid19-static.cdn-apple.com/covid19-mobility-data/2024HotfixDev12/v3/en-us/applemobilitytrends-2021-01-10.csv

    Example use
    -----------
    df_apple = mobility.get_apple_mobility_data(update=False)
    # Extracting transit mobility data for Belgium
    df_apple.loc[('Belgium','transit')]
    # Visualizing mobility data
    plt.plot(df_apple.loc[('Belgium','transit')])
    """

    # Data source
    url = 'https://covid19-static.cdn-apple.com/covid19-mobility-data/2024HotfixDev12/v3/en-us/applemobilitytrends-2021-01-10.csv'
    abs_dir = os.path.dirname(__file__)

    if update:
        # download raw data
        df_raw = pd.read_csv(url)
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/covid19_DTM/raw/mobility/apple/apple_mobility_trends.csv')
        df_raw.to_csv(rel_dir, index=False)
    else:
        df_raw = pd.read_csv(os.path.join(abs_dir,
            '../../../data/covid19_DTM/raw/mobility/apple/apple_mobility_trends.csv'))

    # Start by extracting the overall data for Belgium
    df = df_raw[df_raw['region']=='Belgium']
    columns = pd.to_datetime(df_raw.columns[6:].values)
    data = df.values[:,6:]
    arrays = [
        np.array(["Belgium","Belgium","Belgium"]),
        np.array(["driving", "transit", "walking"]),
    ]
    df_apple = pd.DataFrame(data,index=arrays,columns=columns)

    # Loop over all available provinces and transit types
    df = df_raw[((df_raw['country']=='Belgium')&(df_raw['geo_type']=='sub-region'))]
    for province in set(df['region']):
        for mobility_type in df['transportation_type'][df['region'] == province]:
            data = df[((df['transportation_type']==mobility_type)&(df['region']==province))].values[:,6:]
            df_entry = pd.DataFrame(data,index=[[province],[mobility_type]],columns=columns)
            df_apple=df_apple.append(df_entry)

    # Re-define baseline and cut away data before February 15th 2020:
    for spatial_unit,mobility_type in df_apple.index:
        base=np.mean(df_apple.loc[(spatial_unit,mobility_type)][pd.to_datetime('2020-02-15'):pd.to_datetime('2020-03-10')])
        df_apple.loc[(spatial_unit,mobility_type)] = -(100-df_apple.loc[(spatial_unit,mobility_type)]/base*100)
        df_apple.loc[(spatial_unit,mobility_type)] = df_apple.loc[(spatial_unit,mobility_type)][pd.to_datetime('2020-02-15'):]

    return df_apple

def get_google_mobility_data(update=True, provincial=False, plot=False, filename_plot=None):
    """Download Google Community mobility report data

    This function downloads, formats and returns the available Belgian Google Community mobility report data.
    A copy of the downloaded dataset is automatically saved in the /data/covid19_DTM/raw folder.

    Parameters
    -----------
    update : boolean (default True)
        True if you want to update the data,
        False if you want to read only previously saved data
    provincial : boolean (default False)
        True to return the google mobility data per province (10 provinces + Brussels)
    plot : boolean (default False)
        If True, return a preformatted plot of the data
    filename_viz: string
        filename and extension to automatically save the generated visualisation of the data
        The argument has no effect when plot is False

    Returns
    -----------
    data : pandas.DataFrame
        DataFrame with the google mobility data on daily basis. The following columns are returned:

        - retail_recreation : Mobility trends for places such as restaurants, cafés, shopping centres, theme parks, museums, libraries and cinemas.
        - grocery : Mobility trends for places such as grocery shops, food warehouses, farmers markets, specialty food shops and pharmacies.
        - parks: Mobility trends for places such as local parks, national parks, public beaches, marinas, dog parks, plazas and public gardens.
        - transport: Mobility trends for places that are public transport hubs, such as underground, bus and train stations.
        - work: Mobility trends for places of work.
        - residential: Mobility trends for places of residence.
    
        If provincial data is requested the resulting dataframe has a pandas multiindex containing two levels: 1) date and 2) region.
        To access multi-indexed data, use the 'loc' method: df.loc[(index_val_1, index_val_2), column_name].
        F.i. df[('2020-03-15', 'Brussels'), :] returns the mobility reductions on March 15th, 2020 in Brussels.
        To access the entries corresponding to an entire index level (f.i. all dates), use the slice method.
        F.i. df[(slice(None), 'Brussels'),:] returns the mobility reductions in Brussels on all possible dates

    Notes
    ----------
    Mobility data can be extracted as a report for any country from: https://www.google.com/covid19/mobility/
    Dataset was downloaded from: 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    Documentation by Google on data collection can be found here : https://www.google.com/covid19/mobility/data_documentation.html?hl=nl

    Example use
    -----------
    data = get_google_mobility_data(filename='community_report.svg')
    >>> # download google mobility data and store new version (no viz)
    >>> data = get_google_mobility_data()
    >>> # load google mobility data from raw data directory (no new download)
    >>> data = get_google_mobility_data(update=False)
    >>> # load google mobility data from raw data directory and create viz
    >>> data = get_google_mobility_data(update=False, plot=True)
    >>> # load google mobility data from raw data directory, create viz and save viz
    >>> data = get_google_mobility_data(update=False, plot=True, filename_plot="my_viz.png")
    """

    # Data source
    url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
    abs_dir = os.path.dirname(__file__)
    dtypes = {'sub_region_1': str, 'sub_region_2': str}

    # Update data if necessary
    if update:
        # download raw data
        df = pd.read_csv(url, parse_dates=['date'], dtype=dtypes)
        # save a copy in the raw folder
        rel_dir = os.path.join(abs_dir, '../../../data/covid19_DTM/raw/mobility/google/google_community_mobility_data_BE.csv')
        # Extract only Belgian data: full file is over 300 Mb
        df=df[df['country_region']=='Belgium']
        # Save data
        df.to_csv(rel_dir, index=False)

    else:
        df = pd.read_csv(os.path.join(abs_dir,
            '../../../data/covid19_DTM/raw/mobility/google/google_community_mobility_data_BE.csv'),
            parse_dates=['date'], dtype=dtypes)

    # Map column names
    variable_mapping = {
    'retail_and_recreation_percent_change_from_baseline': 'retail_recreation',
    'grocery_and_pharmacy_percent_change_from_baseline': 'grocery',
    'parks_percent_change_from_baseline': 'parks',
    'transit_stations_percent_change_from_baseline': 'transport',
    'workplaces_percent_change_from_baseline': 'work',
    'residential_percent_change_from_baseline': 'residential'
    }
    df = df.rename(columns=variable_mapping)

    if provincial == True:
        # Hardcode a dictionary to convert region name to NIS
        regions = ['Brussels', 'Antwerp', 'East Flanders', 'Flemish Brabant', 'Limburg', 'West Flanders', 'Hainaut', 'Liege', 'Luxembourg', 'Province of Namur', 'Walloon Brabant']
        corresponding_NIS = [21000, 10000, 40000, 20001, 70000, 30000, 50000, 60000, 80000, 90000, 20002]
        region_to_NIS_dict = dict(zip(regions, corresponding_NIS))        
        # Extract region names and append Brussels
        regions = list(df['sub_region_2'].dropna().unique())
        regions.insert(0,'Brussels')
        # Build multiindex dataframe
        iterables = [df['date'].unique(), corresponding_NIS]
        index = pd.MultiIndex.from_product(iterables, names=["date", "NIS"])
        new_df = pd.DataFrame(index=index, columns=list(variable_mapping.values()))
        # Loop over regions
        for region in regions:
            NIS = region_to_NIS_dict[region]
            if region == 'Brussels':
                new_df.loc[(slice(None),NIS),:] = df.loc[(df['sub_region_1'] == 'Brussels')][list(variable_mapping.values())].values
            else:
                new_df.loc[(slice(None),NIS),:] = df[df['sub_region_2']==region][list(variable_mapping.values())].values
            # Replace NaN with the last observed non-null value forward
            new_df.loc[(slice(None),NIS),:] = new_df.loc[(slice(None),NIS),:].fillna(method='ffill')
        df = new_df
    else:
            
        # Extract national values
        df=df[df['sub_region_1'].isnull().values]
        df = df.set_index("date")
        df.index.freq = 'D'
        df = df[list(variable_mapping.values())]

    if filename_plot and not plot:
        print("Filename plot has no effect, plot is not activated. Set `plot=True` to create plot.")

    if plot:
        if provincial:
            print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Plotting of provincial-level google community mobility data is not implemented (yet).")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

        fig, ax = google_mobility(df)
        
        if filename_plot:
            plt.savefig(filename_plot, dpi=600, bbox_inches='tight',
                        orientation='portrait', papertype='a4')
        else:
            plt.show()

    return df

####################################
# Proximus mobility data functions #
####################################

# Load all data in a big DataFrame
def get_proximus_mobility_data(agg):
    """
    Function that fetches all available mobility data and adds it to a DataFrame with dates as indices and numpy matrices as values. The loaded data are provided as fractions of the staytime between regions within Belgium.
    
    Make sure to regularly update the mobility data with the notebook notebooks/preprocessing/Quick-update_mobility-matrices.ipynb to get the data for the most recent days. Also returns the average mobility over all available data, which might NOT always be desirable as a back-up mobility.
    
    Input
    -----
    agg : str
        Denotes the spatial aggregation at hand. Either 'prov', 'arr' or 'mun'. NOTE currently only 'prov' data is available.
    
    Returns
    -------
    proximus_mobility_data : pd.DataFrame
        DataFrame with datetime objects as indices ('DATE') and np.arrays ('place') as value column
    """

    ### Validate input ###
    
    if agg not in ['mun', 'arr', 'prov']:
        raise ValueError(
                    "spatial stratification '{0}' is not legitimate. Possible spatial "
                    "stratifications are 'mun', 'arr', or 'prov'".format(agg)
                )
    
    ### Load all available data ###
    
    # Define absolute location of this file
    abs_dir = os.path.dirname(__file__)
    # Define data location for this particular aggregation level
    data_location = f'../../../data/covid19_DTM/interim/mobility/{agg}/fractional'
    
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
        if agg == 'prov':
            place = pd.read_csv(f'{directory}/{csv}', index_col='mllp_postalcode').values
        elif agg =='arr':
            place = pd.read_csv(f'{directory}/{csv}', index_col='mllp_postalcode').drop(index='Foreigner', columns='ABROAD').values
        # Create list of places
        all_available_places.append(place)
    # Create new empty dataframe with available dates. Load mobility later
    df = pd.DataFrame({'DATE' : all_available_dates, 'place' : all_available_places}).set_index('DATE')
    proximus_mobility_data = df.copy()
    
    return proximus_mobility_data


def update_staytime_mobility_matrix(raw_dir, interim_dir, agg='arr', verbose=True, normalise=True):
    """
    Function that turns the raw Proximus mobility data into interim mobility matrices P for the staytime. CURRENTLY NOT YET ACCOMODATING NEW DISTRICT-DISTRICT PROXIMUS DATA.
    
    Input
    -----
    raw_dir: str
        Directory that contains the raw Proximus data
    interim_dir: str
        Goal directory that contains the processed mobility P matrix data
    verbose: boolean
        If True (default), print current status
    normalise: boolean
        If True (default), normalise daily mobility such that the output is fractional mobility
    """
    
    # Find new dates that need to be processed
    dates_already_processed=[]
    if verbose:
        print(f"\nScanning interim directory {interim_dir}.")
    for csv in os.listdir(interim_dir):
        # take YYYYMMDD information from processed CSVs
        datum = csv[-12:-4]
        dates_already_processed.append(datum)
        
    if verbose:
        print(f"Scanning raw directory {raw_dir}.")
    dates_available=[]
    suffix_len = len(proximus_mobility_suffix())
    for csv in os.listdir(raw_dir):
        # Take YYYYMMDD information from raw CSVs
        os.rename(raw_dir + csv, raw_dir + csv.replace("_", "")) # newest files have unnecessary underscores
        datum = csv[-suffix_len-8:-suffix_len]
        dates_available.append(datum)
    raw_prefix = csv[0:-suffix_len-8]
    dates_new = sorted(np.setdiff1d(dates_available, dates_already_processed))
    if verbose:
        print(f"\nNew dates to be processed: {dates_new}\n")
    
    # Load and process data for new dates
    if normalise:
        savename = 'fractional-mobility-matrix_staytime_'
    else:
        savename = 'absolute-mobility-matrix_staytime_'
    for d in dates_new:
        raw_data = load_datafile_proximus(d, raw_dir)
        # Some seconds went missing. Show this per PC
        missing_seconds = missing_seconds_per_pc(raw_data)
        # Some est_staytime values were GDPR-protected. Estimate the value this should be changed with
        est_hidden_staytime = est_hidden_staytime_per_pc(raw_data)
        # Pivot unprocessed data into correct origin-destination mobility matrix
        mmprox_staytime = load_mmprox(raw_data, values='est_staytime')
        # Add missing postal codes (with value 0)
        mmprox_staytime = fill_missing_pc(mmprox_staytime)
        # Add missing seconds to 'stay at home' patch and subtract time spent asleep
        mmprox_staytime = complete_home_staytime(mmprox_staytime, missing_seconds)
        # Change GDPR-protected -1 values to estimated staytime value
        mmprox_staytime = GDPR_staytime(mmprox_staytime, est_hidden_staytime)
        # Aggregate staytime values at the level of agg (user-defined)
        mmprox_staytime_agg = mm_aggregate(mmprox_staytime, agg=agg)
        # Normalise the matrix
        if normalise:
            P = mmprox_staytime_agg.div(mmprox_staytime_agg.sum(axis=1), axis=0)
        else:
            P = mmprox_staytime_agg.copy()
        
        # Save matrix with descriptive name
        filename = savename + agg + '_' + d + '.csv'
        P.to_csv(interim_dir + filename)
        if verbose:
            print(f"Saved P for date {d}: {filename}", end='\r')

    return
            
def date_to_YYYYMMDD(date, inverse=False):
    """
    Simple function to convert a datetime object to a string representing the date in the shape YYYYMMDD
    
    Input
    -----
    date: datetime.date object or str
        datetime.date object if inverse=False, str if inverse=True
    inverse: boolean
        False if date is converted to YYYYMMDD, True if YYYYMMDD string is converted to datetime.date object
    
    Return
    ------
    YYYYMMDD: str
    """
    if not inverse:
#         Something is wrong with the exception below, not sure what
#         if isinstance(date, datetime.date):
#             raise Exception("First argument in function should be of type datetime.date. If type str (YYYYMMDD), set inverse=True.")
        YYYYMMDD = date.strftime("%Y%m%d")
        return YYYYMMDD
    if inverse:
        if not isinstance(date, str) or (len(date) != 8):
            raise Exception("First argument in function should str in form YYYYMMDD. If type is datetime.time, set inverse=False")
        datetime_object = datetime.strptime(date, "%Y%m%d")
        return datetime_object
        
# I think week_to_date() is obsolete
def week_to_date(week_nr, day = 1, year=2020):
    """
    Function that takes a week number between 1 and 53 (or 53 for 2021) and returns the corresponding dates
    
    Input
    -----
    week_nr: int
        Value from 1 to 54
    day: int
        Day of the week to return date for: 1 is first day, 7 is last day. Default is first day.
    year: int
        Year the week is in. Defaults to 2020.
    
    Returns
    -------
    date: datetime object
        Date of the requested day in format YYYY-MM-DD hh:mm:ss
    """
    
    if ((1 > week_nr) or (week_nr > 53)) and (year==2020):
        raise Exception("'week_nr' parameter must be an integer value from 1 to 53 for the year 2020.")
    if ((1 > week_nr) or (week_nr > 52)) and (year==2021):
        raise Exception("'week_nr' parameter must be an integer value from 1 to 52 for the year 2021.")
    if (day < 1) or (day > 7):
        raise Exception("'day' parameter must be an integer value from 1 to 7.")
    if day == 7:
        day = 0
    if year == 2020:
        d = str(year) + "-W" + str(week_nr - 1)
    if year == 2021:
        d = str(year) + "-W" + str(week_nr)
    date = datetime.strptime(d + '-' + str(day), "%Y-W%W-%w")
    return date

# I think make_date_list is obsolete
def make_date_list(week_nr, year=2020):
    """
    Makes list of dates in week 'week_nr' in the format needed for the identification of the Proximus data
    
    Input
    -----
    week_nr: int
        Value from 1 to 54
    year: int
        Year the week is in. Defaults to 2020.
        
    Returns
    -------
    date_list: list of str
        List of all 7 dates in week 'week_nr' of year 'year', in the format ['YYYYMMDD', ...]
    """
    if (1 > week_nr) or (week_nr > 54):
        raise Exception("'week_nr' parameter must be an integer value from 1 to 54.")
    date_list=[]
    for day in range(1,8):
        start_date = week_to_date(week_nr, day=day, year=year)
        YYYYMMDD = date_to_YYYYMMDD(start_date)
        date_list.append(YYYYMMDD)
    return date_list

def proximus_mobility_suffix():
    """
    Definition of the suffix in the mobility data files of Proximus.
    """
    suffix = "AZUREREF001.csv"
    return suffix

def check_missing_dates(dates, data_location):
    """
    Checks whether the requested dates correspond to existing Proximus mobility data and returns the missing dates
    
    Input
    -----
    dates: list of str
        Output of make_date_list. List for which the user wants to request data
    data_location: str
        Name of directory (relative or absolute) that contains all Proximus data files

    Returns
    -------
    missing: set of str
        All dates that do not correspond with a data file. Ideally, this set is empty.
    """
    suffix = proximus_mobility_suffix()
    full_list = []
    for f in os.listdir(data_location):
        date = f[:-len(suffix)][-8:]
        full_list.append(date)
    missing = set(dates).difference(set(full_list))
    return missing

def load_datafile_proximus(date, data_location, agg='postalcode'):
    """
    Load an entire raw Proximus data file
    
    Input
    -----
    date: str
        Single date in the shape YYYYMMDD
    data_location: str
        Name of directory (relative or absolute) that contains all Proximus data files
    agg: str
        'postalcode' (default) or 'arr'
        
    Output
    ------
    datafile: pandas DataFrame
    """
    if agg=='postalcode':
        suffix = proximus_mobility_suffix()
        datafile_name = data_location + 'outputPROXIMUS122747corona' + date + suffix
        # Note: the dtypes must not be int, because some int values are > 2^31 and this cannot be handled for int32
        datafile = pd.read_csv(datafile_name, sep=';', decimal=',', dtype={'mllp_postalcode' : str,
                                                                                             'postalcode' : str,
                                                                                             'imsisinpostalcode' : float,
                                                                                             'habitatants' : float,
                                                                                             'nrofimsi' : float,
                                                                                             'visitors' : float,
                                                                                             'est_staytime' : float,
                                                                                             'total_est_staytime' : float,
                                                                                             'est_staytime_perc' : float})
    elif agg=='arr':
        name1 = f'corona_report_proximus_district2district_{date}_{date}.csv'
        name2 = f'output_PROXIMUS_122747_coronadistrict_{date}_AZUREREF001.csv'
        try:
            datafile = pd.read_csv(data_location+name1, sep=';', decimal=',', dtype={'mllp_district' : str,
                                                                                     'district' : str,
                                                                                     'imsisindistrict' : float,
                                                                                     'habitatants' : float,
                                                                                     'nrofimsi' : float,
                                                                                     'visitors' : float,
                                                                                     'est_staytime' : float,
                                                                                     'total_est_staytime' : float,
                                                                                     'est_staytime_perc' : float})
        except:
            datafile = pd.read_csv(data_location+name2, sep=';', decimal=',', dtype={'mllp_district' : str,
                                                                                     'district' : str,
                                                                                     'imsisindistrict' : float,
                                                                                     'habitatants' : float,
                                                                                     'nrofimsi' : float,
                                                                                     'visitors' : float,
                                                                                     'est_staytime' : float,
                                                                                     'total_est_staytime' : float,
                                                                                     'est_staytime_perc' : float})
    else:
        raise Exception("agg must be 'arr' or 'postalcode'.")
    return datafile    
    
def load_mmprox(datafile, values='nrofimsi', agg='postalcode'):
    """
    Process raw Proximus datafile into a mobility matrix for either nrofimsi values or est_staytime values
    
    Input
    -----
    datafile: pandas.DataFrame
        loaded from load_datafile_proximus
    values: str
        Either 'nrofimsi' (visit counts) or 'est_staytime' (estimated staytime)
    agg: str
        'postalcode' (default) or 'arr'
    
    Output
    ------
    mmprox: pandas.DataFrame
        pandas DataFrames with visit counts or estimated staytime between postal codes.
    """
    if agg=='postalcode':
        mmprox = datafile.pivot_table(values=values,
                                  index='mllp_postalcode',
                                  columns='postalcode').fillna(value=0)
    elif agg=='arr':
        mmprox = datafile.pivot_table(values=values,
                                  index='mllp_district',
                                  columns='district').fillna(value=0)
    else:
        raise Exception("agg must be 'arr' or 'postalcode'.")
    return mmprox

def complete_home_staytime(mmprox, missing_seconds, minus_sleep=True):
    """
    Add missing seconds to home patch staytime
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
    missing_seconds: pandas DataFrame
        Output of missing_seconds_per_pc
    minus_sleep: boolean
        True if we do not include the time asleep as the active time of the day. Time asleep set to 8 hours.
        
    Returns
    -------
    mmprox_added_home_staytime: pandas DataFrame
        Same as input, but with added value for home patch
    """
    sleep_time= 8*60*60
    if not minus_sleep:
        sleep_time = 0
        print('no sleep')
    
    mmprox_added_home_staytime = mmprox.copy()
    for pc in missing_seconds.index:
            if pc != 'Foreigner':
                mmprox_added_home_staytime.loc[pc, pc] += missing_seconds.loc[pc, 'missing_seconds'] \
                                                            - sleep_time * missing_seconds.loc[pc, 'imsisinpostalcode']
            else:
                mmprox_added_home_staytime.loc[pc, 'ABROAD'] += missing_seconds.loc[pc, 'missing_seconds'] \
                                                            - sleep_time * missing_seconds.loc[pc, 'imsisinpostalcode']
            
    return mmprox_added_home_staytime

def GDPR_staytime(mmprox, values=0):
    """
    Changes every -1 value for the staytime in the origin-destination matrix with the corresponding estimated value
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and est_staytime as values
    values: float or pandas.DataFrame
        Either a value (None by defulOutput of est_hidden_staytime_per_pc: indices are origin postal codes, column contains estimated staytime values
        
    Returns
    -------
    mmprox_added_hidden_staytime: pandas DataFrame
        Dataframe identical to mmprox (input), but with every -1 value changed by the corresponding estimated time
    """
    # Make series from est_hidden_staytime
    if type(values) == pd.core.frame.DataFrame:
        values = pd.Series(data=est_hidden_staytime['est_hidden_staytime'], index=est_hidden_staytime.index)
    
    # Replace every -1 value with the corresponding estimated value that is protected
    mmprox_added_hidden_staytime = mmprox.mask(mmprox==-1, other=values, axis=0)
    
    return mmprox_added_hidden_staytime
    
def load_Pmatrix_staytime(dates, data_location, complete=False, verbose=True, return_missing=False, agg='mun'):
    """
    Load clean mobility P matrices in a big dictionary, and return missing dates if needed
    
    Input
    -----
    dates: str or list of str
        Single date in YYYYMMDD form or list of these (output of make_date_list function): requested date(s)
    data_location: str
        Name of directory (relative or absolute) that contains all processed (interim) mobility matrices
    complete: boolean
        If True, this function raises an exception when 'dates' contains a date that does not correspond to a data file.
    verbose: boolean
        If True, print statement every time data for a date is loaded.
    return_missing: boolean
        If True, return array of missing dates in form YYYYMMDD as second return. False by default.
    agg: str
        Aggregation level: 'mun', 'arr' or 'prov'
    
    Returns
    -------
    mmprox_dict: dict of pandas DataFrames
        Dictionary with YYYYMMDD dates as keys and pandas DataFrames with fractional mobility (from staytime) as values
    """
    # Check dates type and change to single-element list if needed
    single_date = False
    if isinstance(dates,str):
        dates = [dates]
        single_date = True

    # Make list of dates in data_location and save the difference
    suffix = ".csv"
    full_list = []
    for f in os.listdir(data_location):
        date = f[:-len(suffix)][-8:]
        full_list.append(date)
    missing = set(dates).difference(set(full_list))
    
    # Dates to be loaded are all dates, except the ones that are missing
    load_dates = set(dates).difference(missing)
    dates_left = len(load_dates)
    if dates_left == 0:
        raise Exception("None of the requested dates correspond to a fractional staytime origin-destination matrix.")
    if missing != set():
        print(f"Warning: some or all of the requested dates do not correspond to a fractional staytime origin-destination matrix. Dates: {sorted(missing)}")
        if complete:
            raise Exception("Some requested data is not found amongst the fractional staytime origin-destination matrices. Set 'complete' parameter to 'False' and rerun if you wish to proceed with an incomplete data set (not using all requested data).")
        print(f"... proceeding with {dates_left} dates.")

    # Initiate dict for remaining dates and load files
    mmprox_dict=dict({})
    filename = 'fractional-mobility-matrix_staytime_' + agg + '_'
    load_dates = sorted(list(load_dates))
    for date in load_dates:
        datafile = pd.read_csv(data_location + filename + date + ".csv", index_col = 'mllp_postalcode')
        mmprox_dict[date] = datafile
        if verbose==True:
            print(f"Loaded dataframe for date {date}.    ", end='\r')
    print(f"Loaded dataframe for date {date}.")
    
    if not return_missing:
        return mmprox_dict
    else:
        return mmprox_dict, sorted(missing)
    
    
    
def load_mobility_proximus(dates, data_location, values='nrofimsi', complete=False, verbose=True, return_missing=False):
    """
    Load Proximus mobility data (number of visitors or visitor time) corresponding to the requested dates
    
    Input
    -----
    dates: str or list of str
        Single date in YYYYMMDD form or list of these (output of make_date_list function): requested date(s)
    data_location: str
        Name of directory (relative or absolute) that contains all Proximus data files
    values: str
        Choose between absolute visitor count ('nrofimsi', default) or other values of interest (e.g. 'est_staytime') on one day.
        Special case for values='est_staytime' or 'total_est_staytime': the absolute value is taken
    complete: boolean
        If True, this function raises an exception when 'dates' contains a date that does not correspond to a data file.
    verbose: boolean
        If True, print statement every time data for a date is loaded.
    return_missing: boolean
        If True, return array of missing dates in form YYYYMMDD as second return. False by default.
    
    Returns
    -------
    mmprox_dict: dict of pandas DataFrames
        Dictionary with YYYYMMDD dates as keys and pandas DataFrames with visit counts or estimated staytime between postal codes.
    """
    # Check dates type and change to single-element list if needed
    single_date = False
    if isinstance(dates,str):
        dates = [dates]
        single_date = True
    
    missing = check_missing_dates(dates, data_location)
    load_dates = set(dates).difference(missing)
    dates_left = len(load_dates)
    if dates_left == 0:
        raise Exception("None of the requested dates correspond to a Proximus mobility file.")
    if missing != set():
        print(f"Warning: some or all of the requested dates do not correspond to Proximus data. Dates: {sorted(missing)}")
        if complete:
            raise Exception("Some requested data is not found amongst the Proximus files. Set 'complete' parameter to 'False' and rerun if you wish to proceed with an incomplete data set (not using all requested data).")
        print(f"... proceeding with {dates_left} dates.")

    # Initiate dict for remaining dates
    mmprox_dict=dict({})
    load_dates = sorted(list(load_dates))
    for date in load_dates:
        datafile = load_datafile_proximus(date, data_location)
        mmprox_temp = datafile[['mllp_postalcode', 'postalcode', values]]
        mmprox_temp = mmprox_temp.pivot_table(values=values,
                                              index='mllp_postalcode',
                                              columns='postalcode')
        mmprox_temp = mmprox_temp.fillna(value=0)
        if values in ['est_staytime', 'total_est_staytime']:
            mmprox_dict[date] = mmprox_temp.convert_dtypes().abs()
        else:
            mmprox_dict[date] = mmprox_temp.convert_dtypes()
        if verbose==True:
            print(f"Loaded dataframe for date {date}.    ", end='\r')
    print(f"Loaded dataframe for date {date}.")
    
    if not return_missing:
        return mmprox_dict
    else:
        return mmprox_dict, sorted(missing)

def missing_seconds_per_pc(datafile):
    """
    For all postalcodes (mllp_postalcode) mentioned in datafile, calculate (or estimate, if GDPR-protection is in force) the discrepancy between the total available time (i.e. imsisinpostalcode * 24 hours * 60 minutes * 60 seconds) and the time on record (total_est_staytime). This time must have gone SOMEwhere, so we may add it to e.g. the diagonal elements (time spent from X in X). Additionally, this function estimates imsisinpostalcode when it is GDPR-protected.
    
    Input
    -----
    datafile: pandas DataFrame
        Raw data: output of load_datafile_proximus function
        
    Returns
    -------
    missing_seconds_per_pc: pandas DataFrame
        DataFrame with all mllp_postalcodes from datafile as indices. Column "imsisinpostalcode" contains the (estimated) number of clients in the postal code. "total_est_staytime" the total estimated staytime (copied from datafile). "total_available_time" is imsisinpostalcode * 24 * 60 * 60. "missing_seconds" is total - registered amount of time.
    """
    # Pivot table such that postalcodes are indices (~1120 entries)
    missing_seconds_per_pc = pd.pivot_table(datafile, index='mllp_postalcode', \
                                     values=['imsisinpostalcode', 'total_est_staytime'], aggfunc='first')

    # Filter those entries with non-censored data and make a copy
    missing_seconds_per_pc_available = missing_seconds_per_pc.loc[missing_seconds_per_pc['imsisinpostalcode']>0].copy()
    
    # Create new column with number of people per second of total staytime (0 < number < 1)
    missing_seconds_per_pc_available['pop_per_staytime'] = missing_seconds_per_pc_available['imsisinpostalcode'] \
        / missing_seconds_per_pc_available['total_est_staytime']

    # calculate median number of people per staytime (0 < number < 1)
    median_people_per_staytime = missing_seconds_per_pc_available['pop_per_staytime'].median()
    #print("Median number of people per staytime:", median_people_per_staytime)

    # Change negative values with estimated number of people living somewhere
    missing_seconds_per_pc.loc[missing_seconds_per_pc['imsisinpostalcode']<0, 'imsisinpostalcode'] \
        = missing_seconds_per_pc.loc[missing_seconds_per_pc['imsisinpostalcode']<0, 'total_est_staytime'] * median_people_per_staytime
    
    # Add column with the maximum number of seconds available for all registered clients combined
    missing_seconds_per_pc['total_available_time'] = missing_seconds_per_pc['imsisinpostalcode'] * 24 * 60 * 60
    
    # Add column with the missing seconds for every entry (i.e. for every postal code mentioned in datafile)
    missing_seconds_per_pc['missing_seconds'] = missing_seconds_per_pc['total_available_time'] \
                                                    - missing_seconds_per_pc['total_est_staytime']
    
    return missing_seconds_per_pc



def load_pc_to_nis():
    # Data source
    abs_dir = os.path.dirname(__file__)
    pc_to_nis_file = os.path.join(abs_dir, '../../../data/covid19_DTM/raw/GIS/Postcode_Niscode.xlsx')
    pc_to_nis_df = pd.read_excel(pc_to_nis_file)[['Postcode', 'NISCode']]
    return pc_to_nis_df
    
def show_missing_pc(mmprox):
    """
    Function to return the missing postal codes in Proximus mobility data frames.
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values

    Returns
    -------
    from_PCs_missing: array of str
        Missing postal codes in the indices (missing information FROM where people are coming)
    to_PCs_missing: array of str
        Missing postal codes in the columns (missing information WHERE people are going)
    """
    pc_to_nis = load_pc_to_nis()
    all_PCs = set(pc_to_nis['Postcode'].values)
    index_list=list(mmprox.index.values)
    if 'Foreigner' in index_list:
        index_list.remove('Foreigner')
    column_list=list(mmprox.columns.values)
    if 'ABROAD' in column_list:
        column_list.remove('ABROAD')
    from_PCs = set(np.array(index_list).astype(int))
    to_PCs = set(np.array(column_list).astype(int))

    from_PCs_missing = all_PCs.difference(from_PCs)
    to_PCs_missing = all_PCs.difference(to_PCs)
    
    from_PCs_missing = np.array(sorted(from_PCs_missing)).astype(str)
    to_PCs_missing = np.array(sorted(to_PCs_missing)).astype(str)
    
    return from_PCs_missing, to_PCs_missing

def fill_missing_pc(mmprox):
    """
    Function that adds the missing postal codes to the mobility matrix and fills these with zeros.
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
        
    Returns
    -------
    mmprox_complete: pandas DataFrame
        Square mobility matrix with all from/to postal codes
    """
    mmprox_complete = mmprox.copy()
    from_PCs_missing, to_PCs_missing = show_missing_pc(mmprox_complete)
    # Add missing PCs as empty rows/columns (and turn the matrix square)
    for pc in from_PCs_missing:
        mmprox_complete.loc[pc] = 0
    for pc in to_PCs_missing:
        mmprox_complete[pc] = 0

    if 'Foreigner' not in mmprox_complete.index:
        mmprox_complete.loc['Foreigner'] = 0
    if 'ABROAD' not in mmprox_complete.columns:
        mmprox_complete['ABROAD'] = 0
        
    return mmprox_complete
    


# What to dot with GDPR-protected <30 values
# TO DO: this needs some work
def GDPR_exponential_pick_visits(avg=5):
    """
    Choose a value ranging from 1 to 30 with an exponential distribution. Used to change the GDPR-protected value -1 for the number of visits, with an estimated number that it actually signifies.
    
    Input
    -----
    avg: int
        Expectation value of the exponential distribution. Default: 5
        
    Returns
    -------
    number: int
        Integer value between 1 and 30, with a higher chance for lower numbers.
    """
    number = np.ceil(np.random.exponential(scale=avg))
    if number > 30:
        number = 30
    return number

# TO BE CHANGED!
def GDPR_exponential_pick_length(avg=10000):
    """
    Choose a value ranging from an exponential distribution. Used to change the GDPR-protected value -1 for the length of stay, with an estimated number that it actually signifies.
    
    Input
    -----
    avg: int
        Expectation value of the exponential distribution. Default: 10000
        
    Returns
    -------
    number: int
        Integer value, with a higher chance for lower numbers.
    """
    return

def GDPR_replace(mmprox, replace_func=GDPR_exponential_pick_visits, **kwargs):
    """
    Function to replace the -1 values that denote small values proteted by the GDPR privacy protocol
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
    replace_func: function
        Function used to determine the value that -1 is to be replaced with
        
c
    """
    values = mmprox.values
    for (x,y), value in np.ndenumerate(values):
        if value < 0:
            values[x,y] = replace_func(**kwargs)
    mmprox_GDPR = pd.DataFrame(values, columns=mmprox.columns, index=mmprox.index)

    return mmprox_GDPR


def est_hidden_staytime_per_pc(datafile):
    """
    Replace all staytime values that are protected by GDPR protocol (-1 values) with an estimate.
    TODO: include several estimate types. Uses missing_seconds_per_pc as input to find total_est_staytime.
    
    Input
    -----
    datafile: pandas DataFrame
        Raw data: output of load_datafile_proximus function
    
    Returns
    -------
    est_hidden_staytime: pandas.DataFrame
        DataFrame with postal codes as indices and estimated staytimes as values (replacements for -1 values for that particular postal code)
    """
    # Load missing seconds per postal code
    missing_seconds = missing_seconds_per_pc(datafile)
    
    # Simple "raw" grid of "people from g spend x time in h". 0 values means none are registered, -1 means it is protected
    staytime_matrix = pd.pivot_table(datafile, index='mllp_postalcode', columns='postalcode', \
                                         values='est_staytime').fillna(value=0)

    # Total non-GDPR protected time per PC (should be smaller than total_est_time)
    est_staytime_noGDPR = staytime_matrix[staytime_matrix>0].fillna(value=0).sum(axis=1)    

    # Number of times -1 occurs per PC (i.e. number of postal codes for which data is protected)
    # Note: this is a lot! The maximum is 700 (which is more than half of all 1148 PCs)
    small_number_freq = staytime_matrix[staytime_matrix==-1].fillna(value=0).sum(axis=1).astype(int).abs()

    # Average time difference per GDPR-protected postalcode, between total registered time and the sum of the individually registered times
    est_staytime_GDPR = (missing_seconds['total_est_staytime'] - est_staytime_noGDPR) / small_number_freq
    
    # Raise exception if some of these values are negative
    if (est_staytime_GDPR<0).any():
        raise Exception("The total_est_staytime value is bigger than the sum of all non-GDPR-protected est_staytime values. Are your sure you used the raw data as input?")

    # For most mllp_postalcode, est_staytime_GDPR is not a big value (order of couple hours)
    # For foreigners it is of the order of a few days. Reasoning:
    #   1. The registered total_est_staytime is large
    #   2. The est_staytimes that are registered are low (just over threshold)
    # This is due to the nature of foreigners visiting Belgium: they travel around much (truckers) and typically don't
    # Stay at a particular place for a long time.
    
    est_hidden_staytime = pd.DataFrame(est_staytime_GDPR, columns=['est_hidden_staytime'])
    return est_hidden_staytime

# Aggregate

def mm_aggregate(mmprox, agg='mun', from_agg='postalcode'):
    """
    Aggregate cleaned-up mobility dataframes at the aggregation level of municipalities, arrondissements or provinces
    
    Input
    -----
    mmprox: pandas DataFrame
        Mobility matrix with postal codes as indices and as column heads, and visit counts or visit lenghts as values
    agg: str
        The level at which to aggregate. Choose between 'mun', 'arr' or 'prov'. Default is 'mun'.
    from_agg: str
        The starting level from which to aggregate. Only 'arr' implemented so far. 'postalcode' does nothing.
        
    Output
    ------
    mmprox_agg: pandas DataFrame
        Mobility dataframe aggregated at the level 'agg'
    """
    # validate
    mmprox_shape = mmprox.shape
    if from_agg != 'arr' and mmprox_shape != (1148, 1148):
        raise Exception(f"The input dataframe is of the shape {mmprox_shape}, not (1148, 1148) which is all 1147 postal codes + destinations/origins abroad. Fix this first.")
    if agg not in ['mun', 'arr', 'prov']:
        raise Exception("The aggregation level must be either municipality ('mun'), arrondissements ('arr') or provinces ('prov').")
    
    if agg=='prov' and from_agg=='arr':
        # Rename columns
        for nis in mmprox.columns:
            if nis not in ['ABROAD', '21000', '23000', '24000', '25000']: # Brussels is '11th province'
                new_nis = nis[:-4] + '0000'
                mmprox = mmprox.rename(columns={nis : new_nis})
            if nis in ['23000', '24000']:
                new_nis = '20001'
                mmprox = mmprox.rename(columns={nis : new_nis})
            if nis == '25000':
                new_nis = '20002'
                mmprox = mmprox.rename(columns={nis : new_nis})

        # Rename rows
        for nis in mmprox.index:
            if nis not in ['Foreigner', '21000', '23000', '24000', '25000']:
                new_nis = nis[:-4] + '0000'
                mmprox = mmprox.rename(index={nis : new_nis})
            if nis in ['23000', '24000']:
                new_nis = '20001'
                mmprox = mmprox.rename(index={nis : new_nis})
            if nis == '25000':
                new_nis = '20002'
                mmprox = mmprox.rename(index={nis : new_nis})

        # Collect rows and columns with the same NIS code, and automatically order column/row names
        mmprox_agg = mmprox.groupby(level=0, axis=1).sum()
        mmprox_agg = mmprox.groupby(level=0, axis=0).sum()#.astype(int)
    
    else:
        # copy dataframe and load the postal-code-to-NIS-value translator
        mmprox_agg = mmprox.copy()
        pc_to_nis = load_pc_to_nis()

        rename_abroad = 'ABROAD'
        rename_foreigner = 'Foreigner'

        # initiate renaming dictionaries
        rename_col_dict = dict({})
        rename_idx_dict = dict({})
        for pc in mmprox_agg.columns:
            if pc != 'ABROAD':
                NIS = str(pc_to_nis[pc_to_nis['Postcode']==int(pc)]['NISCode'].values[0])
                rename_col_dict[pc] = NIS
        rename_col_dict['ABROAD'] = rename_abroad
        for pc in mmprox_agg.index:
            if pc != 'Foreigner':
                NIS = str(pc_to_nis[pc_to_nis['Postcode']==int(pc)]['NISCode'].values[0])
                rename_idx_dict[pc] = NIS
        rename_idx_dict['Foreigner'] = rename_foreigner

        # Rename the column names and indices to prepare for merging
        mmprox_agg = mmprox_agg.rename(columns=rename_col_dict, index=rename_idx_dict)

        mmprox_agg = mmprox_agg.groupby(level=0, axis=1).sum()
        mmprox_agg = mmprox_agg.groupby(level=0, axis=0).sum()#.astype(int)

        if agg in ['arr', 'prov']:
            # Rename columns
            for nis in mmprox_agg.columns:
                if nis != 'ABROAD':
                    new_nis = nis[:-3] + '000'
                    mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})

            # Rename rows
            for nis in mmprox_agg.index:
                if nis != 'Foreigner':
                    new_nis = nis[:-3] + '000'
                    mmprox_agg = mmprox_agg.rename(index={nis : new_nis})

            # Collect rows and columns with the same NIS code, and automatically order column/row names
            mmprox_agg = mmprox_agg.groupby(level=0, axis=1).sum()
            mmprox_agg = mmprox_agg.groupby(level=0, axis=0).sum()#.astype(int)

            if agg == 'prov':
                # Rename columns
                for nis in mmprox_agg.columns:
                    if nis not in ['ABROAD', '21000', '23000', '24000', '25000']: # Brussels is '11th province'
                        new_nis = nis[:-4] + '0000'
                        mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})
                    if nis in ['23000', '24000']:
                        new_nis = '20001'
                        mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})
                    if nis == '25000':
                        new_nis = '20002'
                        mmprox_agg = mmprox_agg.rename(columns={nis : new_nis})

                # Rename rows
                for nis in mmprox_agg.index:
                    if nis not in ['Foreigner', '21000', '23000', '24000', '25000']:
                        new_nis = nis[:-4] + '0000'
                        mmprox_agg = mmprox_agg.rename(index={nis : new_nis})
                    if nis in ['23000', '24000']:
                        new_nis = '20001'
                        mmprox_agg = mmprox_agg.rename(index={nis : new_nis})
                    if nis == '25000':
                        new_nis = '20002'
                        mmprox_agg = mmprox_agg.rename(index={nis : new_nis})

                # Collect rows and columns with the same NIS code, and automatically order column/row names
                mmprox_agg = mmprox_agg.groupby(level=0, axis=1).sum()
                mmprox_agg = mmprox_agg.groupby(level=0, axis=0).sum()#.astype(int)
    
    return mmprox_agg
    
def complete_data_clean(mmprox, agg='mun'):
    """
    Execute several standard functions at the same time
    """
    mmprox_clean = mm_aggregate( fill_missing_pc( GDPR_replace(mmprox) ), agg=agg)
    return mmprox_clean

def complete_data_clean_new(unprocessed_data, initN):
    """
    unprocessed_data: DataFrame
        Loaded with load_datafile_proximus()
    initN: np.array
        Loaded with initN, _, _, _ = get_model_parameters(spatial='prov')
        
    Example use:
        unprocessed_data = load_datafile_proximus(pd.Timestamp(2021, 1, 1), data_location)
        initN, _, _, _ = get_model_parameters(spatial='prov')
        mmprox_clean = complete_data_clean_new(unprocessed_data, initN)
    """
    # Takes about twenty seconds per date
    # Make array of all unique postalcodes in the data
    pc_array = unprocessed_data.mllp_postalcode.unique()

    # Loop over all postal codes and fill in GDPR protected data with an estimate
    # It is important to do this in the first step: this will compensate automatically at the provincial level
    for pc in pc_array:
        # Define two masks
        mask_mllp = unprocessed_data['mllp_postalcode']==pc
        mask_GDPR = unprocessed_data['est_staytime']==-1

        # Number of GDPR protected data points for this particular postalcode
        GDPR_protected_size = unprocessed_data[mask_mllp & mask_GDPR].est_staytime.size

        # Calculate the reported total est_staytime versus the calculated (summed) total est_staytime
        total_est_staytime = unprocessed_data.loc[mask_mllp, 'total_est_staytime'].values[0]
        summed_est_staytime = unprocessed_data[mask_mllp][['est_staytime']].sum(axis=0).values[0]

        # Calculate number of hidden seconds and make up for summing the -1 values
        hidden_seconds = total_est_staytime - summed_est_staytime + GDPR_protected_size
        hidden_seconds_per_pc = hidden_seconds / GDPR_protected_size

        # Distribute all hidden seconds evenly to all GDPR-protected postalcodes
        unprocessed_data.loc[mask_mllp & mask_GDPR,'est_staytime'] = hidden_seconds_per_pc
        
    # Next, merge all postalcodes together in provinces
    # Note that this still contains 'Foreigner' and 'ABROAD'

    # pc-to-pc matrix
    mmprox = load_mmprox(unprocessed_data, values='est_staytime')

    # Add postalcodes that Proximus has missed
    mmprox = fill_missing_pc(mmprox)

    # Aggregate at the level of provinces
    mmprox = mm_aggregate(mmprox, agg='prov')

    # Loop over province NIS values and add unregistered waking-life seconds
    for NIS in initN.index:
        NIS_pop = initN.sum(axis=1).loc[NIS]
        available_seconds_NIS = NIS_pop * 60 * 60 * 16 # 8 hours asleep
        registered_seconds_NIS = mmprox.loc[str(NIS)].sum()
        unregistered_seconds_NIS = available_seconds_NIS - registered_seconds_NIS
        mmprox.loc[str(NIS), str(NIS)] += unregistered_seconds_NIS
        
    # Now we only have to deal with the 'ABROAD' category, which might be a pain in the ass ...
    # The final step is to just normalise everything

    mmprox_clean = mmprox.drop(index=['Foreigner'], columns=['ABROAD'])
    mmprox_clean = mmprox_clean.div(mmprox_BE.sum(axis=1),axis=0)

    # Clearly the vast majority of time is spent in the home province, which obiously makes sense.
    
    return mmprox_clean

# Temporal aggregation/averaging
def average_mobility(mmprox_dict):
    """
    Calculates the average mobility over all dates in the mmprox_dict dictionary
    
    Input
    -----
    mmprox_dict: dict
        Dictionary with YYYYMMDD strings as keys and pandas DataFrames with clean Proximus data as values
        
    Output
    ------
    mmprox_avg: pandas DataFrame
        Mobility DataFrame with average values of all DataFrames in input
    """
    # Access first dict element and demand all are of the same shape
    matrix_shape = list(mmprox_dict.values())[0].shape
    for date in mmprox_dict:
        matrix_shape_next = mmprox_dict[date].shape
        if matrix_shape_next != matrix_shape:
            raise Exception("All shapes of the mobility matrices must be the same.")
        matrix_shape = matrix_shape_next

    first=True
    for date in mmprox_dict:
        if not first:
            mmprox_avg = mmprox_avg.add(mmprox_dict[date])
        if first:
            mmprox_avg = mmprox_dict[date]
            first=False
            
    mmprox_avg = mmprox_avg / len(mmprox_dict)
    return mmprox_avg

