import requests
from aiohttp import ClientError, ClientConnectorError, ClientResponseError
import nightscout
from .base_parser import BaseParser
import pandas as pd
import datetime
import json
import os
import urllib.parse
import numpy as np

# Monkey patch the Treatment class for Loop/Trio compatibility
from nightscout.models import Treatment

original_init = Treatment.__init__

def new_init(self, **kwargs):
    self.param_defaults = {
        'temp': None,
        'enteredBy': None,
        'eventType': None,
        'glucose': None,
        'glucoseType': None,
        'units': None,
        'device': None,
        'created_at': None,
        'timestamp': None,
        'absolute': None,
        'percent': None,
        'percentage': None,
        'rate': None,
        'duration': None,
        'carbs': None,
        'insulin': None,
        'unabsorbed': None,
        'suspended': None,
        'type': None,
        'programmed': None,
        'foodType': None,
        'absorptionTime': None,
        'profile': None,
        'insulinNeedsScaleFactor': None,  # Added for Loop/Trio
        'reason': None,  # Added for Loop/Trio
        'automatic': None  # Added for Loop/Trio
    }
    for (param, default) in self.param_defaults.items():
        setattr(self, param, kwargs.get(param, default))
    original_init(self, **kwargs)

Treatment.__init__ = new_init

class Parser(BaseParser):
    def __init__(self):
        super().__init__()
        self.test_mode = False
        self.test_data_dir = None

    def __call__(self, start_date, end_date, username: str, password: str, test_mode=False, test_data_dir=None):
        """
        Main method to parse Nightscout data with enhanced validation.
        In the nightscout parser, the username is the nightscout URL, and the password is the API key.
        """
        try:
            if test_mode:
                # Load test data from local files
                from pathlib import Path
                test_dir = Path(test_data_dir)
                
                with open(test_dir / "nightscout_profiles.json") as f:
                    profiles = json.load(f)
                
                with open(test_dir / "nightscout_treatments.json") as f:
                    treatments_data = json.load(f)
                treatments = []
                for t in treatments_data:
                    treatment = type('Treatment', (), {})()
                    for k, v in t.items():
                        setattr(treatment, k, v)
                    treatments.append(treatment)
                
                with open(test_dir / "nightscout_entries.json") as f:
                    entries_data = json.load(f)
                entries = []
                for e in entries_data:
                    entry = type('Entry', (), {})()
                    for k, v in e.items():
                        setattr(entry, k, v)
                    entries.append(entry)

            else:
                api = nightscout.Api(username, api_secret=password)
                api_start_date = start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                api_end_date = end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z')

                # Fetch profiles
                profile_query_params = {
                    'count': 0,
                    'find[created_at][$gte]': api_start_date,
                    'find[created_at][$lte]': api_end_date
                }
                base_url = username.rstrip('/')
                profile_url = f"{base_url}/api/v1/profile?{urllib.parse.urlencode(profile_query_params)}"
                profiles_response = requests.get(profile_url, headers=api.request_headers())
                profiles = profiles_response.json()
                self.save_json_profiles(profiles, 'profiles', api_start_date, api_end_date)

                # Fetch treatments
                treatment_query_params = {
                    'count': 0,
                    'find[created_at][$gte]': api_start_date,
                    'find[created_at][$lte]': api_end_date
                }
                treatments = api.get_treatments(treatment_query_params)
                self.save_json(treatments, 'treatments', api_start_date, api_end_date)

                # Fetch entries (CGM data)
                query_params = {
                    'count': 0,
                    'find[dateString][$gte]': api_start_date,
                    'find[dateString][$lte]': api_end_date
                }
                entries = api.get_sgvs(query_params)
                self.save_json(entries, 'entries', api_start_date, api_end_date)

            # Process CGM data
            df_glucose = self.create_dataframe(entries, 'date', 'sgv', 'CGM')
            # Handle CGM values - replace 0s with NaN and interpolate
            df_glucose['CGM'] = df_glucose['CGM'].replace(0, np.nan)
            df_glucose['CGM'] = df_glucose['CGM'].interpolate(
                method='time', 
                limit=3  # max 15 minutes gap
            )
            print("Created Glucose DataFrame")

            # Process carbs
            df_carbs = self.create_dataframe(treatments, 'created_at', 'carbs', 'carbs',
                                           event_type=['Carb Correction', 'Meal Bolus', 'Snack Bolus'])
            print("Created Carbs DataFrame")

            # Process bolus insulin
            df_bolus = self.create_dataframe(treatments, 'created_at', 'insulin', 'bolus',
                                           event_type=['Bolus', 'Meal Bolus', 'Snack Bolus',
                                                     'Correction Bolus', 'SMB'])
            print("Created Bolus DataFrame")

            # Process temporary basal rates
            df_temp_basal = self.create_dataframe(treatments, 'created_at', ['absolute', 'rate'],
                                                'basal', event_type='Temp Basal')
            df_temp_duration = self.create_dataframe(treatments, 'created_at', 'duration',
                                                   'duration', event_type='Temp Basal')
            print("Created Temporary Basal DataFrame")

            # Get and process basal profiles
            basal_rates = self.get_basal_rates_from_profile(profiles)
            df_basal_profile = self.create_basal_dataframe([start_date, end_date], basal_rates)
            print("Created Profile Basal DataFrame")

            # Process profile switches and apply them
            df_profile_switches = self.create_profile_switches_df(treatments)
            df_basal_profile = self.apply_profile_switches(df_basal_profile, df_profile_switches, profiles)
            print("Applied Profile Switches")

            # Initialize main dataframe with glucose data
            df = df_glucose.resample('5min').mean()

            # Merge all components
            df = self.merge_and_process(df, df_carbs, 'carbs')
            df = self.merge_and_process(df, df_bolus, 'bolus')
            df = self.merge_basal_rates(df, df_basal_profile, df_temp_basal, df_temp_duration)

            # Convert basal rates from U/hr to U/5min and ensure no negative values
            df['basal'] = df['basal'].apply(lambda x: max(0, float(x)))
            df['basal'] = round(df['basal'] / 60 * 5, 5)  # Convert from U/hr to U/5min
            df['basal'] = df['basal'].fillna(0)
            
            # Ensure no negative values in bolus
            df['bolus'] = df['bolus'].apply(lambda x: max(0, float(x))).fillna(0)
            
            # Calculate total insulin
            df['insulin'] = df['bolus'] + df['basal']

            # Convert timezone
            current_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
            df.index = df.index.tz_convert(current_timezone)

            # Add additional columns
            df['hour'] = df.index.hour
            df['id'] = 1
            df['is_test'] = False

            # Final validation for negative values
            for col in ['basal', 'bolus', 'insulin', 'carbs']:
                if (df[col] < 0).any():
                    print(f"Warning: Found negative values in {col}, converting to absolute values")
                    df[col] = df[col].abs()

            # Reorder columns
            df = df.reset_index()
            df = df[['date', 'id', 'CGM', 'insulin', 'carbs', 'is_test', 'hour', 'basal', 'bolus']]
            
            # Final cleanup
            for col in ['insulin', 'carbs', 'basal', 'bolus']:
                df[col] = df[col].fillna(0)

            # Drop rows where CGM is still NaN after interpolation
            df = df.dropna(subset=['CGM'])
            
            df.set_index('date', inplace=True)
            print("Final DataFrame Created")

            # Verify treatments
            df = self.verify_treatments(treatments, df)

            return df

        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            raise

    def merge_basal_rates(self, df, df_profile_basal, df_temp_basal, df_temp_duration):
        """Merge profile basal rates with temporary basal overrides, ensuring no negative values."""
        # Start with profile basal rates
        df = df.merge(df_profile_basal, left_index=True, right_index=True, how='left')
        df['basal'] = df['basal'].fillna(0)
        
        if not df_temp_basal.empty:
            temp_basals = pd.concat([df_temp_basal, df_temp_duration], axis=1)
            temp_basals.columns = ['basal', 'percent_x', 'duration', 'percent_y']
            
            # Sort temp basals chronologically
            temp_basals = temp_basals.sort_index()
            
            for time, row in temp_basals.iterrows():
                if pd.isna(row['duration']) or float(row['duration']) <= 0:
                    continue
                    
                end_time = time + pd.Timedelta(minutes=float(row['duration']))
                mask = (df.index >= time) & (df.index < end_time)
                
                # Get base rates for this period
                base_rates = df.loc[mask, 'basal'].copy()
                
                if pd.notnull(row['basal']) and float(row['basal']) >= 0:
                    # Absolute temp basal
                    df.loc[mask, 'basal'] = float(row['basal'])
                elif pd.notnull(row['percent_x']):
                    # Percentage temp basal
                    percent = float(row['percent_x'])
                    multiplier = max(0, percent / 100)  # Ensure non-negative
                    df.loc[mask, 'basal'] = base_rates * multiplier
        
        # Final validation
        df['basal'] = df['basal'].abs()
        return df

    def create_profile_switches_df(self, treatments):
        """Create DataFrame for profile switches and temporary overrides."""
        switches = []
        for treatment in treatments:
            if not hasattr(treatment, 'eventType'):
                continue
                
            switch = {
                'date': pd.to_datetime(treatment.created_at, utc=True),
                'profile': None,
                'scale_factor': 1.0,
                'duration': None,
                'source': None
            }
            
            if treatment.eventType == 'Profile Switch':
                switch.update({
                    'profile': getattr(treatment, 'profile', None),
                    'duration': getattr(treatment, 'duration', None),
                    'source': 'AndroidAPS'
                })
                if switch['profile'] is not None:
                    switches.append(switch)
                    
            elif treatment.eventType == 'Temporary Override':
                scale_factor = getattr(treatment, 'insulinNeedsScaleFactor', None)
                if scale_factor is not None:
                    switch.update({
                        'scale_factor': float(scale_factor),
                        'duration': getattr(treatment, 'duration', None),
                        'source': 'Loop' if 'Loop' in getattr(treatment, 'enteredBy', '') else 'Trio'
                    })
                    switches.append(switch)
        
        if not switches:
            return pd.DataFrame(columns=['profile', 'scale_factor', 'duration', 'source'])
        
        df = pd.DataFrame(switches)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        return df

    def apply_profile_switches(self, df_basal, profile_switches, profiles):
        """Apply profile switches and temporary overrides to basal rates with validation."""
        if profile_switches.empty:
            return df_basal
        
        store = profiles[0].get('store', {})
        result_df = df_basal.copy()
        
        for time, row in profile_switches.iterrows():
            if pd.notnull(row['duration']):
                end_time = time + pd.Timedelta(minutes=float(row['duration']))
            else:
                later_switches = profile_switches.index[profile_switches.index > time]
                end_time = later_switches[0] if len(later_switches) > 0 else df_basal.index[-1]
            
            mask = (result_df.index >= time) & (result_df.index < end_time)
            
            if row['source'] == 'AndroidAPS':
                if row['profile'] and row['profile'] in store:
                    new_profile = store[row['profile']]
                    new_basal_rates = []
                    for entry in new_profile.get('basal', []):
                        seconds = entry.get('timeAsSeconds', 0)
                        rate = max(0, float(entry.get('value', 0)))  # Ensure non-negative
                        new_basal_rates.append((seconds, rate))
                    new_basal_rates.sort()
                    
                    for idx in result_df[mask].index:
                        seconds = (idx.hour * 3600 + idx.minute * 60 + idx.second)
                        rate = self.get_basal_rate_for_time(new_basal_rates, seconds)
                        result_df.loc[idx, 'basal'] = rate
                        
            elif row['source'] in ['Loop', 'Trio']:
                scale_factor = max(0, float(row['scale_factor']))  # Ensure non-negative
                result_df.loc[mask, 'basal'] *= scale_factor
        
        # Final validation
        result_df['basal'] = result_df['basal'].abs()
        return result_df

    def get_basal_rates_from_profile(self, profiles):
        """Extract basal rates from the default profile, ensuring non-negative values."""
        if not profiles or len(profiles) == 0:
            return []
            
        default_profile_name = profiles[0].get('defaultProfile')
        if not default_profile_name:
            return []
            
        store = profiles[0].get('store', {})
        if not store or default_profile_name not in store:
            return []
            
        default_profile = store[default_profile_name]
        basal_schedule = default_profile.get('basal', [])
        
        def time_to_seconds(time_str):
            """Convert time string (HH:MM) to seconds since midnight."""
            hours, minutes = map(int, time_str.split(':'))
            return hours * 3600 + minutes * 60
        
        # Convert to list of (seconds, rate) tuples, ensuring non-negative rates
        basal_rates = []
        for entry in basal_schedule:
            seconds = entry.get('timeAsSeconds', None)
            if seconds is None:
                seconds = time_to_seconds(entry['time'])
            # Ensure rate is non-negative
            rate = max(0, float(entry.get('value', 0)))
            basal_rates.append((seconds, rate))
        
        return sorted(basal_rates)

    def get_basal_rate_for_time(self, basal_rates, seconds_since_midnight):
        """Get the appropriate basal rate for a given time, ensuring non-negative values."""
        if not basal_rates:
            return 0.0
            
        # Find the last basal rate that started before or at this time
        applicable_rate = basal_rates[0][1]  # Default to first rate
        for time_sec, rate in basal_rates:
            if time_sec <= seconds_since_midnight:
                applicable_rate = rate
            else:
                break
        return max(0, float(applicable_rate))  # Ensure non-negative

    def create_basal_dataframe(self, date_range, basal_rates):
        """Create a DataFrame with basal rates for every 5 minutes in the date range."""
        dates = []
        rates = []
        
        current_date = date_range[0]
        while current_date <= date_range[1]:
            seconds = (current_date.hour * 3600 + 
                    current_date.minute * 60 + 
                    current_date.second)
            
            # Get rate and ensure it's non-negative
            rate = max(0, float(self.get_basal_rate_for_time(basal_rates, seconds)))
            rate = round(rate, 5)
            
            dates.append(current_date)
            rates.append(rate)
            
            current_date += datetime.timedelta(minutes=5)
        
        df = pd.DataFrame({'date': dates, 'basal': rates})
        df['basal'] = df['basal'].fillna(0)  # Fill any NaN basal rates with 0
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df.set_index('date', inplace=True)
        return df

    def create_dataframe(self, data, date_column, value_column, new_column_name, event_type=None):
        """Create a DataFrame from the given data, ensuring non-negative values."""
        dates = []
        values = []
        percents = []
        
        for entry in data:
            try:
                if event_type:
                    # Handle treatments (insulin, carbs, etc.)
                    if isinstance(event_type, list):
                        if any(et in entry.eventType for et in event_type):
                            dates.append(pd.to_datetime(getattr(entry, date_column), utc=True))
                            if isinstance(value_column, list):
                                value = getattr(entry, value_column[0], None)
                                if value is None or pd.isna(value):
                                    value = getattr(entry, value_column[1], 0)
                            else:
                                value = getattr(entry, value_column, 0)
                            # Ensure non-negative values
                            value = max(0, float(value)) if pd.notnull(value) else 00
                            values.append(value)
                            percent = getattr(entry, 'percent', 0)
                            percent = max(0, float(percent)) if pd.notnull(percent) else 0
                            percents.append(percent)
                    elif event_type in entry.eventType:
                        dates.append(pd.to_datetime(getattr(entry, date_column), utc=True))
                        if isinstance(value_column, list):
                            value = getattr(entry, value_column[0], None)
                            if value is None or pd.isna(value):
                                value = getattr(entry, value_column[1], 0)
                        else:
                            value = getattr(entry, value_column, 0)
                        # Ensure non-negative values
                        value = max(0, float(value)) if pd.notnull(value) else 0
                        values.append(value)
                        percent = getattr(entry, 'percent', 0)
                        percent = max(0, float(percent)) if pd.notnull(percent) else 0
                        percents.append(percent)
                else:
                    # Handle entries (glucose values)
                    if hasattr(entry, 'dateString'):
                        date_value = pd.to_datetime(entry.dateString, utc=True)
                    elif hasattr(entry, 'date'):
                        try:
                            date_value = pd.to_datetime(entry.date, utc=True)
                        except (TypeError, ValueError):
                            date_value = pd.to_datetime(entry.date, unit='ms', utc=True)
                    else:
                        raise AttributeError(f"No valid date field found in entry")

                    dates.append(date_value)
                    value = getattr(entry, value_column, 0)
                    # For glucose values, we don't force non-negative as they might be special codes
                    values.append(value if pd.notnull(value) else 0)
                    percents.append(0)
            
            except Exception as e:
                print(f"Error processing entry: {entry}")
                print(f"Error details: {str(e)}")
                continue
        
        df = pd.DataFrame({
            'date': dates,
            new_column_name: values,
            'percent': percents
        })
        
        if not df.empty:
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        
        return df

    def merge_and_process(self, df, df_to_merge, column_name):
        """Merge and process dataframes ensuring non-negative values."""
        if not df_to_merge.empty:
            # Convert index to exact 5-minute marks
            df_to_merge.index = df_to_merge.index.round('5min')
            
            # Ensure non-negative values before resampling
            if column_name in ['basal', 'bolus', 'insulin']:
                df_to_merge[column_name] = df_to_merge[column_name].apply(lambda x: max(0, float(x)))
            
            # Fill NaN values
            df_to_merge[column_name] = df_to_merge[column_name].fillna(0)
            
            # For bolus data, use last() instead of sum()
            if column_name == 'bolus':
                df_to_merge = df_to_merge.resample('5min').last().fillna(0)
            else:
                df_to_merge = df_to_merge.resample('5min').sum().fillna(0)
            
            # Merge with original dataframe
            df = pd.merge(df, df_to_merge, left_index=True, right_index=True, how='outer')
            
            # Fill NaN values after merge
            df[column_name] = df[column_name].fillna(0)
            
            # Final validation for non-negative values
            if column_name in ['basal', 'bolus', 'insulin']:
                df[column_name] = df[column_name].apply(lambda x: max(0, float(x)))
            
        else:
            df[column_name] = 0
        
        return df

    def verify_treatments(self, treatments, final_df):
        """Verify treatments and ensure non-negative values."""
        print("\nVerifying treatments capture:")
        
        for treatment in treatments:
            treatment_time = pd.to_datetime(treatment.created_at).tz_convert(final_df.index.tz)
            rounded_time = treatment_time.round('5min')
            
            if hasattr(treatment, 'insulin') and treatment.insulin:
                insulin_value = max(0, float(treatment.insulin)) if not pd.isna(treatment.insulin) else 0
                if rounded_time in final_df.index:
                    df_value = final_df.loc[rounded_time, 'bolus']
                    print(f"Treatment insulin: {insulin_value}, DataFrame bolus: {df_value}")
            
            if hasattr(treatment, 'carbs') and treatment.carbs:
                carbs_value = max(0, float(treatment.carbs)) if not pd.isna(treatment.carbs) else 0
                if rounded_time in final_df.index:
                    df_value = final_df.loc[rounded_time, 'carbs']
                    print(f"Treatment carbs: {carbs_value}, DataFrame carbs: {df_value}")

        return final_df

    def save_json(self, data, data_type, start_date, end_date):
        """Save raw data to JSON file."""
        os.makedirs('data/raw', exist_ok=True)
        filename = f'data/raw/{data_type}_{start_date}_{end_date}.json'
        with open(filename, 'w') as f:
            json.dump([self.entry_to_dict(entry) for entry in data], f, indent=2, default=str)

    def save_json_profiles(self, profiles, data_type, start_date, end_date):
        """Save profiles data to JSON file."""
        os.makedirs('data/raw', exist_ok=True)
        filename = f'data/raw/{data_type}_{start_date}_{end_date}.json'
        with open(filename, 'w') as f:
            json.dump(profiles, f, indent=2)

    def entry_to_dict(self, entry):
        """Convert entry object to dictionary."""
        if hasattr(entry, '_json'):
            return entry._json
        elif hasattr(entry, '__dict__'):
            data = {}
            for key, value in entry.__dict__.items():
                if not key.startswith('_'):
                    data[key] = value
            return data
        return dict(entry)