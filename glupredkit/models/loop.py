from glupredkit.models.base_model import BaseModel
from glupredkit.helpers.scikit_learn import process_data
import datetime
import pandas as pd
import numpy as np
from pyloopkit.dose import DoseType
from pyloopkit.loop_data_manager import update


class Model(BaseModel):

    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.DIA = 360  # TODO: Is this the default in Loop?
        self.subject_ids = None
        self.basal = []
        self.insulin_sensitivity_factor = []
        self.carb_ratio = []

    def fit(self, x_train, y_train):
        self.subject_ids = x_train['id'].unique()
        x_train['insulin'] = x_train['bolus'] + (x_train['basal'] / 12)

        # TODO: Create a map for those values? Or store subject_ids in a list?
        for subject_id in self.subject_ids:
            filter_df = x_train['id'] == subject_id
            # Calculate total daily insulin
            daily_avg_insulin = np.mean(x_train[filter_df].groupby(pd.Grouper(freq='D')).agg({'insulin': 'sum'}))

            print(f"Daily average insulin for subject {subject_id}: ", daily_avg_insulin)

            self.insulin_sensitivity_factor += [1800 / daily_avg_insulin]  # ISF 1800 rule
            self.carb_ratio += [500 / daily_avg_insulin]  # CR 500 rule
            self.basal += [daily_avg_insulin * 0.45 / 24]  # Basal 45% of TDI

        print(f"Therapy settings: ISF {self.insulin_sensitivity_factor}, CR: {self.carb_ratio}, basal: {self.basal}")

        return self

    def predict(self, x_test):
        """
        Return:
        y_pred -- A list of lists of the predicted trajectories.
        """
        n_predictions = x_test.shape[0]

        if n_predictions <= 0:
            print("Not enough data to predict. Needs to be at least 24 hours plus duration of insulin absorption.")
            return

        # Note that the prediction output starts at the reference value, so element 1 is the first prediction
        prediction_index = int(self.prediction_horizon / 5)

        # For each glucose measurement, get a new prediction
        # Skip iteration if there are not enough predictions.
        y_pred = []

        for index, subject_id in enumerate(self.subject_ids):
            df_subset = x_test[x_test['id'] == subject_id]
            n_predictions = df_subset.shape[0]
            input_dict = self.get_input_dict(self.insulin_sensitivity_factor[index], self.carb_ratio[index],
                                             self.basal[index])

            for i in range(0, n_predictions):
                current_data = df_subset.iloc[i]
                output_dict = self.get_prediction_output(current_data, input_dict)

                if i % 50 == 0 and i != 0:  # Check if i is a multiple of 50 and not 0
                    print(f"Prediction number {i} of {n_predictions} for {subject_id}")

                if len(output_dict.get("predicted_glucose_values")) < prediction_index:
                    print("Not enough predictions. Skipping iteration...")
                    # TODO: Here we should just repeat the last predicted value until enough predictions
                    continue
                else:
                    current_predictions = output_dict.get("predicted_glucose_values")[1:prediction_index + 1]
                    y_pred.append(current_predictions)

        return np.array(y_pred)

    def get_prediction_output(self, df_row, input_dict, time_to_calculate=None):
        # Important docs: https://github.com/miriamkw/PyLoopKit/blob/develop/pyloopkit/docs/pyloopkit_documentation.md
        # For correct predictions, at least 24 hours + duration of insulin absorption (DIA) of data is needed
        # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
        if not time_to_calculate:
            time_to_calculate = df_row.name

        input_dict["time_to_calculate_at"] = time_to_calculate

        def get_dates_and_values(column, data):
            relevant_columns = [val for val in data.index if val.startswith(column)]
            dates = []
            values = []
            for col in relevant_columns:
                if col == column:
                    values.append(data[col])
                    dates.append(data.name)
                elif "what_if" in col:
                    values.append(data[col])
                    new_date = data.name + datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)
                else:
                    values.append(data[col])
                    new_date = data.name - datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)

            if not dates or not values:
                # Handle the case where one or both lists are empty
                pass
            else:
                # Sorting
                combined = list(zip(dates, values))
                # Sort by the dates (first element of each tuple)
                combined.sort(key=lambda x: x[0])
                # Separate into individual lists
                dates, values = zip(*combined)

            return dates, values

        glucose_dates, glucose_values = get_dates_and_values("CGM", df_row)
        input_dict["glucose_dates"] = glucose_dates
        input_dict["glucose_values"] = glucose_values

        dose_types, start_times, end_times, dose_values, dose_delivered_units = self.get_insulin_data(df_row)

        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        carb_data = df_row[~df_row.index.str.startswith('carbs') | (df_row != 0)]
        carb_dates, carb_values = get_dates_and_values("carbs", carb_data)
        input_dict["carb_dates"] = carb_dates
        input_dict["carb_values"] = carb_values

        # Adding the default carb absorption time because it is not available in data sources.
        input_dict["carb_absorption_times"] = [180 for _ in carb_values]

        return update(input_dict)


    def get_insulin_data(self, df_row):
        def get_dose_type(x):
            if x == 'temp':
                return DoseType.from_str("tempbasal")
            elif x == 'bolus':
                return DoseType.from_str("bolus")
            else:
                return DoseType.from_str("basal")

        def get_dates_and_values(column, data):
            relevant_columns = [val for val in data.index if val.startswith(column)]
            dates = []
            values = []
            for col in relevant_columns:
                if col == column:
                    values.append(data[col])
                    dates.append(data.name)
                elif "what_if" in col:
                    values.append(data[col])
                    new_date = data.name + datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)
                else:
                    values.append(data[col])
                    new_date = data.name - datetime.timedelta(minutes=int(col.split("_")[-1]))
                    dates.append(new_date)
            return dates, values

        basal_dates, basal_values = get_dates_and_values("basal", df_row)
        # Using tempbasal as default, assuming that users are using
        basal_dose_types = [get_dose_type("tempbasal") for _ in basal_dates]
        basal_start_times = basal_dates
        basal_end_times = [val + pd.Timedelta(minutes=5) for val in basal_dates]
        # TODO: Are basals in U or U/hr in dataframe? PyLoopKit is expecting U/hr
        # basal_values = [value / 5 * 60 for value in df.basal.tolist()]
        basal_values = basal_values
        basal_units = [None for _ in basal_values]

        bolus_data = df_row[~df_row.index.str.startswith('bolus') | (df_row != 0)]
        bolus_dates, bolus_values = get_dates_and_values("bolus", bolus_data)
        bolus_dose_types = [get_dose_type("bolus") for _ in bolus_values]
        bolus_start_times = bolus_dates
        bolus_end_times = bolus_dates
        bolus_units = [None for _ in bolus_values]

        # Step 1: Combine into tuples
        combined_basal = list(zip(basal_dose_types, basal_start_times, basal_end_times, basal_values, basal_units))
        combined_bolus = list(zip(bolus_dose_types, bolus_start_times, bolus_end_times, bolus_values, bolus_units))

        # Step 2: Merge lists
        combined = combined_basal + combined_bolus

        # Step 3: Sort by the start times (second element of each tuple)
        combined.sort(key=lambda x: x[1])

        # Step 4: Separate into individual lists
        dose_types, start_times, end_times, values, units = zip(*combined)

        # Convert the tuples back to lists if needed
        dose_types = list(dose_types)
        start_times = list(start_times)
        end_times = list(end_times)
        values = list(values)
        units = list(units)

        return dose_types, start_times, end_times, values, units

    def get_input_dict(self, insulin_sensitivity, carb_ratio, basal):
        return ({
            'carb_value_units': 'g',
            'settings_dictionary':
                {
                    'model': [self.DIA, 75],
                    'momentum_data_interval': 15.0,
                    'suspend_threshold': None,
                    'dynamic_carb_absorption_enabled': True,
                    'retrospective_correction_integration_interval': 30,
                    'recency_interval': 15,
                    'retrospective_correction_grouping_interval': 30,
                    'rate_rounder': 0.05,
                    'insulin_delay': 10,
                    'carb_delay': 0,
                    'default_absorption_times': [120.0, 180.0, 240.0],
                    'max_basal_rate': 2.5,  # Doesn't matter for prediction, but mandatory to include
                    'max_bolus': 12.0,  # Doesn't matter for prediction, but mandatory to include
                    'retrospective_correction_enabled': True
                },
            'sensitivity_ratio_start_times': [datetime.time(0, 0)],
            'sensitivity_ratio_end_times': [datetime.time(0, 0)],
            'sensitivity_ratio_values': [insulin_sensitivity],
            'sensitivity_ratio_value_units': 'mg/dL/U',

            'carb_ratio_start_times': [datetime.time(0, 0)],
            'carb_ratio_values': [carb_ratio],
            'carb_ratio_value_units': 'g/U',

            'basal_rate_start_times': [datetime.time(0, 0)],
            'basal_rate_minutes': [1440],  # the length of time the basal runs for (in minutes)
            'basal_rate_values': [basal],  # the infusion rate in U/hour

            'target_range_start_times': [datetime.time(0, 0)],
            'target_range_end_times': [datetime.time(0, 0)],
            'target_range_minimum_values': [100.0],
            'target_range_maximum_values': [114.0],
            'target_range_value_units': 'mg/dL',

            'last_temporary_basal': []
        })

    def process_data(self, df, model_config_manager, real_time):
        return process_data(df, model_config_manager, real_time)

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None
