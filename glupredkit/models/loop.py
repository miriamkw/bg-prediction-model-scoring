from glupredkit.models.base_model import BaseModel
from glupredkit.helpers.unit_config_manager import unit_config_manager
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyloopkit.dose import DoseType
from pyloopkit.loop_math import predict_glucose
from pyloopkit.loop_data_manager import update


class Model(BaseModel):

    def __init__(self, prediction_horizon):
        super().__init__(prediction_horizon)
        self.DIA = 360

    def fit(self, x_train, y_train):
        # Manually adjust the therapy settings at the bottom of this file to adjust the prediction model

        # TODO: Add inputs for user-defined adjustments, like DIA, insulin absorption model...
        # TODO: It should be a therapy settings dict as input here

        return self

    def predict(self, x_test):
        """
        Return:
        y_pred -- A list of lists of the predicted trajectories.
        """

        history_length = 24 * 12 + int(self.DIA / 5)
        n_predictions = x_test.shape[0] - history_length

        if n_predictions <= 0:
            print("Not enough data to predict. Needs to be at least 24 hours plus duration of insulin absorption.")
            return

        # Note that the prediction output starts at the reference value, so element 1 is the first prediction
        prediction_index = int(self.prediction_horizon / 5)
        input_dict = self.get_input_dict()

        # For each glucose measurement, get a new prediction
        # Skip iteration if there are not enough predictions.
        y_pred = []
        for i in range(0, n_predictions):
            df_subset = x_test.iloc[i:i + history_length]
            output_dict = self.get_prediction_output(df_subset, input_dict)

            if len(output_dict.get("predicted_glucose_values")) < 73:
                print("Not enough predictions. Skipping iteration...")
                continue
            else:
                y_pred.append(output_dict.get("predicted_glucose_values")[prediction_index])
        return y_pred

    def get_prediction_output(self, df, input_dict, time_to_calculate=None):
        # Important docs: https://github.com/miriamkw/PyLoopKit/blob/develop/pyloopkit/docs/pyloopkit_documentation.md
        # For correct predictions, at least 24 hours + duration of insulin absorption (DIA) of data is needed
        # NOTE: Not filtering away future glucose values will lead to erroneous prediction results!
        if not time_to_calculate:
            time_to_calculate = df.index[-1]

        history_length = 24 * 12 - int(self.DIA / 5)
        n_predictions = df.shape[0] - history_length

        if n_predictions <= 0:
            print("Not enough data to predict. Needs to be at least 24 hours plus duration of insulin absorption.")
            return

        filtered_df = df[df.index <= time_to_calculate].tail(history_length)

        input_dict["time_to_calculate_at"] = time_to_calculate

        input_dict["glucose_dates"] = filtered_df.index.tolist()
        input_dict["glucose_values"] = filtered_df.CGM.tolist()

        dose_types, start_times, end_times, dose_values, dose_delivered_units = self.get_insulin_data(df)

        input_dict["dose_types"] = dose_types
        input_dict["dose_start_times"] = start_times
        input_dict["dose_end_times"] = end_times
        input_dict["dose_values"] = dose_values
        input_dict["dose_delivered_units"] = dose_delivered_units

        input_dict["carb_dates"] = filtered_df[filtered_df.carbs > 0].index.tolist()
        input_dict["carb_values"] = filtered_df[filtered_df.carbs > 0].carbs.tolist()
        # TODO: Add proper carb absorption times (if in metadata of data source...)
        input_dict["carb_absorption_times"] = [180 for _ in filtered_df[filtered_df.carbs > 0].index.tolist()]

        # TODO: Test and make sure that the predictions are indeed correct
        return update(input_dict)

    # TODO: Implement the setting of changing units
    def plot_prediction(self, df):
        recommendations = self.get_prediction_output(df)

        dates = recommendations.get("predicted_glucose_dates")
        values = recommendations.get("predicted_glucose_values")

        inputs = recommendations.get("input_data")
        previous_glucose_dates = inputs.get("glucose_dates")[-15:]
        previous_glucose_values = inputs.get("glucose_values")[-15:]

        starting_date = previous_glucose_dates[-1]
        starting_glucose = previous_glucose_values[-1]

        (momentum_predicted_glucose_dates,
         momentum_predicted_glucose_values
         ) = predict_glucose(
            starting_date, starting_glucose,
            momentum_dates=recommendations.get("momentum_effect_dates"),
            momentum_values=recommendations.get("momentum_effect_values")
        )

        (insulin_predicted_glucose_dates,
         insulin_predicted_glucose_values
         ) = predict_glucose(
            starting_date, starting_glucose,
            insulin_effect_dates=recommendations.get("insulin_effect_dates"),
            insulin_effect_values=recommendations.get("insulin_effect_values")
        )

        (carb_predicted_glucose_dates,
         carb_predicted_glucose_values
         ) = predict_glucose(
            starting_date, starting_glucose,
            carb_effect_dates=recommendations.get("carb_effect_dates"),
            carb_effect_values=recommendations.get("carb_effect_values")
        )

        if recommendations.get("retrospective_effect_dates"):
            (retrospective_predicted_glucose_dates,
             retrospective_predicted_glucose_values
             ) = predict_glucose(
                starting_date, starting_glucose,
                correction_effect_dates=recommendations.get(
                    "retrospective_effect_dates"
                ),
                correction_effect_values=recommendations.get(
                    "retrospective_effect_values"
                )
            )
        else:
            (retrospective_predicted_glucose_dates,
             retrospective_predicted_glucose_values
             ) = ([], [])

        unit = "mg/dL"
        y_min = 70
        y_max = 180
        if not unit_config_manager.use_mgdl:
            values = [unit_config_manager.convert_value(val) for val in values]
            previous_glucose_values = [unit_config_manager.convert_value(val) for val in previous_glucose_values]
            momentum_predicted_glucose_values = [unit_config_manager.convert_value(val) for val in
                                                 momentum_predicted_glucose_values]
            insulin_predicted_glucose_values = [unit_config_manager.convert_value(val) for val in
                                                insulin_predicted_glucose_values]
            carb_predicted_glucose_values = [unit_config_manager.convert_value(val) for val in
                                             carb_predicted_glucose_values]
            retrospective_predicted_glucose_values = [unit_config_manager.convert_value(val) for val in
                                                      retrospective_predicted_glucose_values]
            unit = "mmol/L"
            y_min = unit_config_manager.convert_value(y_min)
            y_max = unit_config_manager.convert_value(y_max)

        plt.figure(figsize=(10, 5))
        plt.plot(dates, values, marker='o', label='Prediction')
        plt.plot(previous_glucose_dates, previous_glucose_values, marker='o', label='Measurements')
        plt.plot(momentum_predicted_glucose_dates, momentum_predicted_glucose_values, marker='o', label='Momentum')
        plt.plot(insulin_predicted_glucose_dates, insulin_predicted_glucose_values, marker='o', label='Insulin')
        plt.plot(carb_predicted_glucose_dates, carb_predicted_glucose_values, marker='o', label='Carbohydrates')
        plt.plot(retrospective_predicted_glucose_dates, retrospective_predicted_glucose_values, marker='o',
                 label='Retrospective')

        plt.axhspan(y_min, y_max, facecolor='blue', alpha=0.1)

        # Formatting the plot
        plt.xlabel('Date')
        plt.ylabel(f'Blood glucose value [{unit}]')
        plt.title('Predicted Trajectory Contributions')
        plt.legend()

        # Format the x-axis to show only the time (hours and minutes)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        # TODO: Fix the time zone conversion

        # Show the plot
        plt.show()

    def get_insulin_data(self, df):
        def get_dose_type(x):
            if x == 'temp':
                return DoseType.from_str("tempbasal")
            elif x == 'bolus':
                return DoseType.from_str("bolus")
            else:
                return DoseType.from_str("basal")

        basal_dose_types = [get_dose_type(basal_type) for basal_type in df.basal_type.tolist()]
        basal_start_times = df.index.tolist()
        basal_end_times = [val + pd.Timedelta(minutes=5) for val in df.index.tolist()]
        basal_values = [value / 5 * 60 for value in df.basal.tolist()]
        basal_units = [None for _ in df.index.tolist()]

        df_bolus = df.copy()[df.bolus > 0]
        bolus_dose_types = [get_dose_type("bolus") for _ in df_bolus.index.tolist()]
        bolus_start_times = df_bolus.index.tolist()
        bolus_end_times = df_bolus.index.tolist()
        bolus_values = df_bolus.bolus.tolist()
        bolus_units = [None for _ in df_bolus.index.tolist()]

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

    def get_input_dict(self):
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
                    'carb_delay': 10,
                    'default_absorption_times': [120.0, 180.0, 240.0],
                    'max_basal_rate': 2.5,
                    'max_bolus': 12.0,
                    'retrospective_correction_enabled': True
                },
            'sensitivity_ratio_start_times': [datetime.time(0, 0), datetime.time(0, 30)],
            'sensitivity_ratio_end_times': [datetime.time(0, 30), datetime.time(0, 0)],
            'sensitivity_ratio_values': [50.4, 50.4],
            'sensitivity_ratio_value_units': 'mg/dL/U',

            'carb_ratio_start_times': [datetime.time(0, 0), datetime.time(8, 30)],
            'carb_ratio_values': [9.0, 9.0],
            'carb_ratio_value_units': 'g/U',

            'basal_rate_start_times': [datetime.time(0, 0), datetime.time(12, 0)],
            'basal_rate_minutes': [720, 720],  # the length of time the basal runs for (in minutes)
            'basal_rate_values': [0.9, 0.9],  # the infusion rate in U/hour

            'target_range_start_times': [datetime.time(0, 0)],
            'target_range_end_times': [datetime.time(0, 0)],
            'target_range_minimum_values': [100.0],
            'target_range_maximum_values': [114.0],
            'target_range_value_units': 'mg/dL',

            'last_temporary_basal': []
        })

    def process_data(self, df, model_config_manager, real_time):
        # Implement library specific preprocessing steps that are required before training a pandas dataframe
        return df

    def best_params(self):
        # Return the best parameters found by GridSearchCV
        return None
