"""Model Inputs"""
# import mom_trans.utils as utils
import numpy as np
import sklearn.preprocessing
import pandas as pd
import datetime as dt
import enum

from sklearn.preprocessing import MinMaxScaler

# Type defintions
class DataTypes(enum.IntEnum):
    """Defines numerical types of each column."""

    REAL_VALUED = 0
    CATEGORICAL = 1
    DATE = 2


class InputTypes(enum.IntEnum):
    """Defines input types of each column."""

    TARGET = 0
    OBSERVED_INPUT = 1
    KNOWN_INPUT = 2
    STATIC_INPUT = 3
    ID = 4  # Single column used as an entity identifier
    TIME = 5  # Single column used as a time index


def get_single_col_by_input_type(input_type, column_definition):
    """Returns name of single column.
    Args:
      input_type: Input type of column to extract
      column_definition: Column definition list for experiment
    """

    l = [tup[0] for tup in column_definition if tup[2] == input_type]

    if len(l) != 1:
        raise ValueError("Invalid number of columns for {}".format(input_type))

    return l[0]


def extract_cols_from_data_type(data_type, column_definition, excluded_input_types):
    """Extracts the names of columns that correspond to a define data_type.
    Args:
      data_type: DataType of columns to extract.
      column_definition: Column definition to use.
      excluded_input_types: Set of input types to exclude
    Returns:
      List of names for columns with data type specified.
    """
    return [
        tup[0]
        for tup in column_definition
        if tup[1] == data_type and tup[2] not in excluded_input_types
    ]


class ModelFeatures:
    """Defines and formats data for the MomentumCp dataset.
    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    def __init__(
        self,
        df,
        total_time_steps,
        start_boundary=1990,
        test_boundary=2020,
        test_end=2021,
        changepoint_lbws=None,
        train_valid_sliding=False,
        # add_buffer_years_to_test=1,  # TODO FIX THIS!!!!
        transform_real_inputs=False,  # TODO remove this
        train_valid_ratio=0.9,
        split_tickers_individually=True,
        add_ticker_as_static=False,
        time_features=False,
        lags=None,
        asset_class_dictionary=None,
        static_ticker_type_feature = False,
    ):
        """Initialises formatter. Splits data frame into training-validation-test data frames.
        This also calibrates scaling object, and transforms data for each split."""
        self._column_definition = [
            ("ticker", DataTypes.CATEGORICAL, InputTypes.ID),
            ("date", DataTypes.DATE, InputTypes.TIME),
            ("target_returns", DataTypes.REAL_VALUED, InputTypes.TARGET),
            ("norm_daily_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_monthly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_quarterly_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_biannual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("norm_annual_return", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_8_24", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_16_48", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
            ("macd_32_96", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ]
        df = df.dropna()
        df = df[df["year"] >= start_boundary].copy()
        years = df["year"]

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None
        self.total_time_steps = total_time_steps
        self.lags = lags

        if changepoint_lbws:
            for lbw in changepoint_lbws:
                self._column_definition.append(
                    (f"cp_score_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )
                self._column_definition.append(
                    (f"cp_rl_{lbw}", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
                )

        if time_features:
            self._column_definition.append(
                ("days_from_start", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                ("day_of_week", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                (f"day_of_month", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            self._column_definition.append(
                (f"week_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            )
            # self._column_definition.append(
            #     (f"month_of_year", DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
            # )

            # dataframe could have later years
            start_date = dt.datetime(start_boundary, 1, 1)
            days_from_start_max = (dt.datetime(test_end - 1, 12, 31) - start_date).days
            df["days_from_start"] = (df.index - start_date).days
            df["days_from_start"] = np.minimum(
                df["days_from_start"], days_from_start_max
            )

            df["days_from_start"] = (
                MinMaxScaler().fit_transform(df[["days_from_start"]].values).flatten()
            )
            df["day_of_week"] = (
                MinMaxScaler().fit_transform(df[["day_of_week"]].values).flatten()
            )
            df["day_of_month"] = (
                MinMaxScaler().fit_transform(df[["day_of_month"]].values).flatten()
            )
            df["week_of_year"] = (
                MinMaxScaler().fit_transform(df[["week_of_year"]].values).flatten()
            )
            # df["month_of_year"] = MinMaxScaler().fit_transform(df[["month_of_year"]].values).flatten()

        if add_ticker_as_static:
            self._column_definition.append(
                (f"static_ticker", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
            )
            df["static_ticker"] = df["ticker"]
            if static_ticker_type_feature:
                df["static_ticker_type"] = df["ticker"].map(
                    lambda t: asset_class_dictionary[t]
                )
                self._column_definition.append(
                    (
                        f"static_ticker_type",
                        DataTypes.CATEGORICAL,
                        InputTypes.STATIC_INPUT,
                    )
                )

        self.transform_real_inputs = transform_real_inputs

        # for static_variables
        # self._column_definition.append(("ticker", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT))

        # Handle very short time windows (like 8 minutes of intraday data)
        data_duration = df.index.max() - df.index.min()
        print(f"Total data duration: {data_duration}")
        print(f"Data frequency: ~{len(df) / (data_duration.total_seconds() / 60):.1f} records per minute")
        
        # Convert boundaries to datetime - handle intraday data
        if isinstance(test_boundary, int):
            # For year boundaries with intraday data, use the year but keep time precision
            test_boundary = pd.Timestamp(f"{test_boundary}-01-01 09:30:00")
        elif isinstance(test_boundary, str):
            test_boundary = pd.to_datetime(test_boundary)
            # If only date provided for intraday data, set to market open
            if test_boundary.time() == pd.Timestamp("00:00:00").time():
                test_boundary = test_boundary.replace(hour=9, minute=30, second=0, microsecond=0)
        elif isinstance(test_boundary, pd.Timestamp):
            test_boundary = test_boundary
        else:
            test_boundary = pd.to_datetime(test_boundary)
        
        if isinstance(test_end, int):
            # For year boundaries, set to end of trading day
            test_end = pd.Timestamp(f"{test_end}-12-31 16:00:00")
        elif isinstance(test_end, str):
            test_end = pd.to_datetime(test_end)
            # If only date provided, set to market close
            if test_end.time() == pd.Timestamp("00:00:00").time():
                test_end = test_end.replace(hour=16, minute=0, second=0, microsecond=0)
        elif isinstance(test_end, pd.Timestamp):
            test_end = test_end
        else:
            test_end = pd.to_datetime(test_end)
        
        print(f"Data date range: {df.index.min()} to {df.index.max()}")
        print(f"Test boundary: {test_boundary}")
        print(f"Test end: {test_end}")
        
        # For very short time windows, adjust test_boundary to be within data range
        if test_boundary < df.index.min() or test_boundary > df.index.max():
            # Calculate boundary as percentage of data duration
            data_start = df.index.min()
            data_end = df.index.max()
            total_duration = data_end - data_start
            
            # Use 70% of data for train+valid, 30% for test
            test_start_offset = total_duration * 0.7
            test_boundary = data_start + test_start_offset
            test_end = data_end
            
            print(f"Adjusted test_boundary to: {test_boundary}")
            print(f"Adjusted test_end to: {test_end}")
        
        # Similarly adjust test_end if needed
        if test_end < df.index.min() or test_end > df.index.max():
            test_end = df.index.max()
            print(f"Adjusted test_end to data maximum: {test_end}")
        
        # Split data using datetime comparison instead of year
        test = df.loc[df.index >= test_boundary]
        print(f"Test data shape: {test.shape}")

        if split_tickers_individually:
            trainvalid = df.loc[df.index < test_boundary]
            print(f"Train+Valid data shape: {trainvalid.shape}")
            
            if lags:
                tickers = (
                    trainvalid.groupby("ticker")["ticker"].count()
                    * (1.0 - train_valid_ratio)
                ) >= total_time_steps
                tickers = tickers[tickers].index.tolist()
            else:
                tickers = list(trainvalid.ticker.unique())
            
            print(f"Available tickers: {len(tickers)}")
            if len(tickers) == 0:
                raise ValueError("No tickers found in training data")

            train, valid = [], []
            for ticker in tickers:
                calib_data = trainvalid[trainvalid.ticker == ticker]
                T = len(calib_data)
                if T == 0:
                    print(f"WARNING: No data for ticker {ticker}")
                    continue
                    
                train_valid_split = int(train_valid_ratio * T)
                if train_valid_split == 0:
                    print(f"WARNING: No training data for ticker {ticker} (T={T})")
                    continue
                    
                train_data = calib_data.iloc[:train_valid_split, :].copy()
                valid_data = calib_data.iloc[train_valid_split:, :].copy()
                
                if len(train_data) > 0:
                    train.append(train_data)
                if len(valid_data) > 0:
                    valid.append(valid_data)
            
            print(f"Train batches: {len(train)}, Valid batches: {len(valid)}")
            
            # Safety check before concatenation
            if len(train) == 0:
                raise ValueError("No training data available after splitting. Check data and parameters.")
            if len(valid) == 0:
                raise ValueError("No validation data available after splitting. Check data and parameters.")

            train = pd.concat(train)
            valid = pd.concat(valid)
            print(f"Final train shape: {train.shape}, valid shape: {valid.shape}")

            test = test[test.ticker.isin(tickers)]
            
        else:
            trainvalid = df.loc[df.index < test_boundary]
            
            # For very short time series, use time-based splitting
            unique_timestamps = trainvalid.index.drop_duplicates().sort_values()
            total_points = len(unique_timestamps)
            
            print(f"Total data points for train+valid: {total_points}")
            
            # Ensure minimum data points for each split
            min_train_points = max(10, self.total_time_steps)  # At least 10 points or total_time_steps
            min_valid_points = max(5, self.total_time_steps // 2)  # At least 5 points or half total_time_steps
            
            if total_points < (min_train_points + min_valid_points):
                print(f"WARNING: Only {total_points} points available, need at least {min_train_points + min_valid_points}")
                # Adjust minimums for very small datasets
                min_train_points = max(2, total_points // 2)
                min_valid_points = max(1, total_points - min_train_points)
                print(f"Adjusted minimums: train={min_train_points}, valid={min_valid_points}")
            
            # Calculate split ensuring minimums
            split_index = max(min_train_points, int(train_valid_ratio * total_points))
            split_index = min(split_index, total_points - min_valid_points)
            
            if split_index > 0 and split_index < total_points:
                train_end_timestamp = unique_timestamps[split_index - 1]
                valid_start_timestamp = unique_timestamps[split_index]
            else:
                # Emergency fallback for very small datasets
                mid_point = total_points // 2
                train_end_timestamp = unique_timestamps[mid_point - 1] if mid_point > 0 else unique_timestamps[0]
                valid_start_timestamp = unique_timestamps[mid_point] if mid_point < total_points else unique_timestamps[-1]
            
            print(f"Train period: {trainvalid.index.min()} to {train_end_timestamp}")
            print(f"Valid period: {valid_start_timestamp} to {trainvalid.index.max()}")
            print(f"Split at index {split_index} of {total_points} timestamps")

            # Split using precise timestamp comparison
            train = trainvalid.loc[trainvalid.index <= train_end_timestamp].copy()
            valid = trainvalid.loc[trainvalid.index > train_end_timestamp].copy()  # Use > to avoid overlap
            
            print(f"Train shape: {train.shape}, Valid shape: {valid.shape}")
            
            # For short time series, be more lenient with ticker requirements
            if lags:
                # Reduce requirements for very short time series
                min_seq_length = min(self.total_time_steps, max(3, total_points // 4))
                tickers = (
                    valid.groupby("ticker")["ticker"].count() >= min_seq_length
                )
                tickers = tickers[tickers].index.tolist()
                train = train[train.ticker.isin(tickers)]
            else:
                # For short time series, just require some data in both splits
                train_ticker_counts = train.groupby("ticker").size()
                valid_ticker_counts = valid.groupby("ticker").size()
                
                # Very lenient requirements for short time series
                min_points_per_ticker = max(1, total_points // 20)  # At least 5% of data
                
                valid_train_tickers = train_ticker_counts[train_ticker_counts >= min_points_per_ticker].index
                valid_valid_tickers = valid_ticker_counts[valid_ticker_counts >= min_points_per_ticker].index
                
                tickers = list(set(valid_train_tickers).intersection(set(valid_valid_tickers)))
                
                # If no intersection, just use tickers with most data
                if len(tickers) == 0:
                    all_ticker_counts = df.groupby("ticker").size().sort_values(ascending=False)
                    tickers = all_ticker_counts.head(min(5, len(all_ticker_counts))).index.tolist()
                    print(f"No common tickers found, using top {len(tickers)} tickers by data volume")
                
            print(f"Selected tickers: {len(tickers)} - {tickers}")
            
            if len(tickers) == 0:
                raise ValueError("No suitable tickers found for short time series data")
                
            train = train[train.ticker.isin(tickers)]
            valid = valid[valid.ticker.isin(tickers)]
            test = test[test.ticker.isin(tickers)]
            
            # Final check for minimum data
            if len(train) == 0:
                raise ValueError("No training data after ticker filtering")
            if len(valid) == 0:
                print("WARNING: No validation data, using last 20% of training data for validation")
                # Emergency: split training data
                train_size = int(0.8 * len(train))
                valid = train.iloc[train_size:].copy()
                train = train.iloc[:train_size].copy()

        # Filter test data by end date if specified
        if test_end:
            test = test.loc[test.index < test_end]
            print(f"Test data after end date filter: {test.shape}")

        # Create test_with_buffer using datetime filtering
        test_with_buffer_list = []
        for t in tickers:
            ticker_trainvalid = trainvalid[trainvalid.ticker == t]
            ticker_test = test[test.ticker == t]
            
            if len(ticker_trainvalid) >= self.total_time_steps - 1:
                buffer_data = ticker_trainvalid.iloc[-(self.total_time_steps - 1):]
                combined = pd.concat([buffer_data, ticker_test]).sort_index()
                test_with_buffer_list.append(combined)
            elif len(ticker_test) > 0:
                # If not enough buffer data, just use test data
                test_with_buffer_list.append(ticker_test)
        
        if test_with_buffer_list:
            test_with_buffer = pd.concat(test_with_buffer_list)
        else:
            test_with_buffer = pd.DataFrame()  # Empty dataframe as fallback
        
        print(f"Test with buffer shape: {test_with_buffer.shape}")

        # Handle case where fixed window did not have a full sequence
        if lags:
            additional_test_data = []
            for t in tickers:
                test_ticker = test[test["ticker"] == t]
                diff = len(test_ticker) - self.total_time_steps
                if diff < 0:
                    # Need more data from trainvalid
                    additional_data = trainvalid[trainvalid["ticker"] == t].iloc[diff:]
                    additional_test_data.append(additional_data)
            
            if additional_test_data:
                test = pd.concat([pd.concat(additional_test_data), test]).sort_index()

        self.tickers = tickers
        self.num_tickers = len(tickers)
        
        print(f"Final tickers: {len(self.tickers)}")
        print(f"Setting scalers with train data shape: {train.shape}")
        
        if len(train) == 0:
            raise ValueError("No training data available for scaling")
            
        self.set_scalers(train)

        train, valid, test, test_with_buffer = [
            self.transform_inputs(data)
            for data in [train, valid, test, test_with_buffer]
        ]

        if lags:
            self.train = self._batch_data_smaller_output(
                train, train_valid_sliding, self.lags
            )
            self.valid = self._batch_data_smaller_output(
                valid, train_valid_sliding, self.lags
            )
            self.test_fixed = self._batch_data_smaller_output(test, False, self.lags)
            self.test_sliding = self._batch_data_smaller_output(
                test_with_buffer, True, self.lags
            )
        else:
            self.train = self._batch_data(train, train_valid_sliding)
            self.valid = self._batch_data(valid, train_valid_sliding)
            self.test_fixed = self._batch_data(test, False)
            self.test_sliding = self._batch_data(test_with_buffer, True)
            

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.
        Args:
          df: Data to use to calibrate scalers.
        """
        column_definitions = self.get_column_definition()
        id_column = get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = get_single_col_by_input_type(
            InputTypes.TARGET, column_definitions
        )

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values
        )  # used for predictions

        # Format categorical scalers
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values
            )
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations.
        This includes both feature engineering, preprocessing and normalisation.
        Args:
          df: Data frame to transform.
        Returns:
          Transformed data frame.
        """
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        column_definitions = self.get_column_definition()

        real_inputs = extract_cols_from_data_type(
            DataTypes.REAL_VALUED,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )
        categorical_inputs = extract_cols_from_data_type(
            DataTypes.CATEGORICAL,
            column_definitions,
            {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
        )

        # Format real inputs
        if self.transform_real_inputs:
            output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.
        Args:
          predictions: Dataframe of model predictions.
        Returns:
          Data frame of unnormalised predictions.
        """
        output = predictions.copy()

        column_names = predictions.columns

        if self.transform_real_inputs:
            for col in column_names:
                if col not in {"forecast_time", "identifier"}:
                    output[col] = self._target_scaler.inverse_transform(
                        predictions[col]
                    )
        else:
            categorical_inputs = extract_cols_from_data_type(
                DataTypes.CATEGORICAL,
                self.get_column_definition(),
                {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET},
            )
            for col in column_names:
                if col in categorical_inputs:
                    output[col] = self._target_scaler.inverse_transform(
                        predictions[col]
                    )

        return output

    def get_column_definition(self):
        """ "Returns formatted column definition in order expected."""

        column_definition = self._column_definition

        # Sanity checks first.
        # Ensure only one ID and time column exist
        def _check_single_column(input_type):

            length = len([tup for tup in column_definition if tup[2] == input_type])

            if length != 1:
                raise ValueError(
                    "Illegal number of inputs ({}) of type {}".format(
                        length, input_type
                    )
                )

        _check_single_column(InputTypes.ID)
        _check_single_column(InputTypes.TIME)

        identifier = [tup for tup in column_definition if tup[2] == InputTypes.ID]
        time = [tup for tup in column_definition if tup[2] == InputTypes.TIME]
        real_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.REAL_VALUED
            and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]
        categorical_inputs = [
            tup
            for tup in column_definition
            if tup[1] == DataTypes.CATEGORICAL
            and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]
        target = [tup for tup in column_definition if tup[2] == InputTypes.TARGET]

        return identifier + time + real_inputs + categorical_inputs + target

    @staticmethod
    def _unpack(data):
        return (
            data["inputs"],
            data["outputs"],
            data["active_entries"],
            data["identifier"],
            data["date"],
        )

    def _batch_data(self, data, sliding_window):
        """Batches data for training.

        Converts raw dataframe from a 2-D tabular format to a batched 3-D array
        to feed into Keras model.

        Args:
          data: DataFrame to batch

        Returns:
          Batched Numpy array with shape=(?, self.time_steps, self.input_size)
        """
        # TODO this works but is a bit of a mess
        data = data.copy()
        data["date"] = data.index.strftime("%Y-%m-%d")

        id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        time_col = get_single_col_by_input_type(
            InputTypes.TIME, self._column_definition
        )
        target_col = get_single_col_by_input_type(
            InputTypes.TARGET, self._column_definition
        )

        input_cols = [
            tup[0]
            for tup in self._column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        data_map = {}

        if sliding_window:
            # Functions.
            def _batch_single_entity(input_data):
                time_steps = len(input_data)
                lags = self.total_time_steps  # + int(self.extra_lookahead_steps)
                x = input_data.values
                if time_steps >= lags:
                    return np.stack(
                        [x[i : time_steps - (lags - 1) + i, :] for i in range(lags)],
                        axis=1,
                    )
                else:
                    return None

            for _, sliced in data.groupby(id_col):

                col_mappings = {
                    "identifier": [id_col],
                    "date": [time_col],
                    "outputs": [target_col],
                    "inputs": input_cols,
                }

                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = _batch_single_entity(sliced[cols].copy())

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

            # Combine all data
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)

            active_entries = np.ones_like(data_map["outputs"])
            if "active_entries" not in data_map:
                data_map["active_entries"] = active_entries
            else:
                data_map["active_entries"].append(active_entries)

        else:
            for _, sliced in data.groupby(id_col):

                col_mappings = {
                    "identifier": [id_col],
                    "date": [time_col],
                    "inputs": input_cols,
                    "outputs": [target_col],
                }

                time_steps = len(sliced)
                lags = self.total_time_steps
                additional_time_steps_required = lags - (time_steps % lags)

                def _batch_single_entity(input_data):
                    x = input_data.values
                    if additional_time_steps_required > 0:
                        x = np.concatenate(
                            [x, np.zeros((additional_time_steps_required, x.shape[1]))]
                        )
                    return x.reshape(-1, lags, x.shape[1])

                # for k in col_mappings:
                k = "outputs"
                cols = col_mappings[k]
                arr = _batch_single_entity(sliced[cols].copy())

                batch_size = arr.shape[0]
                sequence_lengths = [
                    (
                        lags
                        if i != batch_size - 1
                        else lags - additional_time_steps_required
                    )
                    for i in range(batch_size)
                ]
                active_entries = np.ones((arr.shape[0], arr.shape[1], arr.shape[2]))
                for i in range(batch_size):
                    active_entries[i, sequence_lengths[i] :, :] = 0
                sequence_lengths = np.array(sequence_lengths, dtype=np.int64)

                if "active_entries" not in data_map:
                    data_map["active_entries"] = [
                        active_entries[sequence_lengths > 0, :, :]
                    ]
                else:
                    data_map["active_entries"].append(
                        active_entries[sequence_lengths > 0, :, :]
                    )

                if k not in data_map:
                    data_map[k] = [arr[sequence_lengths > 0, :, :]]
                else:
                    data_map[k].append(arr[sequence_lengths > 0, :, :])

                for k in set(col_mappings) - {"outputs"}:
                    cols = col_mappings[k]
                    arr = _batch_single_entity(sliced[cols].copy())

                    if k not in data_map:
                        data_map[k] = [arr[sequence_lengths > 0, :, :]]
                    else:
                        data_map[k].append(arr[sequence_lengths > 0, :, :])

            # Combine all data
            for k in data_map:
                data_map[k] = np.concatenate(data_map[k], axis=0)

        active_flags = (np.sum(data_map["active_entries"], axis=-1) > 0.0) * 1.0
        data_map["inputs"] = data_map["inputs"][: len(active_flags)]
        data_map["outputs"] = data_map["outputs"][: len(active_flags)]
        data_map["active_entries"] = active_flags
        data_map["identifier"] = data_map["identifier"][: len(active_flags)]
        data_map["identifier"][data_map["identifier"] == 0] = ""
        data_map["date"] = data_map["date"][: len(active_flags)]
        data_map["date"][data_map["date"] == 0] = ""
        return data_map

    def _batch_data_smaller_output(self, data, sliding_window, output_length):
        """Batches data for training.

        Converts raw dataframe from a 2-D tabular format to a batched 3-D array
        to feed into Keras model.

        Args:
          data: DataFrame to batch

        Returns:
          Batched Numpy array with shape=(?, self.time_steps, self.input_size)
        """
        # TODO this works but is a bit of a mess
        data = data.copy()
        data["date"] = data.index.strftime("%Y-%m-%d")

        id_col = get_single_col_by_input_type(InputTypes.ID, self._column_definition)
        time_col = get_single_col_by_input_type(
            InputTypes.TIME, self._column_definition
        )
        target_col = get_single_col_by_input_type(
            InputTypes.TARGET, self._column_definition
        )

        input_cols = [
            tup[0]
            for tup in self._column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        data_map = {}

        col_mappings = {
            "identifier": [id_col],
            "date": [time_col],
            "inputs": input_cols,
            "outputs": [target_col],
        }

        if sliding_window:
            for _, sliced in data.groupby(id_col):

                time_steps = len(sliced)
                batch_size = time_steps - self.total_time_steps + 1
                seq_len = self.total_time_steps
                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = sliced[cols].copy().values
                    arr = np.concatenate(
                        [arr[start : start + seq_len] for start in range(0, batch_size)]
                    ).reshape(-1, seq_len, arr.shape[1])

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

        else:
            for _, sliced in data.groupby(id_col):

                time_steps = len(sliced)
                batch_size = (
                    time_steps - self.total_time_steps + output_length
                ) // output_length
                active_time_steps = batch_size * output_length + (
                    self.total_time_steps - output_length
                )
                disregard_time_steps = time_steps % active_time_steps
                seq_len = self.total_time_steps
                for k in col_mappings:
                    cols = col_mappings[k]
                    arr = sliced[cols].copy().values[disregard_time_steps:]
                    arr = np.concatenate(
                        [
                            arr[start : start + seq_len]
                            for start in range(
                                0, output_length * batch_size, output_length
                            )
                        ]
                    ).reshape(-1, seq_len, arr.shape[1])

                    if k not in data_map:
                        data_map[k] = [arr]
                    else:
                        data_map[k].append(arr)

        # Combine all data
        for k in data_map:
            data_map[k] = np.concatenate(data_map[k], axis=0)

        data_map["active_entries"] = (np.sum(data_map["inputs"], axis=-1) > 0.0) * 1.0
        # TODO
        data_map["identifier"][data_map["identifier"] == 0] = ""
        data_map["date"][data_map["date"] == 0] = ""

        data_map["inputs_identifier"] = data_map["identifier"].copy()
        data_map["identifier"] = data_map["identifier"][:, -output_length:, :]

        data_map["inputs_date"] = data_map["date"].copy()
        data_map["date"] = data_map["date"][:, -output_length:, :]

        data_map["outputs"] = data_map["outputs"][:, -output_length:, :]

        return data_map

    def _get_input_columns(self):
        """Returns names of all input columns."""
        return [
            tup[0]
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

    @property
    def num_classes_per_cat_input(self):
        """Returns number of categories per relevant input.

        This is seqeuently required for keras embedding layers.
        """
        return self._num_classes_per_cat_input

    @property
    def input_params(self):
        """Returns the relevant indexes and input sizes."""

        # Functions
        def _extract_tuples_from_data_type(data_type, defn):
            return [
                tup
                for tup in defn
                if tup[1] == data_type
                and tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
            ]

        def _get_locations(input_types, defn):
            return [i for i, tup in enumerate(defn) if tup[2] in input_types]

        # Start extraction
        column_definition = [
            tup
            for tup in self.get_column_definition()
            if tup[2] not in {InputTypes.ID, InputTypes.TIME, InputTypes.TARGET}
        ]

        categorical_inputs = _extract_tuples_from_data_type(
            DataTypes.CATEGORICAL, column_definition
        )
        real_inputs = _extract_tuples_from_data_type(
            DataTypes.REAL_VALUED, column_definition
        )

        input_size = len(self._get_input_columns())

        # TODO artefact of previous code - need to clean up
        locations = {
            # "lstm_input_size": input_size,
            "input_size": input_size,
            "output_size": len(
                _get_locations({InputTypes.TARGET}, self._column_definition)
            ),
            "category_counts": self.num_classes_per_cat_input,
            # "input_obs_loc": _get_locations({InputTypes.TARGET}, column_definition),
            "static_input_loc": _get_locations(
                {InputTypes.STATIC_INPUT}, column_definition
            ),
            "known_regular_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, real_inputs
            ),
            "known_categorical_inputs": _get_locations(
                {InputTypes.STATIC_INPUT, InputTypes.KNOWN_INPUT}, categorical_inputs
            ),
        }

        return locations
