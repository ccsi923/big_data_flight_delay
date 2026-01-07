"""
Flight Delay Prediction Spark Application

This application:
1. Loads raw flight and plane data.
2. Loads a pre-trained regression model (Best Model).
3. Preprocesses the data using saved metadata (encoding/imputing) to match training conditions.
4. Generates predictions for flight delays.
5. Evaluates model performance (RMSE, MAE, R2) and saves predictions.
   (Automatically drops Vector columns to allow CSV saving).

Usage:
    spark-submit app.py --raw_flights ./data/2008.csv.bz2 --raw_plane ./data/plane-data.csv --out ./output/predictions.csv
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from math import cos, pi, sin

# Spark Imports
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import VectorUDT
from pyspark.sql import DataFrame, SparkSession, functions as F
from pyspark.sql.functions import col, lit, udf, when
from pyspark.sql.types import StringType

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# --- Configuration Constants ---
DEFAULT_MODEL_PATH = "best_model/retrained_best_model"
DEFAULT_PROCESSING_PARAMS_DIR = "best_model/processing/"
TEMP_DIR = "./temp_app/"


class DataLoader:
    """
    Handles reading raw CSV data, schema management, and intermediate Parquet storage.
    """
    def __init__(self, spark: SparkSession, temp_dir: str):
        self.spark = spark
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def load_csv_and_cache(self, raw_path: str, name: str) -> DataFrame:
        """
        Reads a CSV, infers schema, saves to Parquet for performance, and reloads.
        """
        try:
            parquet_path = os.path.join(self.temp_dir, f"{name}.parquet")
            schema_path = os.path.join(self.temp_dir, f"{name}_schema.json")

            logger.info(f"Reading raw data from: {raw_path}")
            df = self.spark.read.csv(
                raw_path,
                header=True,
                inferSchema=True,
                nullValue="NA"
            )

            # Save schema for consistency
            with open(schema_path, 'w') as f:
                f.write(df.schema.json())

            # Write to parquet to standardize types and speed up downstream ops
            df.write.mode("overwrite").parquet(parquet_path)

            # Reload with strict schema
            df_parquet = self.spark.read.schema(df.schema).parquet(parquet_path)
            return df_parquet

        except Exception as e:
            logger.error(f"Failed to load data from {raw_path}: {e}")
            raise


class Preprocessor:
    """
    Encapsulates all data cleaning, feature engineering, and transformation logic.
    Ensures that test data is processed exactly like the training data.
    """
    def __init__(self, params_dir: str):
        self.params_dir = params_dir
        self.cyclic_ordinal_time = ['Month', 'DayofMonth', 'DayOfWeek']
        self.quant_time_features = ['DepTime', 'CRSDepTime', 'CRSArrTime']
        self.quantitative_features = ['CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut']
        self.nominal_features = [
            'UniqueCarrier', 'FlightNum', 'TailNum', 'Origin', 'Dest', 'Cancelled',
            'CancellationCode', 'EngineType', 'AircraftType', 'Manufacturer', 'Model',
            "issue_date", "status", "type"
        ]
        self.non_cyclic_ordinal_time = ['Year', 'PlaneIssueYear']

    def _cast_columns_safely(self, df: DataFrame, columns: list, target_type="int") -> DataFrame:
        """Safely casts columns to avoid runtime errors with 'NA' strings."""
        for col_name in columns:
            df = df.withColumn(col_name, F.expr(f"try_cast({col_name} as {target_type})"))
        return df

    def _polar_time_encode(self, df: DataFrame) -> DataFrame:
        """Transforms cyclic time features into polar coordinates (sin/cos)."""
        logger.info("Applying polar encoding to time features...")

        def polar_calc(value, max_val):
            if value is None: return None, None
            angle = (value / max_val) * 2 * pi
            return float(cos(angle)), float(sin(angle))

        polar_udf = udf(polar_calc, "struct<cos:double, sin:double>")

        df = df.withColumn("Month_polar", polar_udf(col("Month"), lit(12))) \
            .withColumn("DayofMonth_polar", polar_udf(col("DayofMonth"),
                                                      when(col("Month") == 2, lit(28))
                                                      .when(col("Month").isin([4, 6, 9, 11]), lit(30))
                                                      .otherwise(lit(31)))) \
            .withColumn("DayOfWeek_polar", polar_udf(col("DayOfWeek"), lit(7)))

        # Unpack struct columns
        for base in ["Month", "DayofMonth", "DayOfWeek"]:
            df = df.withColumn(f"{base}_cos", col(f"{base}_polar.cos")) \
                   .withColumn(f"{base}_sin", col(f"{base}_polar.sin"))

        return df.drop("Month_polar", "DayofMonth_polar", "DayOfWeek_polar")

    def static_transform(self, df_flights: DataFrame, df_planes: DataFrame) -> (DataFrame, list, list, list):
        """
        Performs stateless transformations: joins, cleaning, casting, feature engineering.
        """
        logger.info("Starting static preprocessing...")

        # 1. Standardization and Join
        df_planes = df_planes.withColumnRenamed("tailnum", "TailNum")
        df = df_flights.join(df_planes, on="TailNum", how="inner")

        # 2. Filter Cancelled
        df = df.withColumn("Cancelled", col("Cancelled").cast("string"))
        df = df.filter(col("Cancelled") != "1")

        # 3. Drop forbidden leakage columns and useless features
        forbidden = [
            "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
            "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"
        ]
        useless = ["TailNum", "FlightNum", "UniqueCarrier", "CancellationCode", "Cancelled",
                   "issue_date", "status", "type"]

        df = df.drop(*(forbidden + useless))

        # 4. Renaming
        rename_map = {
            "year": "PlaneIssueYear", "engine_type": "EngineType",
            "aircraft_type": "AircraftType", "model": "Model", "manufacturer": "Manufacturer"
        }
        for old, new in rename_map.items():
            df = df.withColumnRenamed(old, new)

        # 5. Safe Casting & Null Handling
        target_col = "ArrDelay"
        cols_to_cast = self.quantitative_features + [target_col]

        logger.info("Sanitizing quantitative columns...")
        df = self._cast_columns_safely(df, cols_to_cast, "int")
        df = df.dropna(subset=[target_col])

        # 6. Time Column Processing
        logger.info("Processing time columns...")
        # Handle 'NA' strings in time features
        for t_col in self.quant_time_features:
            df = df.withColumn(t_col, when(F.trim(col(t_col)) == "NA", None).otherwise(col(t_col)))

            # Convert HHMM string to minutes
            df = df.withColumn(
                f"{t_col}_minutes",
                (F.expr(f"try_cast(substring({t_col}, 1, 2) as int)") * 60 +
                 F.expr(f"try_cast(substring({t_col}, 3, 2) as int)"))
            ).fillna(0, subset=[f"{t_col}_minutes"])

            self.quantitative_features.append(f"{t_col}_minutes")

        df = df.drop(*self.quant_time_features)

        # 7. Polar Encoding
        df = self._polar_time_encode(df)
        ordinal_feats = [f"{f}_{s}" for f in self.cyclic_ordinal_time for s in ['sin', 'cos']]

        # Define nominals
        final_nominal = ["Origin", "Dest", "EngineType", "AircraftType", "Manufacturer", "Model"]
        all_nominals = final_nominal + self.non_cyclic_ordinal_time

        # Force all nominal columns to StringType
        logger.info("Casting nominal features to String...")
        for col_name in all_nominals:
            if col_name in df.columns:
                df = df.withColumn(col_name, col(col_name).cast("string"))

        return df, self.quantitative_features, ordinal_feats, all_nominals

    def dynamic_transform(self, df: DataFrame, nominal: list, ordinal: list, quantitative: list) -> DataFrame:
        """
        Applies stateful transformations (Imputing, Encoding, Vectorizing) using saved metadata.
        """
        logger.info("Applying dynamic preprocessing using saved parameters...")
        spark = SparkSession.builder.getOrCreate()

        # 1. Imputation
        imputer_path = os.path.join(self.params_dir, 'imputer_maps.json')
        if os.path.exists(imputer_path):
            with open(imputer_path, 'r') as f:
                imputer_maps = json.load(f)

            for fea in quantitative + ordinal + nominal:
                if fea not in df.columns: continue
                if fea in imputer_maps:
                    fill_val = imputer_maps[fea]['fill_value']
                    extra_nulls = imputer_maps[fea]['extra_nulls']

                    if extra_nulls:
                        # --- FIX: Type-safe Null Checking ---
                        # Prevent comparing Double columns (sin/cos) with String 'None'
                        is_string = isinstance(df.schema[fea].dataType, StringType)
                        if is_string:
                             df = df.withColumn(fea, when(col(fea).isin(extra_nulls), lit(None)).otherwise(col(fea)))
                        else:
                             # For Numeric columns, filter out strings like "None" before check
                             valid_numeric_nulls = [x for x in extra_nulls if isinstance(x, (int, float))]
                             if valid_numeric_nulls:
                                 df = df.withColumn(fea, when(col(fea).isin(valid_numeric_nulls), lit(None)).otherwise(col(fea)))

                    df = df.fillna(fill_val, subset=[fea])

        # 2. Nominal Encoding
        encode_meta_path = os.path.join(self.params_dir, 'encode_types.json')
        freq_meta_path = os.path.join(self.params_dir, 'non_aggregated.json')

        if os.path.exists(encode_meta_path) and os.path.exists(freq_meta_path):
            with open(encode_meta_path, 'r') as f: encode_types = json.load(f)
            with open(freq_meta_path, 'r') as f: fea_freqs = json.load(f)

            # Aggregate rare categories
            # --- FIX: Match Notebook Suffix '_agg' instead of '_aggregated' ---
            for fea, valid_cats in fea_freqs.items():
                if fea in df.columns:
                    df = df.withColumn(f"{fea}_agg",
                                       when(col(fea).isin(valid_cats), col(fea)).otherwise(lit("Other")))

            # Apply Encoders
            for agg_fea, method in encode_types.items():
                # --- FIX: Match Notebook Key Generation ---
                # Key is "Year_agg", so replace "_agg" to get "Year"
                original_fea = agg_fea.replace("_agg", "")

                if method == 'binary':
                    # --- FIX: Match Notebook File Name f'{fea}_encoder' ---
                    path = os.path.join(self.params_dir, f'{original_fea}_encoder')
                    if os.path.exists(path):
                        model = PipelineModel.load(path)
                        df = model.transform(df)
                    else:
                        logger.warning(f"Binary encoder not found at {path}")

                elif method == 'mean':
                    # --- FIX: Match Notebook File Name f'{fea}_mean_map.csv' ---
                    path = os.path.join(self.params_dir, f'{original_fea}_mean_map.csv')
                    if os.path.exists(path):
                        mapping = spark.read.csv(path, header=True, inferSchema=True)

                        # Rename "mean_enc" column to avoid ambiguity
                        mean_col_name = f"{original_fea}_mean_enc"
                        if "mean_enc" in mapping.columns:
                            mapping = mapping.withColumnRenamed("mean_enc", mean_col_name)

                        # Join on the aggregated column (e.g., Year_agg)
                        df = df.join(mapping, on=agg_fea, how='left')

                        # Fill unknown categories with 0 (Global Mean approximation)
                        df = df.fillna(0.0, subset=[mean_col_name])
                    else:
                        logger.warning(f"Mean map not found at {path}")

        # 3. Vector Assembler
        cols_path = os.path.join(self.params_dir, 'feature_columns.json')
        if os.path.exists(cols_path):
            with open(cols_path, 'r') as f:
                feature_cols = json.load(f)

            existing_cols = [c for c in feature_cols if c in df.columns]

            # --- DEBUG: Log missing features if any ---
            if len(existing_cols) < len(feature_cols):
                missing = set(feature_cols) - set(existing_cols)
                logger.error(f"WARNING: The following features are MISSING from the dataset: {missing}")
                logger.error("This will likely cause a Vector Size Mismatch error in the model.")

            logger.info(f"Assembling {len(existing_cols)} features into vector...")

            assembler = VectorAssembler(inputCols=existing_cols, outputCol="features", handleInvalid="skip")
            df = assembler.transform(df)

        return df


class ModelHandler:
    """
    Manages loading the pre-trained model and performing predictions/evaluation.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict_and_evaluate(self, df: DataFrame, output_csv_path: str):
        """
        Loads model, transforms data, calculates metrics, and saves predictions.
        AUTOMATICALLY DROPS VECTOR COLUMNS to allow saving to CSV.
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model path does not exist: {self.model_path}")
            return

        logger.info(f"Loading model from {self.model_path}")
        model = PipelineModel.load(self.model_path)

        logger.info("Generating predictions...")
        predictions = model.transform(df)

        # Evaluation
        logger.info("Evaluating performance...")
        metrics = {}
        for metric in ["rmse", "mae", "r2"]:
            evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction", metricName=metric)
            value = evaluator.evaluate(predictions)
            metrics[metric] = value
            logger.info(f"Model Performance - {metric.upper()}: {value}")

        # Saving Results
        output_dir = os.path.dirname(output_csv_path)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving predictions to {output_csv_path}")

        # Identify and remove Vector columns
        cols_to_save = []
        for field in predictions.schema.fields:
            if isinstance(field.dataType, VectorUDT):
                continue
            if field.name in ["features", "scaledFeatures"]:
                continue
            cols_to_save.append(field.name)

        if "prediction" in cols_to_save:
            cols_to_save.remove("prediction")
            cols_to_save.append("prediction")

        predictions.select(cols_to_save).write.mode("overwrite").option("header", "true").csv(output_csv_path)
        logger.info("Process completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Spark Flight Delay Prediction")
    parser.add_argument("--raw_flights", type=str, default="./data/2003.csv.bz2", help="Path to raw flight CSV/BZ2")
    parser.add_argument("--raw_plane", type=str, default="./training_data/flight_data/plane-data.csv", help="Path to raw plane data CSV")
    parser.add_argument("--out", type=str, default="./output/predictions.csv", help="Output path for prediction CSV")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model directory")
    parser.add_argument("--params_dir", type=str, default=DEFAULT_PROCESSING_PARAMS_DIR, help="Path to preprocessing metadata")

    args = parser.parse_args()

    try:
        # Initialize Spark
        spark = SparkSession.builder \
            .appName("FlightDelayPredictor") \
            .config("spark.sql.caseSensitive", "true") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")

        # 1. Load Data
        loader = DataLoader(spark, TEMP_DIR)
        df_planes = loader.load_csv_and_cache(args.raw_plane, "planes")
        df_flights = loader.load_csv_and_cache(args.raw_flights, "flights")

        # Repartition
        df_planes = df_planes.repartition(10)
        df_flights = df_flights.repartition(20)

        # 2. Preprocess Data
        preprocessor = Preprocessor(args.params_dir)
        df_clean, quant, ordinal, nominal = preprocessor.static_transform(df_flights, df_planes)
        df_final = preprocessor.dynamic_transform(df_clean, nominal, ordinal, quant)

        # 3. Model Prediction & Evaluation
        model_handler = ModelHandler(args.model_path)
        model_handler.predict_and_evaluate(df_final, args.out)

    except Exception as e:
        logger.error("Critical failure in application execution.")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup temp directory
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()