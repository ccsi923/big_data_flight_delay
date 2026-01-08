import argparse
import json
import logging
import sys
import os
import shutil
from math import pi

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

# --- Configuration & Renaming ---
EXCLUDED_COLS = [
    "ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted",
    "CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"
]

IGNORED_COLS = [
    "TailNum", "FlightNum", "UniqueCarrier", "CancellationCode", "Cancelled",
    "issue_date", "status", "type"
]

COLUMN_TRANSLATION = {
    "year": "PlaneIssueYear",
    "engine_type": "EngineType",
    "aircraft_type": "AircraftType",
    "model": "Model",
    "manufacturer": "Manufacturer"
}

# Setup Logging
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s', level=logging.INFO)
log = logging.getLogger("InferencePipeline")


def init_spark() -> SparkSession:
    """Initializes a robust Spark Session."""
    return SparkSession.builder \
        .appName("DelayPredictionSystem") \
        .config("spark.sql.caseSensitive", "true") \
        .config("spark.sql.ansi.enabled", "false") \
        .getOrCreate()


def fetch_and_convert_data(spark: SparkSession, source_path: str, staging_dir: str, alias: str) -> DataFrame:
    """
    Ingests raw CSV data and caches it as Parquet.
    FIX: Now robustly handles corrupt/empty Parquet caches by forcing a reload.
    """
    os.makedirs(staging_dir, exist_ok=True)
    parquet_target = os.path.join(staging_dir, f"{alias}.parquet")

    def ingest_raw():
        """Internal helper to read CSV and write Parquet."""
        log.info(f"Staging raw data from {source_path}...")

        # Check if source exists (Python check) to avoid Spark 'Path does not exist' crashes
        # Note: wildcard expansion handled by Spark, but we check directory if possible
        if not source_path.startswith("s3") and "*" not in source_path and not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        raw_reader = spark.read.option("header", "true") \
            .option("inferSchema", "true") \
            .option("nullValue", "NA") \
            .option("emptyValue", None)

        df = raw_reader.csv(source_path)

        # Guard: If DF is empty, don't write bad parquet
        if len(df.head(1)) == 0:
            raise ValueError(f"Data source {source_path} appears empty.")

        df.write.mode("overwrite").parquet(parquet_target)

    # 1. Try to read existing cache
    if os.path.exists(parquet_target):
        try:
            log.info(f"Loading staged data: {parquet_target}")
            df = spark.read.parquet(parquet_target)
            # Force a check to see if schema is valid
            if len(df.columns) == 0: raise Exception("Empty Schema")
            return df
        except Exception as e:
            log.warning(f"Cached data at {parquet_target} is corrupt or empty ({e}). Re-staging...")
            shutil.rmtree(parquet_target, ignore_errors=True)

    # 2. If not exists or corrupt, ingest
    ingest_raw()
    return spark.read.parquet(parquet_target)


def generate_cyclic_features(df: DataFrame) -> DataFrame:
    """
    Transforms Month, Day, and Week into geometric (sin/cos) features.
    """
    log.info("Applying geometric time transformations...")

    # 1. Convert Time to Radians
    df = df.withColumn("rad_month", (F.col("Month") / 12) * 2 * pi)
    df = df.withColumn("rad_dow", (F.col("DayOfWeek") / 7) * 2 * pi)

    # Logic for Days in Month
    days_in_month = (
        F.when(F.col("Month") == 2, 28)
        .when(F.col("Month").isin([4, 6, 9, 11]), 30)
        .otherwise(31)
    )
    df = df.withColumn("rad_dom", (F.col("DayofMonth") / days_in_month) * 2 * pi)

    # 2. Generate Sin/Cos pairs
    time_units = [("Month", "rad_month"), ("DayOfWeek", "rad_dow"), ("DayofMonth", "rad_dom")]

    for prefix, rad_col in time_units:
        df = df.withColumn(f"{prefix}_sin", F.sin(F.col(rad_col))) \
            .withColumn(f"{prefix}_cos", F.cos(F.col(rad_col)))

    return df.drop("rad_month", "rad_dow", "rad_dom")


def standardize_schema(flights: DataFrame, planes: DataFrame) -> DataFrame:
    """
    Executes deterministic schema cleaning.
    """
    # 1. Merge Datasets
    planes = planes.withColumnRenamed("tailnum", "TailNum")
    merged = flights.join(planes, on="TailNum", how="inner")

    # 2. Filter Invalid Rows
    merged = merged.filter(F.col("Cancelled").cast("string") != "1")

    # 3. Prune Columns
    merged = merged.drop(*(EXCLUDED_COLS + IGNORED_COLS))

    # 4. Apply Naming Convention
    for old, new in COLUMN_TRANSLATION.items():
        merged = merged.withColumnRenamed(old, new)

    return merged


# In app.py
def parse_time_columns(df: DataFrame) -> (DataFrame, list):
    time_cols_hhmm = ['DepTime', 'CRSDepTime', 'CRSArrTime']
    new_numeric_cols = ['CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut']

    for col_name in time_cols_hhmm:
        # Sanitize
        df = df.withColumn(col_name,
                           F.when((F.trim(F.col(col_name)) == "") | (F.trim(F.col(col_name)) == "NA"), None)
                           .otherwise(F.col(col_name).cast(IntegerType()))) # Cast to int

        # Mathematical Parse: HHMM -> (HH * 60) + MM (MATCHING NOTEBOOK)
        hours = F.floor(F.col(col_name) / 100)
        mins = F.col(col_name) % 100

        minutes_val = (hours * 60) + mins

        # Fill missing time with 0
        df = df.withColumn(f"{col_name}_minutes", F.coalesce(minutes_val, F.lit(0)))
        new_numeric_cols.append(f"{col_name}_minutes")

    df = df.drop(*time_cols_hhmm)
    return df, new_numeric_cols


def apply_training_state(df: DataFrame, artifacts_path: str) -> DataFrame:
    """
    Restores the 'Memory' of the training phase (Imputation Means, Encodings).
    """
    spark = SparkSession.getActiveSession()

    # --- Phase 1: Imputation ---
    impute_meta_path = os.path.join(artifacts_path, 'imputer_maps.json')
    if os.path.exists(impute_meta_path):
        with open(impute_meta_path, 'r') as f:
            impute_rules = json.load(f)

        log.info("Restoring imputation logic...")
        for col, rule in impute_rules.items():
            if col not in df.columns: continue

            bad_values = rule.get('extra_nulls', [])
            if bad_values:
                is_string = isinstance(df.schema[col].dataType, StringType)
                if is_string:
                    df = df.withColumn(col, F.when(F.col(col).isin(bad_values), None).otherwise(F.col(col)))
                else:
                    numeric_bad_vals = [x for x in bad_values if isinstance(x, (int, float))]
                    if numeric_bad_vals:
                        df = df.withColumn(col, F.when(F.col(col).isin(numeric_bad_vals), None).otherwise(F.col(col)))

            df = df.fillna(rule['fill_value'], subset=[col])

    # --- Phase 2: Categorical Encoding ---
    enc_types_path = os.path.join(artifacts_path, 'encode_types.json')
    groups_path = os.path.join(artifacts_path, 'non_aggregated.json')

    if os.path.exists(enc_types_path) and os.path.exists(groups_path):
        with open(enc_types_path, 'r') as f:
            enc_types = json.load(f)
        with open(groups_path, 'r') as f:
            valid_groups = json.load(f)

        log.info("Restoring categorical encoders...")

        # 1. Group Rare Categories
        for col, preserved_vals in valid_groups.items():
            if col in df.columns:
                df = df.withColumn(f"{col}_agg",
                                   F.when(F.col(col).isin(preserved_vals), F.col(col))
                                   .otherwise(F.lit("Other")))

        # 2. Apply Encoders
        for agg_key, method in enc_types.items():
            base_col = agg_key.replace("_agg", "")

            if method == 'binary':
                # File name follows training convention: {col}_aggregated_encoder
                model_path = os.path.join(artifacts_path, f'{base_col}_aggregated_encoder')
                if os.path.exists(model_path):
                    df = PipelineModel.load(model_path).transform(df)
                else:
                    log.warning(f"OHE Model missing for {base_col} at {model_path}")

            elif method == 'mean':
                # File name follows training convention: {col}_aggregated_encoder.csv
                map_path = os.path.join(artifacts_path, f'{base_col}_aggregated_encoder.csv')

                if os.path.exists(map_path):
                    mapping = spark.read.csv(map_path, header=True, inferSchema=True)

                    # Expected column name by Vectorizer
                    target_col = f"{base_col}_mean_enc"

                    # If training script saved header as "Origin_mean_enc", this rename is a no-op, which is fine.
                    if "mean_enc" in mapping.columns:
                        mapping = mapping.withColumnRenamed("mean_enc", target_col)

                    # Ensure join works by checking if agg_key exists
                    if agg_key in df.columns:
                        df = df.join(mapping, on=agg_key, how="left")
                        df = df.fillna(0.0, subset=[target_col])
                    else:
                        log.warning(f"Join key {agg_key} missing in DataFrame. Skipping mean encoding for {base_col}.")
                else:
                    log.warning(f"Mean mapping CSV missing for {base_col} at {map_path}")

    # --- Phase 3: Feature Assembly ---
    vec_path = os.path.join(artifacts_path, 'vectorizer')
    if os.path.exists(vec_path):
        log.info("Assembling feature vector...")
        vectorizer = PipelineModel.load(vec_path)
        df = vectorizer.transform(df)
    else:
        log.error("CRITICAL: Vectorizer model not found. Cannot proceed safely.")
        sys.exit(1)

    return df


def execute_pipeline(args):
    """Main execution flow."""
    spark = init_spark()

    try:
        # 1. Ingestion
        # Use args directly, but ensure absolute paths in main block
        df_flights = fetch_and_convert_data(spark, args.raw_flights, args.temp_dir, "flights_raw")
        df_planes = fetch_and_convert_data(spark, args.raw_plane, args.temp_dir, "planes_raw")

        df_flights = df_flights.repartition(args.partitions)

        # 2. Static Preprocessing
        df_clean = standardize_schema(df_flights, df_planes)
        df_clean, num_cols = parse_time_columns(df_clean)
        df_clean = generate_cyclic_features(df_clean)

        # Ensure numerics are cast safely
        target = "ArrDelay"
        for c in num_cols + [target]:
            df_clean = df_clean.withColumn(c, F.expr(f"try_cast({c} as int)"))

        df_clean = df_clean.dropna(subset=[target])

        # Ensure nominals are strings
        nominal_cols = ["Origin", "Dest", "EngineType", "AircraftType", "Manufacturer", "Model", "Year",
                        "PlaneIssueYear"]
        for n in nominal_cols:
            if n in df_clean.columns:
                df_clean = df_clean.withColumn(n, F.col(n).cast("string"))

        # 3. Dynamic Preprocessing (Applying learned artifacts)
        df_ready = apply_training_state(df_clean, args.params_dir)

        # 4. Inference
        log.info(f"Loading Model: {args.model_path}")
        model = PipelineModel.load(args.model_path)
        predictions = model.transform(df_ready)
        # --- NEW: Evaluation Block ---
        log.info("Evaluating model performance on new data...")
        evaluator = RegressionEvaluator(labelCol="ArrDelay", predictionCol="prediction")

        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

        log.info("=" * 30)
        log.info(f"  DATASET METRICS")
        log.info("=" * 30)
        log.info(f"  RMSE : {rmse:.4f}")
        log.info(f"  MAE  : {mae:.4f}")
        log.info(f"  R2   : {r2:.4f}")
        log.info("=" * 30)
        # -----------------------------

        # 5. Export
        log.info(f"Saving predictions to {args.out}")

        # Filter out complex Spark Vector columns that cannot be saved to CSV
        save_cols = [c for c in predictions.columns
                     if not (c.endswith("features")
                             or c.endswith("vector")
                             or c.endswith("_vec")  # <--- Added this to catch EngineType_vec
                             or c.endswith("_index"))]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

        predictions.select(save_cols) \
            .limit(1000) \
            .write.mode("overwrite") \
            .option("header", "true") \
            .csv(args.out)

        log.info("Process completed successfully.")

    except Exception as e:
        log.error("Pipeline crashed.", exc_info=True)
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(args.temp_dir):
            shutil.rmtree(args.temp_dir, ignore_errors=True)
        #spark.stop()


if __name__ == "__main__":
    # --- Resolve Paths Relatively to the Script Location ---
    # This prevents file-not-found errors when running from different directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir  # Assuming app.py is in root, otherwise adjust

    # Set robustness defaults based on your described changes
    default_flights = os.path.join(project_root, "data/2003.csv.bz2")
    default_planes = os.path.join(project_root, "training_data/flight_data/plane-data.csv")
    default_model = os.path.join(project_root, "best_model/_best_model")
    default_params = os.path.join(project_root, "best_model/encoders/")

    parser = argparse.ArgumentParser(description="Flight Delay Inference System")
    parser.add_argument("--raw_flights", default=default_flights)
    parser.add_argument("--raw_plane", default=default_planes)
    parser.add_argument("--out", default=os.path.join(project_root, "data/output_predictions.csv"))
    parser.add_argument("--model_path", default=default_model)
    parser.add_argument("--params_dir", default=default_params)
    parser.add_argument("--temp_dir", default=os.path.join(project_root, "runtime_cache/"))
    parser.add_argument("--partitions", type=int, default=10)

    args = parser.parse_args()

    # Debug info
    print(f"DEBUG: Reading flights from: {args.raw_flights}")

    execute_pipeline(args)