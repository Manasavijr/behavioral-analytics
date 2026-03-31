
import logging, time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)

def create_spark_session():
    return (SparkSession.builder
        .appName("BehavioralAnalytics")
        .config("spark.driver.memory", "6g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate())

def run_pipeline(input_path="data/raw/events", output_path="data/processed"):
    t0 = time.time()
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    logger.info("Ingesting data...")
    df = spark.read.parquet(input_path)
    total = df.count()
    logger.info(f"Loaded {total:,} rows")

    # Compute user-level aggregates separately via groupBy (memory efficient)
    logger.info("Computing user aggregates...")
    user_agg = df.groupBy("user_id").agg(
        F.sum("revenue").alias("user_total_revenue"),
        F.count("*").alias("user_event_count"),
        F.sum(F.when(F.col("event_type")=="purchase",1).otherwise(0)).alias("user_purchase_count"),
        F.avg("response_time_ms").alias("user_avg_response_ms"),
        F.min("timestamp").alias("user_first_event_ts"),
        F.max("timestamp").alias("user_last_event_ts"),
    )

    # Session-level aggregates via groupBy
    logger.info("Computing session aggregates...")
    session_agg = df.groupBy("session_id").agg(
        F.count("*").alias("session_depth"),
        F.min("timestamp").alias("session_start"),
        F.max("timestamp").alias("session_end"),
        F.max(F.when(F.col("event_type")=="purchase",1).otherwise(0)).alias("converted"),
        F.sum("revenue").alias("session_revenue"),
    )

    # Join back to main df
    logger.info("Joining features...")
    df = df.join(user_agg, on="user_id", how="left")
    df = df.join(session_agg, on="session_id", how="left")

    # Write processed events
    Path(f"{output_path}/events").mkdir(parents=True, exist_ok=True)
    logger.info("Writing processed events...")
    (df.write.mode("overwrite")
       .partitionBy("region", "device")
       .parquet(f"{output_path}/events"))
    logger.info("Processed events written")

    # Hourly aggregates
    logger.info("Computing hourly aggregates...")
    hourly = df.groupBy(
        F.date_trunc("hour", F.to_timestamp(F.col("timestamp"))).alias("hour_bucket"),
        "region", "device"
    ).agg(
        F.count("*").alias("event_count"),
        F.countDistinct("user_id").alias("unique_users"),
        F.countDistinct("session_id").alias("unique_sessions"),
        F.sum(F.when(F.col("event_type")=="purchase",1).otherwise(0)).alias("purchases"),
        F.sum("revenue").alias("total_revenue"),
        F.avg("response_time_ms").alias("avg_response_ms"),
        F.percentile_approx("response_time_ms", 0.95).alias("p95_response_ms"),
        F.sum(F.when(F.col("event_type")=="error",1).otherwise(0)).alias("error_count"),
        (F.sum(F.when(F.col("event_type")=="error",1).otherwise(0)) / F.count("*")).alias("error_rate"),
    ).orderBy("hour_bucket")

    hourly_pd = hourly.toPandas()
    Path(output_path).mkdir(parents=True, exist_ok=True)
    hourly_pd.to_csv(f"{output_path}/hourly_aggregates.csv", index=False)
    logger.info(f"Hourly aggregates: {len(hourly_pd):,} rows")

    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s — {total:,} rows processed")
    spark.stop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline()
