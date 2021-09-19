import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType 
from pyspark.sql.types import ArrayType, DoubleType, BooleanType
from pyspark.sql.functions import col,array_contains

CSV_FILE = "/mnt/nas/openImageNet/oidv6-train-annotations-bbox.csv"
spark = SparkSession.builder.appName('openImage_Annotation_Spark').getOrCreate()

df = spark.read.option("header",True).csv(CSV_FILE)
df.printSchema()
   
# df3 = spark.read.options(header='True', delimiter=',') \
#   .csv(CSV_FILE)
# df3.printSchema()