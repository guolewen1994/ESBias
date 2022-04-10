import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, DoubleType, TimestampType
from pyspark.sql.window import Window
import zipfile
import os
import datetime
from tempo import *
import sys

def zip_extract(path):
    """

    :param path: the path should contains the zipped daily files like HTICST120.20190401.1.zip
    :return: extracted_path
    """
    list_zip_files = os.listdir(path)
    extracted_path = os.path.join(os.path.dirname(path), os.path.basename(path) + '_extracted')
    print(extracted_path)
    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)
    # set working directory as extracted_path and put extracted files there
    os.chdir(extracted_path)
    assert os.getcwd() == extracted_path, "extracted directory should be the current working directory"
    for file in list_zip_files:
        file_path = os.path.join(path, file)
        try:
            print("{} is extracting".format(file))
            zipfile.ZipFile(file_path).extractall()
            print("{} is successfully extracted".format(file))
        except zipfile.BadZipfile:
            print("Bad CRC-32 for {}".format(file), "This file need to be checked")
    return extracted_path


def create_spark_session():
    spark = SparkSession.builder.master('local[12]') \
                                .config("spark.driver.memory", "100g") \
                                .config("spark.executor.memory", "100g") \
                                .config("spark.driver.maxResultSize", "50g") \
                                .config("spark.local.dir", "/home/lguo5/temp") \
                                .appName('DataClean') \
                                .getOrCreate()
    return spark

def _to_timestamp(s):
    return datetime.datetime.strptime(s, "%m%d%Y%H%M%S%f")

udf_to_timestamp = udf(_to_timestamp, TimestampType())


def clean_quote(df_quote):
    sp500 = pd.read_csv(r"C:\Users\lguo5\2021Fall\FIR7410\sp500list.csv")
    sp500_list = sp500["sp500"].tolist()
    cols = df_quote.columns[0]
    df_quote = df_quote.withColumn("Timestamp", substring(df_quote[cols], 0, 12)) \
        .withColumn("Exchange", substring(df_quote[cols], 13, 1)) \
        .withColumn("Ticker", substring(df_quote[cols], 14, 16)) \
        .withColumn("BidPrice", substring(df_quote[cols], 30, 11)) \
        .withColumn("BidSize", substring(df_quote[cols], 41, 7)) \
        .withColumn("AskPrice", substring(df_quote[cols], 48, 11)) \
        .withColumn("AskSize", substring(df_quote[cols], 59, 7)) \
        .withColumn("QuoteCondiction", substring(df_quote[cols], 66, 1)) \
        .withColumn("BidExchange", substring(df_quote[cols], 71, 1)) \
        .withColumn("AskExchange", substring(df_quote[cols], 72, 1)) \
        .withColumn("NBBOInd", substring(df_quote[cols], 89, 1)) \
        .withColumn("QuoteCancel", substring(df_quote[cols], 91, 1)) \
        .withColumn("SourceOfQuote", substring(df_quote[cols], 92, 1)) \
        .withColumn("NBBOCondition", substring(df_quote[cols], 93, 1)) \
        .withColumn("BestBidExch", substring(df_quote[cols], 94, 1)) \
        .withColumn("BestBidPrice", substring(df_quote[cols], 95, 11)) \
        .withColumn("BestBidSize", substring(df_quote[cols], 106, 7)) \
        .withColumn("BestAskExch", substring(df_quote[cols], 120, 1)) \
        .withColumn("BestAskPrice", substring(df_quote[cols], 121, 11)) \
        .withColumn("BestAskSize", substring(df_quote[cols], 132, 7)) \
        .withColumn("LULDNBBO", substring(df_quote[cols], 147, 1))
    df_quote = df_quote.withColumn("Ticker", trim(df_quote["Ticker"]))
    # only s&p 500 stocks are selected, but the consitutents in this case may be wrong
    df_quote = df_quote[df_quote.Ticker.isin(sp500_list)]
    # LULD non executable quotes are deleted
    df_quote = df_quote.where("LULDNBBO != 'B' and LULDNBBO != 'C' and LULDNBBO != 'D' and LULDNBBO != 'G'")
    # delenative bid-ask spread
    df_quote = df_quote.withColumn("BestBidPrice", (df_quote["BestBidPrice"] / 10000).cast(DoubleType()))
    df_quote = df_quote.withColumn("BestAskPrice", (df_quote["BestAskPrice"] / 10000).cast(DoubleType()))
    df_quote = df_quote.withColumn("BestBidSize", (df_quote["BestBidSize"] * 100).cast(IntegerType()))
    df_quote = df_quote.withColumn("BestAskSize", (df_quote["BestAskSize"] * 100).cast(IntegerType()))
    df_quote = df_quote.withColumn("BidAskSpread", round(df_quote["BestAskPrice"] - df_quote["BestBidPrice"], 2))
    df_quote = df_quote.where("BidAskSpread > 0 and BidAskSpread <= 5")
    df_quote = df_quote.withColumn("Timestamp", udf_to_timestamp(concat(lit(cols[2:10]), df_quote["Timestamp"])))
    df_quote = df_quote.select("Timestamp", "Ticker", "BestBidPrice", "BestBidSize",
                               "BestAskPrice", "BestAskSize", "BidAskSpread")
    return df_quote

def clean_trade(df_trade):
    sp500 = pd.read_csv(r"C:\Users\lguo5\2021Fall\FIR7410\sp500list.csv")
    sp500_list = sp500["sp500"].tolist()
    cols_td = df_trade.columns[0]
    df_trade = df_trade.withColumn("Timestamp", substring(df_trade[cols_td], 0, 12)) \
        .withColumn("Exchange", substring(df_trade[cols_td], 13, 1)) \
        .withColumn("Ticker", substring(df_trade[cols_td], 14, 16)) \
        .withColumn("SalesCond", substring(df_trade[cols_td], 30, 4)) \
        .withColumn("TradingVolume", substring(df_trade[cols_td], 34, 9)) \
        .withColumn("TradePrice", substring(df_trade[cols_td], 43, 11)) \
        .withColumn("TradeRef", substring(df_trade[cols_td], 57, 16))
    df_trade = df_trade.withColumn("Ticker", trim(df_trade["Ticker"]))
    df_trade = df_trade.withColumn("TradingVolume", (df_trade["TradingVolume"] * 100).cast(IntegerType()))
    df_trade = df_trade.withColumn("TradePrice", (df_trade["TradePrice"] / 10000).cast(DoubleType()))
    df_trade = df_trade[df_trade.Ticker.isin(sp500_list)]
    # sales condition
    sales_condition = ["@F I", "@  I", "@F  ", "@   ", " F  ", " F I", "   I"]
    df_trade = df_trade[df_trade.SalesCond.isin(sales_condition)]
    df_trade = df_trade.withColumn("Timestamp", udf_to_timestamp(concat(lit(cols_td[2:10]), df_trade["Timestamp"])))
    df_trade = df_trade.select("Timestamp", "TradeRef", "Ticker", "TradingVolume", "TradePrice")
    # delete trades in first and last five minutes
    dtfrom = datetime.datetime.strptime(cols_td[2:10] + "093500", "%m%d%Y%H%M%S")
    dtto = datetime.datetime.strptime(cols_td[2:10] + "155500", "%m%d%Y%H%M%S")
    df_trade = df_trade.where((df_trade.Timestamp > dtfrom) & (df_trade.Timestamp < dtto))
    return df_trade

def read_data(df_quote, df_trade):
    # using asifJoin from tempo package, we are able to match each trade with previous quote efficiently
    df_quote_ts = TSDF(df_quote, ts_col="Timestamp", partition_cols=["Ticker"])
    df_trade_ts = TSDF(df_trade, ts_col="Timestamp", partition_cols=["Ticker"])
    joined = df_trade_ts.asofJoin(df_quote_ts)
    # apply Lee and Ready (1991) algorithm
    merged = joined.df
    merged = merged.withColumn("MidPrice", round((merged["right_BestAskPrice"] + merged["right_BestBidPrice"]) / 2, 3))
    merged = merged.withColumn("WeightedMidPrice", (merged["right_BestAskPrice"] * merged["right_BestBidSize"]
                                                    + merged["right_BestBidPrice"] * merged["right_BestAskSize"])
                                                    / (merged["right_BestBidSize"] + merged["right_BestAskSize"]))
    win_spec = Window.orderBy(merged["Timestamp"]).partitionBy(merged["Ticker"])
    merged = merged.withColumn("D", when((merged["TradePrice"] > merged["MidPrice"]), 1)
                                   .when((merged["TradePrice"] < merged["MidPrice"]), -1)
                                   .when((merged["TradePrice"] == merged["MidPrice"]) & (merged["TradePrice"] > lag(merged["TradePrice"]).over(win_spec)), 1)
                                   .when((merged["TradePrice"] == merged["MidPrice"]) & (merged["TradePrice"] < lag(merged["TradePrice"]).over(win_spec)), -1)
                                   .otherwise(None)
                               )
    # do the forward fill because Lee and Ready algorithm consider the previous tick test
    window = Window.orderBy('Timestamp').partitionBy("Ticker").rowsBetween(-sys.maxsize, 0)
    # define the forward-filled column
    filled_column = last(merged['D'], ignorenulls=True).over(window)
    # do the fill
    merged = merged.withColumn('DI', filled_column)
    # calculate the dollar weight for each trade
    merged = merged.withColumn("DollarVolume", round(merged["TradingVolume"] * merged["TradePrice"], 0))
    merged = merged.withColumn("DollarWeight", col("DollarVolume") / sum("DollarVolume").over(Window.partitionBy("Ticker")))
    # calculate the quoted spread effective spread based on two different mid price
    merged = merged.withColumn("QuotedSpread", (merged["right_BidAskSpread"] / merged["MidPrice"]) * 10000)
    merged = merged.withColumn("ESMid", ((2 * merged["DI"] * (merged["TradePrice"] - merged["MidPrice"])) / merged["MidPrice"]) * 10000)
    merged = merged.withColumn("ESWMid", ((2 * merged["DI"] * (merged["TradePrice"] - merged["WeightedMidPrice"])) / merged["WeightedMidPrice"]) * 10000)
    merged = merged.withColumn("WQuotedSpread", merged["QuotedSpread"] * merged["DollarWeight"])
    merged = merged.withColumn("WESMid", merged["ESMid"] * merged["DollarWeight"])
    merged = merged.withColumn("WESWMid", merged["ESWMid"] * merged["DollarWeight"])
    merged = merged.withColumn("WTradePrice", merged["TradePrice"] * merged["DollarWeight"])
    stock_day = merged.groupBy("Ticker").agg(sum("WQuotedSpread").alias("QS"),
                                             sum("WESMid").alias("MidPointES"),
                                             sum("WESWMid").alias("WeightedMidPointES"),
                                             sum("WTradePrice").alias("DWTradePrice"),
                                             (sum("DollarVolume") / 1000000).alias("DollarVolume"),
                                             (count("Ticker") / 1000).alias("NT"))
    final_df = stock_day.toPandas()

    return merged


def realized_spread_pi(df_quote, df_trade):
    df_quote_ts = TSDF(df_quote, ts_col="Timestamp", partition_cols=["Ticker"])
    df_trade_ts = TSDF(df_trade, ts_col="Timestamp", partition_cols=["Ticker"])
    joined = df_trade_ts.asofJoin(df_quote_ts)
    # apply Lee and Ready (1991) algorithm
    merged = joined.df
    merged = merged.withColumn("MidPrice", round((merged["right_BestAskPrice"] + merged["right_BestBidPrice"]) / 2, 3))
    merged = merged.withColumn("WeightedMidPrice", (merged["right_BestAskPrice"] * merged["right_BestBidSize"]
                                                    + merged["right_BestBidPrice"] * merged["right_BestAskSize"])
                               / (merged["right_BestBidSize"] + merged["right_BestAskSize"]))
    win_spec = Window.orderBy(merged["Timestamp"]).partitionBy(merged["Ticker"])
    merged = merged.withColumn("D", when((merged["TradePrice"] > merged["MidPrice"]), 1)
                               .when((merged["TradePrice"] < merged["MidPrice"]), -1)
                               .when((merged["TradePrice"] == merged["MidPrice"]) & (
                merged["TradePrice"] > lag(merged["TradePrice"]).over(win_spec)), 1)
                               .when((merged["TradePrice"] == merged["MidPrice"]) & (
                merged["TradePrice"] < lag(merged["TradePrice"]).over(win_spec)), -1)
                               .otherwise(None)
                               )
    # do the forward fill because Lee and Ready algorithm consider the previous tick test
    window = Window.orderBy('Timestamp').partitionBy("Ticker").rowsBetween(-sys.maxsize, 0)
    # define the forward-filled column
    filled_column = last(merged['D'], ignorenulls=True).over(window)
    # do the fill
    merged = merged.withColumn('DI', filled_column)
    # calculate the dollar weight for each trade
    merged = merged.withColumn("DollarVolume", round(merged["TradingVolume"] * merged["TradePrice"], 0))
    merged = merged.withColumn("DollarWeight",
                               col("DollarVolume") / sum("DollarVolume").over(Window.partitionBy("Ticker")))
    merged = merged.withColumnRenamed("right_Timestamp", "previous_quote_Timestamp")
    df_trade_5m = merged.withColumn('Timestamp', df_trade["Timestamp"] + expr('INTERVAL 5 MINUTES'))\
                        .select("Timestamp", "Ticker", "TradePrice", "previous_quote_Timestamp", 
                                "MidPrice", "WeightedMidPrice", "DI", "DollarVolume", "DollarWeight", "TradingVolume")
    df_quote_ts = TSDF(df_quote, ts_col="Timestamp", partition_cols=["Ticker"])
    df_trade_ts = TSDF(df_trade_5m, ts_col="Timestamp", partition_cols=["Ticker"])
    joined_5m = df_trade_ts.asofJoin(df_quote_ts).df
    # calculate 5m after Midprice and weighted midprice
    joined_5m = joined_5m.withColumn("MidPrice_5m", round((joined_5m["right_BestAskPrice"] + joined_5m["right_BestBidPrice"]) / 2, 3))
    joined_5m = joined_5m.withColumn("WeightedMidPrice_5m", (joined_5m["right_BestAskPrice"] * joined_5m["right_BestBidSize"]
                                                    + joined_5m["right_BestBidPrice"] * joined_5m["right_BestAskSize"])
                               / (joined_5m["right_BestBidSize"] + joined_5m["right_BestAskSize"]))
    joined_5m = joined_5m.withColumn("PIMid", (
                (2 * joined_5m["DI"] * (joined_5m["MidPrice_5m"] - joined_5m["MidPrice"])) / joined_5m["MidPrice"]) * 10000)
    joined_5m = joined_5m.withColumn("RSMid", (
            (2 * joined_5m["DI"] * (joined_5m["TradePrice"] - joined_5m["MidPrice_5m"])) / joined_5m["MidPrice"]) * 10000)
    joined_5m = joined_5m.withColumn("PIMid_dw", joined_5m["PIMid"] * joined_5m["DollarWeight"])
    joined_5m = joined_5m.withColumn("RSMid_dw", joined_5m["RSMid"] * joined_5m["DollarWeight"])
    # new measure
    joined_5m = joined_5m.withColumn("PIWMid", (
            (2 * joined_5m["DI"] * (joined_5m["WeightedMidPrice_5m"] - joined_5m["WeightedMidPrice"])) / joined_5m["WeightedMidPrice"]) * 10000)
    joined_5m = joined_5m.withColumn("RSWMid", (
            (2 * joined_5m["DI"] * (joined_5m["TradePrice"] - joined_5m["WeightedMidPrice_5m"])) / joined_5m[
        "WeightedMidPrice"]) * 10000)
    joined_5m = joined_5m.withColumn("PIWMid_dw", joined_5m["PIWMid"] * joined_5m["DollarWeight"])
    joined_5m = joined_5m.withColumn("RSWMid_dw", joined_5m["RSWMid"] * joined_5m["DollarWeight"])
    stock_day = joined_5m.groupBy("Ticker").agg(sum("PIMid_dw").alias("PIMid_dw"),
                                             sum("RSMid_dw").alias("RSMid_dw"),
                                             sum("PIWMid_dw").alias("PIWMid_dw"),
                                             sum("RSWMid_dw").alias("RSWMid_dw"),
                                             (sum("DollarVolume") / 1000000).alias("DollarVolume"),
                                             sum("TradingVolume").alias("TradingVolume"),
                                             (count("Ticker") / 1000).alias("NT"))
    return stock_day.toPandas().to_csv(r"C:\Users\lguo5\2021Fall\FIR7410\pi_rs.csv", index=False)


if __name__ == "__main__":
    if not os.path.exists(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data_extracted"):
        zip_extract(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data")
    if not os.path.exists(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data_extracted"):
        zip_extract(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data")
    spark = create_spark_session()
    quote_1207 = spark.read.format("csv") \
                         .option("header", "true") \
                         .load(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data_extracted\taqnbbo20151207")
    trade_1207 = spark.read.format("csv") \
                         .option("header", "true") \
                         .load(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data_extracted\taqtrade20151207")
    quote_1208 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data_extracted\taqnbbo20151208")
    trade_1208 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data_extracted\taqtrade20151208")
    quote_1209 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data_extracted\taqnbbo20151209")
    trade_1209 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data_extracted\taqtrade20151209")
    quote_1210 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data_extracted\taqnbbo20151210")
    trade_1210 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data_extracted\taqtrade20151210")
    quote_1211 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\quote_data_extracted\taqnbbo20151211")
    trade_1211 = spark.read.format("csv") \
        .option("header", "true") \
        .load(r"C:\Users\lguo5\2021Fall\FIR7410\trade_data_extracted\taqtrade20151211")
    cleaned_quote_1207 = clean_quote(quote_1207)
    cleaned_trade_1207 = clean_trade(trade_1207)
    cleaned_quote_1208 = clean_quote(quote_1208)
    cleaned_trade_1208 = clean_trade(trade_1208)
    cleaned_quote_1209 = clean_quote(quote_1209)
    cleaned_trade_1209 = clean_trade(trade_1209)
    cleaned_quote_1210 = clean_quote(quote_1210)
    cleaned_trade_1210 = clean_trade(trade_1210)
    cleaned_quote_1211 = clean_quote(quote_1211)
    cleaned_trade_1211 = clean_trade(trade_1211)
    kk = realized_spread_pi(cleaned_quote_1207, cleaned_trade_1207)
    df_quote = cleaned_quote_1207.union(cleaned_quote_1208) \
                                 .union(cleaned_quote_1209) \
                                 .union(cleaned_quote_1210) \
                                 .union(cleaned_quote_1211)

    df_trade = cleaned_trade_1207.union(cleaned_trade_1208) \
                                 .union(cleaned_trade_1209) \
                                 .union(cleaned_trade_1210) \
                                 .union(cleaned_trade_1211)
    #read_data(df_quote, df_trade)
    realized_spread_pi(df_quote, df_trade)
    print("end")