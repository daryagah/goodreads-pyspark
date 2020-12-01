# Databricks notebook source
# MAGIC %md
# MAGIC # Data Extraction and Transformation

# COMMAND ----------

import re
import os
import numpy as np
import pandas as pd

from pyspark.sql import functions as f

# COMMAND ----------

import sys
!{sys.executable} -m pip install gdown
import gdown
import gzip
import json

# COMMAND ----------

# DBTITLE 1,Download full books table and genres
# MAGIC %sh
# MAGIC gdown 'https://drive.google.com/uc?id=1LXpK1UfqtP89H1tYy0pBGHjYk8IhigUK'
# MAGIC gdown 'https://drive.google.com/uc?id=1ah0_KpUterVi-AHxJ03iKD6O0NfbK0md'
# MAGIC gzip -d goodreads_books.json.gz goodreads_book_genres_initial.json.gz

# COMMAND ----------

# DBTITLE 1,Load smaller tables
books = spark.read.json("file:/databricks/driver/goodreads_books.json")
books_small = spark.read.format("csv").option("inferSchema", "true").option("header","true").load("/FileStore/tables/books.csv")
genres = spark.read.json("file:/databricks/driver/goodreads_book_genres_initial.json")
ratings = spark.read.format("csv").option("inferSchema", "true").option("header","true").load("/FileStore/tables/ratings.csv")
to_read = spark.read.format("csv").option("inferSchema", "true").option("header","true").load("/FileStore/tables/to_read.csv")

# COMMAND ----------

genres.show()

# COMMAND ----------

genres = genres.select(f.col("book_id"), f.col("genres.*"))
genres.show()

# COMMAND ----------

# https://stackoverflow.com/questions/56389696/select-column-name-per-row-for-max-value-in-pyspark
from pyspark.sql.types import IntegerType, StringType
genres = genres.na.fill(0)

cols = genres.columns
maxcol = f.udf(lambda row: cols[row.index(max(row)) +1], StringType())

genres = genres.withColumn("genre", maxcol(f.struct([genres[x] for x in genres.columns[1:]]))).select("book_id", "genre")
genres.show()

# COMMAND ----------

books_small.show()

# COMMAND ----------

books.show()

# COMMAND ----------

ratings.show()

# COMMAND ----------

books_small.count()

# COMMAND ----------

books = books.withColumnRenamed("book_id", "goodreads_book_id")
genres = genres.withColumnRenamed("book_id", "goodreads_book_id")

books_df = books_small.join(books.select("goodreads_book_id", "publication_year", "description", "popular_shelves", "num_pages", "similar_books"), "goodreads_book_id", "inner")
books_df = books_df.join(genres, "goodreads_book_id", "left")
books_df = books_df.orderBy(["work_id", "publication_year"], ascending=[True, False]).dropDuplicates(["work_id"])
books_df = books_df.filter((f.col("language_code") == "en-US") | (f.col("language_code") == "en-GB") | (f.col("language_code") == "eng") | (f.col("language_code") == "en-CA"))
books_df = books_df.select("book_id", "work_id", "authors", "original_publication_year", "title", "description", "popular_shelves", "genre", "num_pages", "similar_books", "average_rating", "work_ratings_count")
books_df.count()

# COMMAND ----------

ratings.count()

# COMMAND ----------

to_read = to_read.groupBy("user_id").agg(f.collect_set("book_id").alias("to_read"))
to_read.show()

# COMMAND ----------

books_df = books_df.withColumn("authors", f.split(f.col("authors"), ', '))
books_df.show()

# COMMAND ----------

# make book ids different from user ids
books_df = books_df.withColumn("book_id", (books_df.book_id+100000))
ratings = ratings.withColumn("book_id", (ratings.book_id+100000))
print(books_df.select(f.min("book_id")).collect()[0][0])
print(ratings.select(f.min("book_id")).collect()[0][0])

# COMMAND ----------

books_df.printSchema()

# COMMAND ----------

books_df = books_df.withColumn("num_pages", f.col("num_pages").cast("Integer")).withColumn("average_rating", f.col("average_rating").cast("Double")).withColumn("work_ratings_count", f.col("work_ratings_count").cast("Integer"))
books_df.printSchema()

# COMMAND ----------

to_read.write.format("parquet").saveAsTable("users")

# COMMAND ----------

books_df.write.format("parquet").saveAsTable("books")

# COMMAND ----------

ratings.write.format("parquet").saveAsTable("ratings")

# COMMAND ----------

users = spark.read.parquet("/user/hive/warehouse/users")

# COMMAND ----------

books = spark.read.parquet("/user/hive/warehouse/books")

# COMMAND ----------

ratings = spark.read.parquet("/user/hive/warehouse/ratings")

# COMMAND ----------

def get_book_title(book_id):
  return books.filter(f.col("book_id")==book_id).select("title").collect()[0][0]

# COMMAND ----------

# only include books from the dataset in ratings
ratings_small = ratings.join(books.select("book_id"), "book_id", "inner")
ratings_small.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Exploration

# COMMAND ----------

# DBTITLE 1,Rating distribution
display(ratings.select("user_id","rating").groupBy("rating").agg(f.avg("rating")))

# COMMAND ----------

# DBTITLE 1,Highest number of review by a user
# MAGIC %sql
# MAGIC select user_id, count(user_id) from ratings group by user_id order by count(user_id) desc limit 10

# COMMAND ----------

# DBTITLE 1,Highest rated book
# MAGIC %sql 
# MAGIC select book_id, title, average_rating from books group by book_id, title, average_rating order by average_rating desc limit 5

# COMMAND ----------

# DBTITLE 1,Books with the most number of ratings
# MAGIC %sql
# MAGIC select title, work_ratings_count, average_rating from books order by work_ratings_count desc limit 10;

# COMMAND ----------

# DBTITLE 1,Number of reviews per book publication year
# MAGIC %sql
# MAGIC select original_publication_year, count(*) as count from books
# MAGIC where original_publication_year > 1900 group by original_publication_year

# COMMAND ----------

# DBTITLE 1,Average reviews per book publication year
# MAGIC %sql
# MAGIC select mean(average_rating), original_publication_year from books
# MAGIC where original_publication_year > 1900 group by original_publication_year

# COMMAND ----------

# DBTITLE 1,Authors with most books
# MAGIC %sql
# MAGIC select authors, count(distinct title) as number_of_books from books
# MAGIC group by authors
# MAGIC order by number_of_books desc limit 10

# COMMAND ----------

# DBTITLE 1,Top rated authors
# MAGIC %sql
# MAGIC select authors, mean(average_rating) as average_rating from books
# MAGIC group by authors
# MAGIC order by average_rating desc limit 10

# COMMAND ----------

# DBTITLE 1,Best books by Stephen King
# MAGIC %sql
# MAGIC select title, average_rating from books
# MAGIC where authors[0] like 'Stephen King' group by average_rating, title
# MAGIC order by average_rating desc

# COMMAND ----------

# DBTITLE 1,Number of books by year
# MAGIC %sql
# MAGIC select original_publication_year, count(*) as count from books group by original_publication_year order by count desc

# COMMAND ----------

# MAGIC %md
# MAGIC #Collaborative Filtering

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
als = ALS( userCol="user_id", itemCol="book_id", ratingCol="rating",
 coldStartStrategy="drop", nonnegative = True, implicitPrefs = False)

(training, test) = ratings_small.randomSplit([0.8, 0.2])

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
param_grid = ParamGridBuilder() \
 .addGrid(als.rank, [10, 50, 75, 100]) \
 .addGrid(als.maxIter, [5, 50, 75, 100]) \
 .addGrid(als.regParam, [.01, .05, .1, .15]) \
 .build()
# Define evaluator as RMSE
evaluator = RegressionEvaluator(metricName = "rmse", 
 labelCol = "rating", 
 predictionCol = "prediction")
# Print length of evaluator
print ("Num models to be tested using param_grid: ", len(param_grid))

# COMMAND ----------

# Build cross validation using CrossValidator
cv = CrossValidator(estimator = als, 
 estimatorParamMaps = param_grid, 
 evaluator = evaluator, 
 numFolds = 5)
model = als.fit(training)
predictions = model.transform(test)
predictions.show(n = 10)

# COMMAND ----------

rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

# Generate n recommendations for all users
ALS_recommendations = model.recommendForAllUsers(numItems = 10) # n — 10
ALS_recommendations.show(n = 10)

# COMMAND ----------

def recommend_for_user(user_id):
  res = ALS_recommendations.filter(ALS_recommendations.user_id==user_id).select("recommendations.book_id").take(1)[0][0]
  for r in res:
    print(get_book_title(r))

recommend_for_user(1)

# COMMAND ----------

def get_to_read(user_id):
  res = users.filter(users.user_id==user_id).select("to_read").collect()[0][0]
  for r in res:
    print(get_book_title(r + 100000))

get_to_read(1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Content Based Filtering

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering by book descriptions

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

# COMMAND ----------

# DBTITLE 0,Clustering by book descriptions
tokenizer = Tokenizer(inputCol="description", outputCol="words")
wordsData = tokenizer.transform(books)

remover = StopWordsRemover(inputCol="words", outputCol="filtered")
filteredData = remover.transform(wordsData)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(filteredData)

idf = IDF(inputCol="rawFeatures", outputCol="tfidffeatures")
idfModel = idf.fit(featurizedData)

tfidfData = idfModel.transform(featurizedData)

# COMMAND ----------

from pyspark.ml.feature import PCA

pca = PCA(k = 5, inputCol = 'tfidffeatures', outputCol = 'features').fit(tfidfData)

data_pca = pca.transform(tfidfData)

data_pca.select("book_id", "features").show(truncate=False)

# COMMAND ----------

# DBTITLE 1,K-means
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import ClusteringEvaluator

kmeans = KMeans().setK(20).setSeed(1)
model = kmeans.fit(data_pca)

predictions = model.transform(data_pca)

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

def recommend_by_book(book_id):
  cluster = predictions.filter(predictions.book_id == book_id).select("prediction").collect()[0][0]
  titles = predictions.filter(predictions.prediction == cluster).select("title").collect()
  for title in titles:
    print(title[0])
    
recommend_by_book(100001)

# COMMAND ----------

# DBTITLE 1,Locality Sensitive Hashing: Bucketed Random Projection for Euclidean Distance
from pyspark.ml.feature import BucketedRandomProjectionLSH

brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=5, numHashTables=10)
model = brp.fit(data_pca)

# COMMAND ----------

def find_nearest_books(book_id, num):
  key = data_pca.filter(data_pca.book_id == book_id).select("features").collect()[0][0]
  res = model.approxNearestNeighbors(data_pca, key, num).select("book_id").collect()
  for r in res:
    print(get_book_title(r[0]))

find_nearest_books(100001, 10)

# COMMAND ----------

# DBTITLE 1,Latent Dirichlet allocation
from pyspark.ml.clustering import LDA

vectorizer = CountVectorizer(inputCol="filtered", outputCol="features")
cv = vectorizer.fit(filteredData)
featurizedData = cv.transform(filteredData)

lda = LDA(k=20, maxIter=10)
model = lda.fit(featurizedData)

topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show()

transformed = model.transform(featurizedData)
transformed.show()

# COMMAND ----------

kmeans = KMeans(featuresCol="topicDistribution").setK(20).setSeed(1)
model = kmeans.fit(transformed)

predictions = model.transform(transformed)

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering by book features

# COMMAND ----------

# DBTITLE 1,K-means
from pyspark.ml.feature import VectorAssembler, StringIndexer

d = books.select(books.authors[0], "book_id", "title", "genre", "original_publication_year", "average_rating").withColumnRenamed("authors[0]", "author").dropna()

ind1 = StringIndexer(inputCol='genre',outputCol='genre_ind',handleInvalid='skip')
ind2 = StringIndexer(inputCol='author',outputCol='author_ind',handleInvalid='skip')
assembler = VectorAssembler(inputCols=["genre_ind", "author_ind", "original_publication_year", "average_rating"], outputCol="features")
pipe = Pipeline(stages=[ind1, ind2, assembler])
data = pipe.fit(d).transform(d)

kmeans = KMeans().setK(20).setSeed(1)
model = kmeans.fit(data)

predictions = model.transform(data)
predictions.select("title", "prediction").orderBy("prediction").show(truncate=False)

# COMMAND ----------

evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

recommend_by_book(100001)

# COMMAND ----------

# DBTITLE 1,Bucketed Random Projection
brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes", bucketLength=4, numHashTables=8)
model = brp.fit(data)

def find_nearest_books(book_id, num):
  key = data.filter(data.book_id == book_id).select("features").collect()[0][0]
  res = model.approxNearestNeighbors(data, key, num).select("book_id").collect()
  for r in res:
    print(get_book_title(r[0]))

find_nearest_books(100001, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC # Graph Analysis

# COMMAND ----------

from graphframes import *

# COMMAND ----------

# DBTITLE 1,User vertices
usersV = users.withColumnRenamed("user_id", "id").withColumn("type", f.lit("User")).withColumn("authors", f.lit(None)).withColumn("original_publication_year", f.lit(None)).withColumn("title", f.lit(None)).withColumn("description", f.lit(None)).withColumn("genre", f.lit(None)).withColumn("popular_shelves", f.lit(None)).withColumn("num_pages", f.lit(None)).withColumn("similar_books", f.lit(None)).withColumn("average_rating", f.lit(None)).withColumn("work_ratings_count", f.lit(None)).select("id", "type", "to_read", "authors", "original_publication_year", "title", "description", "genre", "popular_shelves", "num_pages", "similar_books", "average_rating", "work_ratings_count").distinct()

usersV.show()

# COMMAND ----------

# DBTITLE 1,Book vertices
booksV = books.withColumnRenamed("book_id", "id").withColumn("type", f.lit("Book")).withColumn("to_read", f.lit(None)).select("id", "type", "to_read", "authors", "original_publication_year", "title", "description", "genre", "popular_shelves", "num_pages", "similar_books", "average_rating", "work_ratings_count").distinct()

booksV.show()

# COMMAND ----------

# DBTITLE 1,Combine vertices, create edges, create the graph
vertices = usersV.union(booksV)
edges = ratings_small.withColumnRenamed("user_id", "src").withColumnRenamed("book_id", "dst").select("src", "dst", "rating")
graph = GraphFrame(vertices, edges)

graph.vertices.show()

# COMMAND ----------

graph.edges.show()

# COMMAND ----------

# DBTITLE 1,Plot the graph
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# https://stackoverflow.com/questions/45720931/pyspark-how-to-visualize-a-graphframe, https://stackoverflow.com/questions/27030473/how-to-set-colors-for-nodes-in-networkx
def PlotGraph(edge_list):
    G = nx.DiGraph()
    for row in edge_list.select('src','dst').take(1000):
      G.add_edge(row['src'], row['dst'])
    color_map = []
    for node in G:
      if node < 100000:
        color_map.append('lightseagreen')
      else: 
        color_map.append('tomato') 
    plt.figure(figsize=(15,10))
    plt.margins(0.0)
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos=pos, node_color=color_map, with_labels=False, node_size=50)
    nx.draw_networkx_edges(G, pos=pos, alpha=0.3, width=0.5, arrowsize=5, node_size=50)
    
PlotGraph(graph.edges)

# COMMAND ----------

# For book 1 only
def PlotGraph(edge_list):
    G = nx.DiGraph()
    for row in edge_list.select('src','dst').take(1000):
      G.add_edge(row['src'], row['dst'])
    color_map = []
    for node in G:
      if node < 100000:
        color_map.append('lightseagreen')
      else: 
        color_map.append('tomato') 
    plt.figure(figsize=(15,10))
    plt.margins(0.0)
    nx.draw(G, node_color=color_map, width=0.5, arrowsize=5, with_labels=True, font_size=8, node_size=50)
    
PlotGraph(graph.filterEdges(f.col("src") == 1).edges)

# COMMAND ----------

# DBTITLE 1,Users with the most reviews
# Dataframe approach
graph.edges.groupBy("src").count().orderBy("count", ascending=False).show(5)

# COMMAND ----------

# Graph approach
graph.outDegrees.orderBy("outDegree", ascending=False).show(5)

# COMMAND ----------

# DBTITLE 1,Most popular books
# Dataframe approach
graph.edges.groupBy("dst").count().orderBy("count", ascending=False).show(5)

# COMMAND ----------

# Graph approach
graph.inDegrees.orderBy("inDegree", ascending=False).show(5)

# COMMAND ----------

# DBTITLE 1,Subgraphs
# New books with the highest ratings
g1 = graph.filterVertices(f.col("original_publication_year") > 2000)
g1.vertices.select("id", "title", "original_publication_year", "average_rating").orderBy("average_rating", ascending=False).show(5, truncate=False)

# COMMAND ----------

# Users who gave the most 1-star ratings
g2 = graph.filterEdges(f.col("rating") == 1)
g2.edges.groupBy("src").count().orderBy("count", ascending=False).show(5, truncate=False)

# COMMAND ----------

# DBTITLE 1,Motif finding
# Find 3 users who read the same book
graph.find("(u1)-[r1]->(b); (u2)-[r2]->(b); (u3)-[r3]->(b)").filter("r1 == r2").filter("r2 == r3").show()

# COMMAND ----------

# DBTITLE 1,Breadth-first search
# Find path between User 1 and Book 100001
bfs = graph.bfs("id = 1", "id = 100004", maxPathLength=10)
bfs.show()

# COMMAND ----------

# DBTITLE 1,Label propagation
lp = graph.labelPropagation(maxIter=4)
lp.show()

# COMMAND ----------

lp.select(f.countDistinct("label")).collect()[0][0]

# COMMAND ----------

lp.groupBy("label").count().sort("count").show()

# COMMAND ----------

assembler = VectorAssembler(inputCols=["id"], outputCol="features")
data = assembler.transform(lp).withColumnRenamed("label", "prediction")

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(data)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

# DBTITLE 1,Page rank
pr = graph.pageRank(resetProbability=0.15, tol=0.01)
pr.vertices.select("id", "authors", "title", "average_rating", "pagerank").orderBy("pagerank", ascending=False).show(10, truncate=False)

# COMMAND ----------

graph.edges.groupBy("dst").count().orderBy("count", ascending=False).show(10)

# COMMAND ----------

display(pr.vertices.orderBy("pagerank", ascending=False).limit(20))

# COMMAND ----------

# DBTITLE 1,Power iteration clustering
from pyspark.ml.clustering import PowerIterationClustering

pic = PowerIterationClustering(k=30, maxIter=10, weightCol="rating")
pic_results = pic.assignClusters(graph.edges)

evaluator = ClusteringEvaluator()

assembler = VectorAssembler(inputCols=["id"], outputCol="features")
data = assembler.transform(pic_results).withColumnRenamed("cluster", "prediction")

silhouette = evaluator.evaluate(data)
print("Silhouette with squared euclidean distance = " + str(silhouette))

# COMMAND ----------

pic_results.filter("id < 100000").groupBy("cluster").agg(f.count("id")).orderBy("cluster").show()

# COMMAND ----------

pic_df = graph.edges.join(pic_results, graph.edges.src==pic_results.id, "left")
pic_df.show()

# COMMAND ----------

# DBTITLE 1,Collaborative filtering based on power iteration clusters
(training, test) = pic_df.randomSplit([0.8, 0.2])

als = ALS(maxIter=20, regParam=0.01, userCol="cluster", itemCol="dst", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# COMMAND ----------

userRecs = model.recommendForAllUsers(10)
userRecs.orderBy("cluster").show(5, truncate=False)

# COMMAND ----------

pic_df = pic_df.join(userRecs, "cluster", "left")

# COMMAND ----------

def recommend_for_user(user_id):
  res = pic_df.filter(pic_df.src==user_id).select("recommendations.dst").take(1)[0][0]
  for r in res:
    print(get_book_title(r))

recommend_for_user(1)
