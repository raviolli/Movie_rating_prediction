import sys
import os
import math
from pyspark.mllib.recommendation import ALS
from test_helper import Test

# DataBricks Test code 
# Mahcine Learning - Mid down the page
# Ravi Bhanabhai

def computeError(predictedRDD, actualRDD):
    """ Compute the root mean squared error between predicted and actual
    Args:
        predictedRDD: predicted ratings for each movie and each user where each entry is in the form
                      (UserID, MovieID, Rating)
        actualRDD: actual ratings where each entry is in the form (UserID, MovieID, Rating)
    Returns:
        RSME (float): computed RSME value
    """
    # Transform predictedRDD into the tuples of the form ((UserID, MovieID), Rating)
    predictedReformattedRDD = predictedRDD.map( lambda pred: ( (pred[0], pred[1]), pred[2] ))

    # Transform actualRDD into the tuples of the form ((UserID, MovieID), Rating)
    actualReformattedRDD = actualRDD.map( lambda actual: ( (actual[0], actual[1]), actual[2] ))

    # Compute the squared error for each matching entry (i.e., the same (User ID, Movie ID) in each
    # RDD) in the reformatted RDDs using RDD transformtions - do not use collect() join:: --> (UserID, MovieID (Rating1, Rating2))
    squaredErrorsRDD = (predictedReformattedRDD.join(actualReformattedRDD).map(lambda data: (data[1][1]-data[1][0])**2 )) 

    # Compute the total squared error - do not use collect()
    totalError = squaredErrorsRDD.reduce(lambda x,y: x+y )	#sum up squared error
    #totalError = squaredErrorsRDD.map(lambda data: sum(data) )	#sum up squared error --> this doesn't work cause it won REDUCE, rather sums up same value
 
    # Count the number of entries for which you computed the total squared error
    numRatings = squaredErrorsRDD.count()

    # Using the total squared error and the number of entries, compute the RSME
    return math.sqrt( float(totalError)/float(numRatings))


def get_ratings_tuple(entry):
    """ Parse a line in the ratings dataset
    Args:
        entry (str): a line in the ratings dataset in the form of UserID::MovieID::Rating::Timestamp
    Returns:
        tuple: (UserID, MovieID, Rating)
    """
    items = entry.split('::')
    return int(items[0]), int(items[1]), float(items[2])


def get_movie_tuple(entry):
    """ Parse a line in the movies dataset
    Args:
        entry (str): a line in the movies dataset in the form of MovieID::Title::Genres
    Returns:
        tuple: (MovieID, Title)
    """
    items = entry.split('::')
    return int(items[0]), items[1]
  
# First, implement a helper function `getCountsAndAverages` using only Python
def getCountsAndAverages(intuple):
    """ Calculate average rating
    Args:
        IDandRatingsTuple: a single tuple of (MovieID, (Rating1, Rating2, Rating3, ...))
    Returns:
        tuple: a tuple of (MovieID, (number of ratings, averageRating))
    """
    
    movieID = intuple[0]
    rating = intuple[1]
    numrating = len ( rating )
    avgrating = sum( rating ) / float( numrating )
    
    return (movieID, (numrating, avgrating) )

#---------------------------------------------------------------
# Function Code Ends
# Work Code Beings
#---------------------------------------------------------------

baseDir = os.path.join('databricks-datasets')
inputPath = os.path.join('cs100', 'lab4', 'data-001')

ratingsFilename = os.path.join(baseDir, inputPath, 'ratings.dat.gz')
moviesFilename = os.path.join(baseDir, inputPath, 'movies.dat')

numPartitions = 2
rawRatings = sc.textFile(ratingsFilename).repartition(numPartitions)
rawMovies = sc.textFile(moviesFilename)
   
ratingsRDD = rawRatings.map(get_ratings_tuple).cache()  # This returns tuple: (UserID, MovieID, Rating)
moviesRDD = rawMovies.map(get_movie_tuple).cache()      # This returns tuple: (MovieID, Title)

idRatingRDD = ratingsRDD.map(lambda truple: (truple[1], truple[2]) ).groupByKey().cache() # This returns tuple: (MovieID, (Rating,...,Rating))
avgRating = idRatingRDD.map(getCountsAndAverages)       # This returns tuple: (MovieID, (# of ratings, avgerge Ratings))

titleRating = avgRating.join(moviesRDD).map(lambda quple: (quple[1][0][1], quple[1][1], quple[1][0][0]) ) # (average rating, movie name, number of ratings)
topTitleRating = titleRating.filter(lambda truple: truple[2] > 500).sortBy(lambda truple: truple[0], False) # This returns truple (AvgRating, Movie Title, # of Ratings > 500 )
  
ratingsCount = ratingsRDD.count()
moviesCount = moviesRDD.count()

#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
# ------------------- Machine Learning Aspect
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------

trainingRDD, validationRDD, testRDD = ratingsRDD.randomSplit([6, 2, 2], seed=0L)

validationForPredictRDD = validationRDD.map(lambda data: (data[0], data[1]))
testForPredictingRDD = testRDD.map(lambda data: (data[0], data[1]))

seed = 5L
iterations = 15
regularizationParameter = 0.1
rank = 4
errors = [0, 0, 0]
err = 0
tolerance = 0.03

myModel = ALS.train(trainingRDD, rank, seed = seed, iterations = iterations,
                    lambda_ = regularizationParameter)
predictedTestRDD = myModel.predictAll(testForPredictingRDD)

testRMSE = computeError(testRDD, predictedTestRDD)

print 'The model had a RMSE on the test set of %s' % testRMSE


print 'Training: %s, validation: %s, test: %s\n' % (trainingRDD.count(),
                                                    validationRDD.count(),
                                                    testRDD.count())


print trainingRDD.take(3)
print validationRDD.take(3)
print testRDD.take(3)

print 'There are %s ratings and %s movies in the datasets' % (ratingsCount, moviesCount)