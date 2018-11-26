
# Introduction

I'm a big fan of the Programming Collective Intelligence book. I think the book explains difficult things in a way that is easy to understand. And while reading it, I was thinking about personal projects I could do.  So, reading the first chapter, I got inspired to build a recommender system.
A recommender system is a software that provides suggestions that are likely to be of interest to the user. It is an important tool for many kinds of business, and two well-known companies that use it are Amazon and Netflix.
When I started to research the subject, I found different terms and approaches, and I was confused about how to perform that task. After some consideration, I decided to use the collaborative filtering technique. 

This approach has pros and cons

Pros
 * It works for many kinds of items
 * No need for feature selection, which is, sometimes, hard to do

Cons
 * New items, or unrated ones, are not recommended
 * It tends to recommend popular items
 * The user/rating matrix is sparse

#### Collaborative Filtering

To build the recommender system I'm going to use the [MovieLens](https://grouplens.org/datasets/movielens/) dataset, the one with 100,000 instances. It contains ratings of 1,682 movies by 943 users. Also, I'm going to make a User-Based Collaborative Filtering



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib_venn import venn2
from sklearn.metrics import mean_squared_error

%matplotlib inline

# Reading ratings file:

ratings = pd.read_csv('u.data', sep='\t', header=None, usecols = [0,1,2], encoding='latin-1')
ratings.columns = ['user_id', 'movie_id', 'rating']

# Reading items file:
items = pd.read_csv('u.item', sep='|',header = None, usecols = [0,1], encoding='latin-1')
items.columns = ['movie_id', 'title']

# to be easier to see the titles, let's merge the ratings and the titles
df = ratings.merge(items, on = 'movie_id')

print(items.shape)
print(ratings.shape)
```

    (1682, 2)
    (100000, 3)


The ratings data frame contains the movie ids, the ratings, and the user ids. That is the information we need. However, it would be nice to have the title of the movie in this data frame. So, we need to merge the items and ratings data frames.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>movie_id</th>
      <th>rating</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>63</td>
      <td>242</td>
      <td>3</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226</td>
      <td>242</td>
      <td>5</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>154</td>
      <td>242</td>
      <td>3</td>
      <td>Kolya (1996)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>306</td>
      <td>242</td>
      <td>5</td>
      <td>Kolya (1996)</td>
    </tr>
  </tbody>
</table>
</div>



To have an idea about the number of ratings each movie received, and the average rating, let's group the data frame by movie and display the count and mean. 


```python
# how many ratings  there are per movie
counts = df[['movie_id','rating']].groupby('movie_id').agg(['count','mean'])
counts.columns = ['count', 'mean']
counts = counts.sort_values(by=['count'])
counts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1682</th>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>814</th>
      <td>1</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>1</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



The number of ratings has an asymmetrical distribution, and more than 75% of the movies has less than 100 ratings. That means that many movies have missing values. Remember that there are 943 users.


```python
counts.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1682.000000</td>
      <td>1682.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>59.453032</td>
      <td>3.076045</td>
    </tr>
    <tr>
      <th>std</th>
      <td>80.383846</td>
      <td>0.781662</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.000000</td>
      <td>2.659600</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.000000</td>
      <td>3.161528</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>80.000000</td>
      <td>3.653428</td>
    </tr>
    <tr>
      <th>max</th>
      <td>583.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# boxplot of the number of ratings per movie
sns.catplot(y ='count', kind='box', data = counts);
```


![png](output_8_0.png)


## Similarity
So far we have loaded the data and found out that the number of ratings has an asymmetrical distribution, meaning that a lot of movies has just a few ratings. 
Now, we need to define how we can compare two users and say they have similar tastes. There are different metrics we can use as Euclidean distance, Pearson Correlation, Jaccard coefficient, Cosine distance, and others. I'm going to use the Pearson correlation because it handles the missing ratings as 'average'.

I'm going to build my user matrix in pandas and use a function to calculate the correlation. I could use numpy as well, and it is faster than pandas. However, the correlation matrix doesn't need to be computed in real time, so, I'm going to stick with pandas. One common practice is to pre-compute the correlation matrix, say daily.


```python
user_matrix = df.pivot_table(index='movie_id', columns='user_id', values='rating')
user_matrix.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
      <th>943</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 943 columns</p>
</div>




```python
# we have 1,682 movies and 943 users
user_matrix.shape
```




    (1682, 943)



When computing the similarity (or correlation), I want to find out how similar users are. My correlation matrix will have 943 rows and columns, and the cells represent the correlation between 2 users. For example, if the column is 1, and the row is 1, then we have the correlation between user 1 with himself, meaning that the correlation must be 1. In column 1, row 2, we have the correlation between users 1 and 2, which is 0.16.


```python
user_corr = user_matrix.corr()
user_corr.shape
```




    (943, 943)




```python
# Pearson correlation between the users
user_corr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
      <th>943</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>0.160841</td>
      <td>0.11278</td>
      <td>0.500000</td>
      <td>0.420809</td>
      <td>0.287159</td>
      <td>0.258137</td>
      <td>0.692086</td>
      <td>-0.102062</td>
      <td>-0.092344</td>
      <td>...</td>
      <td>0.061695</td>
      <td>-0.260242</td>
      <td>0.383733</td>
      <td>0.029000</td>
      <td>0.326744</td>
      <td>0.534390</td>
      <td>0.263289</td>
      <td>0.205616</td>
      <td>-0.180784</td>
      <td>0.067549</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.160841</td>
      <td>1.000000</td>
      <td>0.06742</td>
      <td>0.148522</td>
      <td>0.327327</td>
      <td>0.446269</td>
      <td>0.643675</td>
      <td>0.585491</td>
      <td>0.242536</td>
      <td>0.668145</td>
      <td>...</td>
      <td>0.021007</td>
      <td>-0.271163</td>
      <td>0.214017</td>
      <td>0.561645</td>
      <td>0.331587</td>
      <td>0.000000</td>
      <td>-0.011682</td>
      <td>-0.062017</td>
      <td>0.085960</td>
      <td>0.479702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.112780</td>
      <td>0.067420</td>
      <td>1.00000</td>
      <td>-0.262600</td>
      <td>NaN</td>
      <td>-0.109109</td>
      <td>0.064803</td>
      <td>0.291937</td>
      <td>NaN</td>
      <td>0.311086</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.045162</td>
      <td>0.000000</td>
      <td>-0.137523</td>
      <td>NaN</td>
      <td>-0.104678</td>
      <td>1.000000</td>
      <td>-0.011792</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.500000</td>
      <td>0.148522</td>
      <td>-0.26260</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-0.581318</td>
      <td>-0.266632</td>
      <td>0.642938</td>
      <td>NaN</td>
      <td>-0.301511</td>
      <td>...</td>
      <td>0.500000</td>
      <td>NaN</td>
      <td>-0.203653</td>
      <td>NaN</td>
      <td>0.375000</td>
      <td>NaN</td>
      <td>0.850992</td>
      <td>1.000000</td>
      <td>0.412568</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.420809</td>
      <td>0.327327</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.241817</td>
      <td>0.175630</td>
      <td>0.537400</td>
      <td>0.577350</td>
      <td>0.087343</td>
      <td>...</td>
      <td>0.229532</td>
      <td>-0.500000</td>
      <td>0.439286</td>
      <td>0.608581</td>
      <td>0.484211</td>
      <td>0.880705</td>
      <td>0.027038</td>
      <td>0.468521</td>
      <td>0.318163</td>
      <td>0.346234</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 943 columns</p>
</div>



## Finding Similar Users
Now, there is a user in my system, and I want to make a recommendation to him/her. How should I proceed?
I already have my correlation matrix, so, I know which users are similar to him. It would be better to consider in my recommendation only the users who have a correlation higher than 0 (Pearson correlation goes from -1 to 1). I could establish a threshold, for example, 0.7 or I could use the 10% (or any other percentage) most similar users to consider in my recommendation. I will do the second one. 
So, let's suppose user 10 is in my website and I want to make a recommendation to him. First I will select the 100 users most similar to user 10.


```python
# I need to recommend a movie to user 10
# First, let me find the users more similar to him 

user_corr = user_corr.sort_values(by=[10], ascending=False)

user_corr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>user_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>934</th>
      <th>935</th>
      <th>936</th>
      <th>937</th>
      <th>938</th>
      <th>939</th>
      <th>940</th>
      <th>941</th>
      <th>942</th>
      <th>943</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>0.814092</td>
      <td>0.334719</td>
      <td>0.155860</td>
      <td>-0.290185</td>
      <td>NaN</td>
      <td>0.306186</td>
      <td>0.895144</td>
      <td>0.512989</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.143256</td>
      <td>0.944444</td>
      <td>-0.087149</td>
      <td>NaN</td>
      <td>0.507937</td>
      <td>NaN</td>
      <td>0.660338</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>-0.092344</td>
      <td>0.668145</td>
      <td>0.311086</td>
      <td>-0.301511</td>
      <td>0.087343</td>
      <td>0.273987</td>
      <td>0.299751</td>
      <td>0.455825</td>
      <td>0.329956</td>
      <td>1.0</td>
      <td>...</td>
      <td>-0.083856</td>
      <td>-0.034922</td>
      <td>0.190739</td>
      <td>0.519719</td>
      <td>0.601884</td>
      <td>0.316228</td>
      <td>0.158976</td>
      <td>0.420084</td>
      <td>0.408994</td>
      <td>0.119523</td>
    </tr>
    <tr>
      <th>477</th>
      <td>0.123443</td>
      <td>0.398410</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.166667</td>
      <td>0.557086</td>
      <td>0.204124</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>-0.516398</td>
      <td>-0.066667</td>
      <td>0.219718</td>
      <td>0.408248</td>
      <td>0.196116</td>
      <td>-0.059761</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.017376</td>
    </tr>
    <tr>
      <th>636</th>
      <td>-0.466468</td>
      <td>-0.304997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>-0.280976</td>
      <td>-0.197215</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.801784</td>
      <td>0.609285</td>
      <td>0.024603</td>
      <td>-0.727607</td>
      <td>0.359425</td>
      <td>0.157378</td>
      <td>0.559017</td>
      <td>-0.662266</td>
      <td>NaN</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>729</th>
      <td>NaN</td>
      <td>0.742611</td>
      <td>-0.191741</td>
      <td>0.394771</td>
      <td>NaN</td>
      <td>0.866025</td>
      <td>0.333333</td>
      <td>0.188982</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>-1.000000</td>
      <td>-0.414781</td>
      <td>1.000000</td>
      <td>-0.533114</td>
      <td>NaN</td>
      <td>0.585239</td>
      <td>NaN</td>
      <td>0.182574</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 943 columns</p>
</div>




```python
# I will select 100 users similar to user 10. It is about 10% of the users. Note that I select 101 rows, because, I'm
# going to exclude user 10 (it is the second line of the df)
similar_users = user_corr[10][:101]
similar_users = similar_users.drop(10)
```

## Predicting Ratings

To predict the rating for a movie, I'm going to use the same approach presented in the Programming Collective Intelligence book, which is a weighted average. 


```python
# selected the 100 most similar users. 
predictions_10 = similar_users.to_frame()
predictions_10 = predictions_10.reset_index()
predictions_10.columns = ['user_id', 'correlation']
# merge the correlation and the ratings for each users
predictions_10 = predictions_10.merge(ratings, on='user_id')
```


```python
# The predicted rating is a weighted average. First I will multiply the correlation by the rating for each movie
predictions_10['w_average'] = predictions_10['correlation']*predictions_10['rating']
```


```python
# Now, I need to sum all the correlations by movie, so I'm going to group by movie and sum
user_10_corr_sum = predictions_10[['movie_id','correlation']].groupby('movie_id').agg(['sum'])
user_10_corr_sum.columns = ['correlation_sum']

# Here I'm going to sum all the ratings * correlation
user_10_rating_sum = predictions_10[['movie_id','w_average']].groupby('movie_id').agg(['sum'])
user_10_rating_sum.columns = ['rating_sum']

# Finally, I'm merging in one df the sum of the correlation, and the sum of the weighted average
user_10_pred = user_10_corr_sum.merge(user_10_rating_sum, on='movie_id')

print(user_10_pred.shape)

user_10_pred.head()
```

    (757, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>correlation_sum</th>
      <th>rating_sum</th>
    </tr>
    <tr>
      <th>movie_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>20.227836</td>
      <td>73.787894</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.471337</td>
      <td>12.062676</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.264603</td>
      <td>4.492309</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.792119</td>
      <td>6.559369</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.583507</td>
      <td>2.334027</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the predicted score 
user_10_pred['pred_score'] = user_10_pred['rating_sum']/user_10_pred['correlation_sum']
user_10_pred = user_10_pred.reset_index()

user_10_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>correlation_sum</th>
      <th>rating_sum</th>
      <th>pred_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20.227836</td>
      <td>73.787894</td>
      <td>3.647839</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>4.471337</td>
      <td>12.062676</td>
      <td>2.697779</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1.264603</td>
      <td>4.492309</td>
      <td>3.552346</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>1.792119</td>
      <td>6.559369</td>
      <td>3.660119</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.583507</td>
      <td>2.334027</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.distplot(user_10_pred['pred_score']);
```


![png](output_23_0.png)


### Quick Recap

Our purpose is to make recommendations to user 10. We have 757 predicted ratings. However, we can't recommend a movie that he has already seen. Also, we don't have predictions for all the 1,682 movies. Let's see how many and which movies user 10 has watched. 


```python
# First, we gather user's 10 preferences
user_10_prefs = ratings.loc[ratings['user_id']==10]
# we know the user_id, it is not necessary
user_10_prefs = user_10_prefs.drop(['user_id'], axis=1)
print(user_10_prefs.shape)
user_10_prefs.head()
```

    (184, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>16</td>
      <td>4</td>
    </tr>
    <tr>
      <th>158</th>
      <td>486</td>
      <td>4</td>
    </tr>
    <tr>
      <th>386</th>
      <td>175</td>
      <td>3</td>
    </tr>
    <tr>
      <th>544</th>
      <td>611</td>
      <td>5</td>
    </tr>
    <tr>
      <th>606</th>
      <td>7</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



User 10 has seen 184 movies, and we have predictions for 128 of them. We can use this set later to evaluate our predictions.


```python
watched_movies = user_10_prefs['movie_id'].tolist()
predicted_movies = user_10_pred['movie_id'].tolist()
w_p = set(watched_movies).intersection(predicted_movies)
print(len(w_p))
```

    128



```python
venn2(subsets = (757, 184, 128), set_labels = ('Predict','Watch','Predicted_Watched'));
```


![png](output_28_0.png)



```python
# we recommend movies that he/she hasn't already seen
user_10_recom_mov = user_10_pred.loc[~user_10_pred.index.isin(watched_movies)]
user_10_recom_mov = user_10_recom_mov.drop(['correlation_sum','rating_sum'], axis=1)
user_10_recom_mov = user_10_recom_mov.merge(items, left_on = 'movie_id', right_on = 'movie_id')
user_10_recom_mov = user_10_recom_mov.sort_values(by=['pred_score'], ascending=False)

print(user_10_recom_mov.shape)
user_10_recom_mov.head()
```

    (573, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>pred_score</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>132</th>
      <td>251</td>
      <td>5.0</td>
      <td>Shall We Dance? (1996)</td>
    </tr>
    <tr>
      <th>11</th>
      <td>22</td>
      <td>5.0</td>
      <td>Braveheart (1995)</td>
    </tr>
    <tr>
      <th>105</th>
      <td>197</td>
      <td>5.0</td>
      <td>Graduate, The (1967)</td>
    </tr>
    <tr>
      <th>286</th>
      <td>519</td>
      <td>5.0</td>
      <td>Treasure of the Sierra Madre, The (1948)</td>
    </tr>
    <tr>
      <th>518</th>
      <td>1144</td>
      <td>5.0</td>
      <td>Quiet Room, The (1996)</td>
    </tr>
  </tbody>
</table>
</div>



The user_10_recom_mov data frame contains the movies ordered by predicted rating, so, we can recommend the first 5 or 10 movies. Alternatively, we can suggest the movies whose predicted rating is 5.

## Evaluation


```python
# we have the movies and the ratings. Now, let's add the predictions
user_10_prefs = user_10_prefs.merge(user_10_pred, on='movie_id')
user_10_prefs = user_10_prefs.drop(['correlation_sum','rating_sum'], axis=1)
print(user_10_prefs.shape)
user_10_prefs.head()
```

    (128, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>rating</th>
      <th>pred_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>4</td>
      <td>3.275037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>175</td>
      <td>3</td>
      <td>1.984676</td>
    </tr>
    <tr>
      <th>2</th>
      <td>611</td>
      <td>5</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>4</td>
      <td>3.801467</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>5</td>
      <td>4.542819</td>
    </tr>
  </tbody>
</table>
</div>




```python
# evaluate the predictions - rating is the actual rating, pred_score is the prediction

error = mean_squared_error(user_10_prefs['rating'],user_10_prefs['pred_score'])

print(error)
```

    0.3326790492597428


Below you can see a plot of the actual ratings and the predicted ones.


```python
sns.swarmplot(x='rating', y="pred_score", data=user_10_prefs);
```


![png](output_35_0.png)

