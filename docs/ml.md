## Teaching computers to learn to perform tasks from past experiences
> for computers past experiences are just recorded as data

* Two most common error function of linear regression are 
  > the mean absolute error ()
  > the mean squared error 

多变量分析
* 热图 heatmap， 热图可以用颜色变化反映变量之间的相关性
* corr方法

决定系数R2 衡量模型预测能力好坏的标准








### Pandas

[10 minutes to pandas](http://pandas.pydata.org/pandas-docs/version/0.24.0/getting_started/10min.html)

[Pandas Cookbook](http://pandas.pydata.org/pandas-docs/version/0.24.0/user_guide/cookbook.html)

## Neural Network
Feed forwarding is literally composing a bunch of functions
Back propagation is literally taking the derivative at each piece
Taking the derivative of a composition is the same as multiplying the partial derivatives


Entropy measures precisely how much freedom does a particle have to move around


### 模型3

**模型名称**

回答：RandomForestClassifier


**描述一个该模型在真实世界的一个应用场景。（你需要为此做点研究，并给出你的引用出处）**

回答：Random forest has a variety of applications, such as:
* recommendation engions
* Image classification 
* classify loyal loan applications
* predict patients for high risks
* predict parts failures in manufacturing
* predict loan defaulters

**这个模型的优势是什么？他什么情况下表现最好？**

回答：
* Highly accurate and robust
* Run efficiently on large data bases
* Handles thousands of input variables without variable deletion
* Gives estimates of what variables are important in the classification
* Generates an internal unbiased estimate of the generalization error as the forest building progresses
* Provides effective methods for estimating missing data
* Maintains accuracy when a large proportion of the data are missing
* Provides methods for balancing error in class population unbalanced data sets
* Generated forests can be saved for future use on other data
* Prototypes are computed that give information about the relation between the variables and the classification.
* Computes proximities between pairs of cases that can be used in clustering, locating outliers, or (by scaling) give interesting views of the data
* Capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection
* Offers an experimental method for detecting variable interactions
* Not suffer from the overfitting problem

**这个模型的缺点是什么？什么条件下它表现很差？**

回答：
* Time-consuming, all trees in the forest have to make a prediction and then perform voting on it 
* Difficult to interpret compared to decsion tree

**根据我们当前数据集的特点，为什么这个模型适合这个问题。**

回答：数据量较大

## KMeans 

* K-means algorithm aims to choose centroids that minimise the inertia, within-cluster sum-of-squares criterion

* Drawbacks:
  + convex and isotropic clusters. It responds poorly to elongated clusters or manifolds with irregular shapes
  + 

### Manifold learning is an approach to non-linear dimensionality reduction.

To aid visualization of the structure of a dataset, the dimension must be reduced in some way.
The simplest way to accomplish this dimensionality reduction is by taking a random project of the data.
liner dimensionality reduction frameworks have been designed, such as Principal Component Analysis, Independent Component Analysis, Linear Discriminant Analysis and others.
Manifold learning can be thought of as an attempt to generalize linear frameworks like PCA to be sensitive to non-linear structure in data.
Isomap algorithm is one of the earliest approaches to manifold learning.  <br/>
Isomap can be viewed as an extension of Multi-dimensional Scaling or Kernal PCA. Isomap seeks a lower-dimensional embedding which maintains geodesic distances between all points. <br/>

The Isomap algorithm comprises three stages:
* Nearest neighbor search, Isomap uses sklearn.neighbors.BallTree for efficient neighbor search. 
* Shortest-path graph search, Dijkstra's algorithm costs O[N2(k+log(N))] or the Floyd-Warshall algorithm.
* Partial eigenvalue decomposition

## Locally Linear Embedding

LLE seeks a lower-dimensional projection of the data which preserves distances within local neighborhoods. It can be thought of as a series of local Principal Componment Analyses which are globally compared to find the best non-linear embedding.
* Nearest Neightbors Search
* Weight Matrix Construction 
* Partial Eigenvalue Decomposition 
  
### Modified Locally Linear Embedding
### Hessian Eigenmapping

### Spectral Embedding

### PCA for facial Recognition

In the literature PCA also called eigenfaces when it applied to facial recognition.

## Dimensionality Reduction

Random Projection,  
* Computionally more efficient than PCA
* Commonly used in cases where dataset has too many dimensions for PCA to be directly computed
* application running on a system with limited computational resources
* Johnson-Lindendstrauss lemma
  + A dataset of N points in high-dimensional Euclideanspace can be mapped down to a space in much lower dimension in a way that perserves the distance between the points to a large degree
  + it can be mapped that's just multiplying by this random matrix
  + it can be done in a way that preserves the distances between the points to a large degree
  + the distance between each pair of points in this dataset after projection that is preserved in a certain way

* In most of supervised and unsupervised learning, the algorithms really care about the distances between the points
* these distances will be distorted a little bit but they can be preserved
* How can they be preserved?

Independent component analysis, ICA
* cocktail party
* Blind source separation and the problem that ICA solves
* The classic example used to explain ICA is something called the cocktail party problem
* Independent component analysis: Algorithms and Applications
FastICA algorithm
1. center, white X
2. choose initial random weight matrix W1,..Wn
3. Estimate W, containing Vectors
4. Decorrelate W
5. repeat from step 3 until converged
   
Hyper-tangent function


## Plotting 
* 对于数据中的每一对特征构造一个散布矩阵
```python
pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(14,8), diagonal='kde')
```

## Gradient Descent
Error function is a function of the weights. The gradient of E is given by the vector sum of the partial derivatives of E with respect to W1 and W2. We'll take the negative of the gradient of the error function at that point.
**The gradient of the error function is precisely the vector formed by the partial derivatives of the error function with respect to the weights and bias. **

$\sigma\prime(x)=\sigma(x)(1-\sigma(x))\hat{y}$



## Words
* Compression factor
* Delve deep into the math

