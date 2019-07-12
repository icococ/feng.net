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

