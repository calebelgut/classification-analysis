# *Classification Analysis of Credit Default*
**Caleb Elgut - August 2020**

# Introduction

I built a series of 15 classification models to predict whether someone will default on their credit payment next month or make their payment. After creating the model and investigating the predictive results, I decided on two primary questions to analyze:

1. What is the baseline probability that someone will default on their credit?
1. What are the most important features to consider when predicting whether someone will default on their next credit payment. 

The data I used came from the University of Irvine's Machine Learning Database and contains 30,000 rows. The dataframe's original source was from a reputable bank in Taiwan (a cash and credit issuer) and the data was collected between April and September 2005. The original purpose of the data collection was to predict risk as a credit crisis was impending in Taiwan at this time. 

The data includes a series of predictive features that are both categorical and continuous as well as a target variable that measured whether or not someone would default on their credit payment in the next month (this was measured in binary classification--0 for will pay and 1 for will default).

The following Python packages were used in the analysis of my data:
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-Learn
- StatsModels
- Warnings
- Datetime

# Who are the stakeholders?

Banks that are interested in risk prevention when giving out credit to customers.

# What are we here for?

The two primary questions that guided my research: 

1. What is the basic probability of defaulting on one's next credit payment according to this dataset?
1. What are the most important features to examine when predicting whether or not someone will default on their next payment?

# The Makings of the Models

The most important piece of this project is the creation of the classification models that powered the predictions in my questions; therefore, they will be the center of discussion here however I will also produce visualizations that reflect the answers to my above questions. Additionally, even though I created 15 models I will only write in detail about the 5 most effective here. If you would like to see the rest you can check out the jupyter notebook attached in the repository. 

**NOTE:** You are about to see the list of all 15 models created for this project. I would like to explain a few terms that you will encounter in this list:

- *Untuned* = No changes were made to the hyperparameters in the model. Hyperparameters are parameters that are native to the model, itself. 
  - An example of a hyperparameter is the Max Depth of a Decision Tree--> This determines the number of nodes from the root down to the furthest leaf node. In other words, it is the height of a binary tree.

- *Original Dataset* refers to the original dataframe. After determining the top ten most important features related to predicting one's class of either default or pay for the next month, I created a separate dataframe that only held these features. This is what I mean when I refer to the *Top Ten Features* below.

- *Manually Tuned* will be explained in further detail later but, suffice it to say, this refers to a process of tuning each hyperparameter in a model individually. This in contrast to a method such as *GridSearch* which will also be explained later in full detail but, suffice it to say, this is a method that allows one to discover the ideal hyperparameter measurements without having to tune each parameter individually. 

# Complete List of Models

1. Logistic Regression
1. K-Nearest-Neighbors (Untuned, Original Dataset)
1. K-Nearest-Neighbors with "Best K" (Original Dataset)
1. Decision Tree #1 (Untuned, Original Dataset)
1. Decision Tree #2 (Max Depth: 6, Original Dataset)
1. Decision Tree #3 (Manually Tuned, Original Dataset)
1. Decision Tree #4 (Grid-Searched, Original Dataset)
1. Decision Tree #5 (Untuned with Top Ten Features)
1. Decision Tree #6 (Grid-Searched with Top Ten Features)
1. Bagging #1 (Original Dataset)
1. Bagging #2 (Top Ten Features)
1. Random Forests #1 (Untuned, Original Dataset)
1. Random Forests #2 (Grid-Searched, Original Dataset)
1. Random Forests #3 (Untuned, Top Ten Features)
1. Random Forests #4 (Grid-Searched, Top Ten Features)

# Initial Analysis

## Column Names

I later rename some of these columns but here are the names they started with:

We began with:
1. ID: The ID Number of each unique row
1. LIMIT_BAL: Each individual's credit limit.
1. SEX: 1(Male) or 2(Female)
1. MARRIAGE: 1 = Single, 2 for Married, 3 for Divorced, 9 & 4 for "Other"
1. AGE: Number denoting how old someone was. 
1. PAY_0,2,3,4,5,6 = Payment Status from September through April (backwards). 
  - Values: -2 for no credit use this month, -1 for someone who is paid up, 0 for revolving credit users, 1-9 for the number of months one is behind in their credit payments. 
  - **I found it strange that they skipped 1, here. Later on I will add the month names in**
1. Bill_AMT1,2,3,4,5,6 = The Account Balance for that month. (Sep - April)
1. PAY_AMT1,2,3,4,5,6 = Amount paid for each month
1. Default Payment Next Month: 0 = Will Pay, 1 = Will Default

![Initial Dataset](/readme_images/original_dataset.jpg)


## Distribution of Classes 

After clearing out the single null value that existed as a result of an additional row accidentally added when converting the dataframe from an .xslx to a .csv, I examined the distribution between those likely to pay next month v. those likely to default.

![Distribution Code](/readme_images/distribution_1.jpg)

![Distribution Visualization](/readme_images/distribution_2.jpg)

One thing I immediately notice is that there is an issue of *class imbalance* that I will need to take into account later on when creating models.

## Credit Limit Difference

Next, I examined the difference in credit limit between those who will pay and those who will default. I found the following information:

1. The maximum credit limit among those who will pay next month is 25% higher ($1M vs. $250K). 
1. The 3rd quartile is 20% higher among those who will pay ($250K vs. $200K)
1. There is a 60% higher median among those who will pay ($150K vs. $90K)
1. There is a 40% higher 1st quartile among those who will pay ($70K v. $50K)
1. The minimum credit limit is the SAME. $10K **(We can assume this is the starting credit limit for everyone in this bank).**

## Describing the Variables

1. The majority of sex = 2 (Female)
1. The median and 3rd quartile of education = 2 (Undergraduate)
1. The median and 3rd quartile of marriage = 2 (Married)
1. **Regarding Age:**
  - Minimum: 21
  - 1st Quartile: 28
  - Median Age: 34
  - 3rd Quartile: 41
  - Maximum Age: 79
2. **Regarding Payment Status:**
  - In the 1st quartile of each month: people are paid in full.
  - The median and 3rd quartile reflect people using revolving credit
  - The maximum in each month are folks who are 8 months late in payments
3. **Regarding Credit Balance in Each Month:**
  - September has the highest values in nearly each category
    - 1st Quartile: $3558
    - Median: $22,381
    - 3rd Quartile: $67,091
    - Maximum: Only category where September does not have the highest value
      - July has the highest at $1,664,000
  - Examining the minimum, it seems there are negative values. We can assume that, perhaps, this is an amount the bank owes the individual? That or it is a clerical error. 

4. **Regarding the Amount Paid Each Month**
  - To nobody's surprise: Minimum for each month = 0
  - 1st Q for each month:
    - Sep: 1000
    - August:833
    - July: 390
    - June: 296
    - May: 252
    - April: 117.75
  - Median for each month:
    - Sep: 2100
    - August: 2009
    - July: 1800
    - June: 1500
    - May: 1500:
    - April: 1500
    - **THOUGHT**: Perhaps 1500 is the mandatory minimum for the majority of folks?
  - 3rd Q for each month:
    - Sep: 5006
    - Aug: 5000
    - July: 4505
    - June: 4013.25
    - May: 4031.5
    - April: 4000
  - Max for each month:
    - Sep: 873552 (Each of these seem like folks paying off a whole debt?)
    - Aug: 1.68M
    - July: 896040
    - June: 621000
    - May: 426529
    - April: 528666

## Correlation Chart

I then created a correlation chart to gain an initial understanding of how the variables related to one another. Since multicollinearity does not matter as much with classification problems as it does with regression, I would not have to make many adjustments in the face of this correlation. 

![Heat Map for Correlation](/readme_images/initial_correlation.png)

Not much to note here. Unsurprisingly, columns related to the account balance correlate with each other positively. Columns related to payment status correlate with each other a bit less so. Moderately negative correlation between payment status and credit limit.

## Rename Columns 

Column names were changed to reflect the months to which they were relevant. 

![Name Change](/readme_images/name_change.jpg)

## Predicted Feature Importance?

At the start, I predicted that credit limit, education, and month_status will be of most importance. 

## Are false positives or false negatives more of a concern?

It depends on the bank's goals. Since 1 denotes someone who will default and 0 denotes someone who will pay, I will define "false positive" as the model denoting someone as defaulting when, in fact, they would make their next payment. A "false negative" then would be someone who is denoted as one who would make their next payment when, in reality, they will default. False negatives could lead to loss for the bank and false positives can lead to annoyed customers. While of course, in an ideal world, our model will predict perfectly, bankers would be happy with a model that is more likely to give false positives than false negatives in this case. 

# Given the data that we have, what is the basic probability that someone will default on their payment? 

## Answer
- Head to the column 'default payment next month', sum up the 1s, divide by the length. 
- Our Answer: **22%**
  - *This answers our first question*

![Percent Default](/readme_images/percent_default.jpg)

# Some further preprocessing

Before I could create classification models to address the question of feature importance, I needed some additional preprocessing. 

## One-Hot Encoding the Categorical Variables

To one-hot encode a categorical variable is to create a new column that classifies a single categorical variable as a 1 if an individual fits the category or a 0 if the individual does not. Before I did this, I wanted to bin my age groups by decade. 

After one-hot encoding, I renamed each column to reflect its true value (for example: marriage_1.0 became "Single" ) 

The new resulting columns totalled to 39 and included such new columns as:
1. male
1. female
1. grad_school
1. undergrad
1. high_school
1. single
1. married
1. divorced
1. age_by_decade_20s
and more! 

**The Final Number of Columns was 39 including the target variable.**

# Time for some models

I have chosen some of the highest performing models to break down and explain. The models are as follows:

1. Logistic Regression
1. A Manually Tuned Decision Tree (On the original dataset)
1. A Grid-Searched Decision Tree (On the original dataset and the Top Ten)
1. An untuned Random Forest (On the original dataset)
1. A Grid-Searched Random Forest (On the original dataset and the Top Ten)

## Splitting the data 

Before we fit the data to any models we must split it into training and testing sets. We begin by separating the target variable from the predictors and then using a function called *train_test_split* to split our predictors and our target into training and testing sets. The ratio was 80% training to 20% testing. I also placed this split (and all models) into the random_state of 42. Setting the splits and the modeling into random_state means that when the data is arranged "randomly" it is randomly mixed in the same way each time the code is run. **This helps with reproducability which is essential in any science.**

![Split Data](/readme_images/split_the_data.jpg)

## The first model I try is a logistic regression model

Logistic Regression uses a sigmoid function to plot an "s-like" curve that enables a linear function to act as a binary classifier. **We won't be looking at the s curve here, though. We will be focusing on the results of a *confusion matrix.***

A Confusion Matrix visualizes the performance of a classification model. It shows accuracy as well as your model's ratio of *precision* to *recall* (terms we will explain a bit later). 

The Logistic Regression Model is fit to the data:

![log_reg_one](/readme_images/log_reg_one.jpg)

- The C value prevents overfitting
- The fit_intercept is False, meaning the y-intercept is set to zero (good for when classifying as it can mess with regularization)
- solver = 'liblinear' applies L1 Regularization and is helpful with high-dimension data. 
  - L1 Regularization shrinks the less important feature’s coefficient to zero. This performs estimation and selection simultaneously for us. 

![log_reg_one_conf_mat](/readme_images/log_reg_one_conf_mat.jpg)

- The results of this initial model appear useless! According to the confusion matrix you see above, this model never assumes someone will default on their credit. It only guesses 0s. 
- The matrix above has four quadrants: The top left represents *true negatives*, the top right represents *false positives*, the bottom left represents *false negatives* and the bottom right represents *true positives.*
  - 0 False or True Positives because this model does not guess a single person defaults on their credit! 
- Below the matrix you see the **classification report** this gives us info regarding the precision, recall, f1-score, and accuracy of each model we create (most of these terms will be defined in just a moment). 

To verify whether or not this model is actually useless we need to analyze the *null accuracy.*

## What is the Null Accuracy? 

The Null Accuracy is a baseline measurement from which to measure all future models. It is **the accuracy that could be achieved by always predicting the most frequent class.**

The null accuracy of a binary classifier is found by taking the maximum between either the mean or 1 minus the mean. By making this calculation we come up with 0.7812. Because we know that the majority class is **0** (folks who pay), *if our classifier guesses the most frequent class it will be correct 78% of the time. **This must be considered when we examine further classifiers.** It is why I put a focus on **F1-Score, Recall, Precision, and AUC.**

![null accuracy](/readme_images/null_accuracy.jpg)

## Ok, Let's Explain Recall, Precision, F1, and AUC

Recall and Precision are countermeasurements. If one increases, the other tends to decrease. Let's imagine an example where the instances are documents and the task is to return a set of relevant documents given a search term. Recall is the **number of relevant documents retrieved by a search divided by the total number of existing relevant documents**, while precision is the **number of relevant documents retrieved by a search divided by the total number of documents retrieved by that search.** 

Using our dataset, higher recall means we are going to assume that more people will turn up positive for credit default next month than who will actually default. This gives us a higher rate of false positives. 

Higher precision would mean that our threshold for placing someone in the category of "will default" increases which means that for every individual our model classifies as likely to default, we have a high likelihood that our model is guessing correctly (high rate of true positives) however we also may have such a high threshold for who will default that our model misses others who will, in actuality, default. This will give us a higher rate of false negatives--folks who we classify as likely to pay next month but who, in reality, will default. **High precision has a likelihood of costing banks more money than high recall.**

F1 is a datapoint that reflects the harmony between recall and precision. 

Finally, AUC is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve
  - The ROC curve shows the trade-off between sensitivity (or TPR) and specificity (1 – FPR). Classifiers that give curves closer to the top-left corner indicate a better performance.
    - ROC does not depend on class distribution so it is helpful with cases of class imbalance.
    - We seem to have class imbalance in our dataset as there are far more people who will pay next month than who will default next month.

## Back to the Logistic Regression

For our above regression, I adjusted the *classification threshold* to see if the model would respond by predicting some 1s. 
  - The Classification Threshold is generally set to 0.5 which does not help with a class imbalance problem. 
  - After some tuning, the appropriate classification threshold was 0.3 (0.2 gave us a very high recall but a precision of below 30%). 

![The New Classification Threshold](/readme_images/new_class_thresh_log_reg.jpg)

As we can see here, the classification threshold change gave us good results. Our model is now predicting some credit default (1) values. The classification report is below. 

![Log Reg 2 Classification Report](/readme_images/log_reg_2_class_rep.jpg)

While the F1 score of 0.41 is not the highest of the models you will see here, the **recall score *is* the highest at *0.62.*** The precision is also the lowest of the models at 0.3. *A lower AUC of 0.656 also appears below.* 

![Log Reg ROC Curve & AUC](/readme_images/auc_curve_log_reg.png)

# On to Decision Trees

## What is a Decision Tree?

Imagine if you want to decide whether or not you will play golf based on the weather's various conditions. Certain weather conditions are more important to you than others. The weather category is the root node: Sunny, Overcast, or Rainy. Yes/No will determine whether or not the tree continues. If the weather is rainy then you reach a leaf node of NO. If it is Sunny or Overcast perhaps then you want to know if it windy. If it is not windy and overcast, not windy and sunny, or windy and sunny the result is YES but if the result is overcast and windy the answer is NO. This is how a Decision Tree works! 

Two of the most important terms related to decision trees are **entropy and information gain** Entropy measures the impurity of the input set. Information is a *decrease in entropy.*  

When I refer to the *impurity* of a data set here is what I mean: If you have a bowl of 100 white grapes you know that if you pluck out a seed at random you will get a white grape. Your bowl has purity. If, however, I remove 30 of the white grapes and replace them with purple grapes, your likelihood of plucking out a white grape has decreased to 70%. Your bowl has become impure. The entropy has increased. 

As each split occurs in your decision tree, entropy is measured. The split which has the lowest entropy compared to the parent node and other splits is chosen. The lesser the entropy, the better. 

## Decision Trees with Our Data

Originally, I created an untuned Decision Tree with the original dataset. This did not give me results worth explaining here. Here is what the tree looked like, though!

![DT Untuned](/readme_images/decision_tree_untuned.png)

My next step was to **tune the hyperparameters**

Hyperparameters are parameters whose values are set before the learning process begins. In Decision Tree Models, the relevant hyperparameters are **criterion, max depth, minimum samples leaf with split, minimum leaf sample size, and max features**

1. Criterion: Entropy or Gini. Different measures of impurity. Not a huge difference between each. 
1. Maximum Depth: Reduces the depth of the tree to build a *generalized tree.* This is set depending on your need.
1. Minimum Samples Leaf with Split: Restricts *size* of sample leaf
1. Minimum Leaf Sample Size: Size in terminal nodes can be *fixed*
1. Maximum Features: Max number of features to consider when splitting a node.

I won't bore you further with the details of the code that powers the manual tuning but suffice it to say, many an array and for loop were created to train our decision tree for many different values of each hyperparameter. The resulting graphs are below: 

![Max Tree Depth](/readme_images/max_depth.png)
![Minimum Samples Split](/readme_images/min_sample_split.png)
![Minimum Leaf Sample Size](/readme_images/min_sample_leaf.png)
![Max Features](/readme_images/max_features.png)

My max depth was 11, max features was 22, min samples split was 0.2 and the min samples leaf was 0.2. I chose a criterion of entropy.

**The resulting confusion matrix, classification report, and AUC are below:**

![Decision Tree Manually Tuned](/readme_images/decision_tree_manual_tuned.jpg)

![Decision Tree Man_Tun AUC](/readme_images/decision_tree_manual_tuned_auc.png)

### Great Numbers!
- This F1 of 0.51 is the highest F1 I would get among all the models. 
- This AUC of 0.733 was far greater than the untuned AUC of 0.609. 
- The precision and recall are nearly even at 0.50 and 0.52 respectively.

Here is what the tree looks like, all tuned and clean!

![DT Man_Tun Image](/readme_images/decision_tree_manual_tuned_image.png)

At this point, I wanted to see what this model saw as important. I created a chart of important features. 

![DT Man_Tun Import Feats](/readme_images/dt_man_tune_import_feats.png)

This shows a MASSIVE importance for the payment status in September but such a low readout for the rest. There needed to be a way to gain clearer information. 

# Enter Random Forests 

## Imagine creating MANY decision trees!

That's what a random forest is! It is an ensemble of decision trees. It essentially creates a series of decision trees based on the dataset with each tree being different. Each decision tree makes choices that maximize information. With a diverse series of trees, I have an opportunity to have a model that gives me even more information than the single tree I created. 

Random Forests are also very resilient to overfitting--our random forest of diverse decision trees are trained on different sets of data and looks at different subsets of features to make predictions--> For any given tree there is room for error but odds that **every tree** will make the *same* mistake because they looked at the *same* predictor is infinitesimally small!

The first Random Forest I created on the *original dataset* was *untuned* 

**The resulting confusion matrix, classification report, and AUC are below:**

![RF Untuned Conf Mat Class Rep](/readme_images/rf_untuned_conf_mat.jpg)
![RF Untuned AUC](/readme_images/rf_untuned_auc.png)

F1: 0.43, Precision: 0.68, Recall: 0.31, AUC: 0.771

Before heading on to my next models I checked an updated list of the feature importances to see if the random forest could be more informative than my single manually-tuned decision tree:

![RF Untuned Feat Import](/readme_images/rf_untuned_feat_import.png)

- Similar to the earlier basic decision tree analysis--One's payment status in September is still paramount. 
  - However, this new chart reveals that the payment status in August is more important than we intially realized. This seems to be leading somewhere. Time to dig deeper!

# GridSearch: The Better Way to Hypertune

- GridSearch is a function that tries every possible parameter combination that you feed it to find out which combination of parameters will give you the best possible score. 
  - GridSearch combines K-Fold CrossValidation with a grid search of the parameters.
- I GridSearched to find optimal parameters for both my Decision Tree and Random Forest on the original dataset. 

## Grid-Searched Decision Tree Results
- The optimal decision tree parameters for this dataset are as follows:
  - Criterion: Gini
  - Max Depth: 3
  - Min Samples Leaf: 1
  - Min Samples Split: 2

**The resulting confusion matrix, classification report, and AUC are below:**

![DT Grid Search OG](/readme_images/dt_gridsearch_OG_cmcr.jpg)
![DT GS OG AUC](/readme_images/dt_gridsearch_OG_AUC.png)

F1: 0.47, Precision: 0.67, Recall: 0.36, AUC: 0.727

The Feature Importance Chart attached to this model gives no new information. September and August's status are still the most important. 

## Grid-Searched Random Forest Results
- The optimal random forest parameters for this dataset are as follows:
  - Criterion: Gini
  - Max Depth: None
  - Min Samples Leaf: 6
  - Min Samples Split: 10
  - n_estimators: 100
  
**The resulting confusion matrix, classification report, and AUC are below:**

![RF Grid Search OG](/readme_images/rf_gridsearch_OG_cmcr.jpg)
![RF Grid Search OG AUC](/readme_images/rf_gridsearch_OG_AUC.png)

F1: 0.46, Precision: 0.67, Recall: 0.35, AUC: 0.775

![RF Grid Search OG FI](/readme_images/rf_gridsearch_OG_FI.png)

The above feature importance chart is INCREDIBLY INFORMATIVE! 

- Status of Payment from September is still paramount
  - However now instead of being in the range of above 0.25 it is now close to 0.12
  - The amount one pays each month and the balance have increased in importance, however status of payment in August & September are still the best predictors.
  - Most notable third place: july_status
  
# Analysis Focusing on Important Features

## Top Ten Time! 

I could have halted my analysis above. After all, I received what appears to be the best I can do in terms of an F1 score & recall and I had a decent amount of information in terms of feature importance. However, I wanted to dig a bit more deep into the data and therefore I **created a new dataframe that included the top ten predictors**

The top ten features were the status, monthly payment, and account balance from the past three months (September, August, and July) as well as the credit limit. I came to decide on these features based on the results of the above feature importance charts as well as conversations I had with people who worked in banking, particularly in lending. They explained that recent payment history and information tell a far better story than older information. 

![Top Ten Split](/readme_images/top_ten_split.jpg)

The two models from this analysis that I will focus on are the GridSearched Decision Tree and the GridSearched Random Forest

## GridSearch Decision Tree with Top Ten Features

### Grid-Searched Decision Tree Hyperparameter Results
- The optimal decision tree parameters for this dataset are as follows:
  - Criterion: Gini
  - Max Depth: 3
  - Min Samples Leaf: 1
  - Min Samples Split: 2


![DT GridSearch Top Ten CM & CR](/readme_images/dt_gridsearch_top_ten.jpg)

![DT GridSearch Top Ten AUC](/readme_images/dt_gridsearch_top_ten_AUC.png)

F1: 0.43, Precision: 0.70, Recall: 0.31, AUC: 0.726

![DT GridSearch Top Ten Feature Importance](/readme_images/dt_grid_search_top_ten_fi.png)

From what we can tell here, the two most recent months' payment status seem to be of most importance even when we whiddle down the data to 10 features. Let's try some GridSearched Random Forests and call it a day! 

## Random Forests with Top Ten Features

### Grid-Searched Random Forest Results
- The optimal random forest parameters for this dataset are as follows:
  - Criterion: Gini
  - Max Depth: 6
  - Min Samples Leaf: 6
  - Min Samples Split: 5
  - n_estimators: 10
  
![RF GridSearch Top Ten Feature CM & CR](/readme_images/rf_gridsearch_top_ten.jpg)

![RF GridSearch Top Ten Feature AUC](/readme_images/rf_gridsearch_top_ten_AUC.png)
F1: 0.44, Precision: 0.68, Recall: 0.33, AUC: 0.772

![RF GridSearch Top Ten Feature Importance](/readme_images/final_feature_importance.png)

Well, there we have it! The past three months' payment status are the most important features when determining whether or not someone will default on their next credit payment.

Before we go, here are two final images: An individual decision tree from our top ten features and a random forest with 5 trees

![An Individual Decision Tree for Top Ten](/readme_images/individual_tree_top_ten.png)

![A Random Forest for Top Ten](/readme_images/random_forest_final.png)

# Conclusion & Recommendations:

- A manually tuned Decision Tree from the original dataset gave us the best results in terms of F1
  - Pros for a single decision tree:
    - Less computationally complex than a random forest
    - This model, in particular, is tuned to what seems to be maximum performance given the dataset.
      - F1 at 0.51 means that there is moderate harmony between the precision and the recall. This means that a moderate amount of results are returned with a moderate amount of those results labeled correctly.
- Since we are pitching our model to bankers & loan officers, however, it seems that leaning towards a higher recall and a lower precision will be appropriate.
  - In this case, our best model would be the logistic regression model.
    - This model's high recall and low precision means that there our model will predict more people will default on their loans than exist in reality. The lower precision means our model may not be very accurate per person however, in the long run, more false positives could save a bank/lender money than more false negatives.
- After speaking with a banker friend, he said that in the interest of ethics as well as with advancements in tech, high precision can be very valuable as well.
  - While a higher recall can save the bank money, it can also turn away valuable business partners and, with a high enough recall and a low enough precision, your bank can gain a reputation for not having a solid predictive model and you will lose business.
    - I used to experience this myself when my bank would too often categorize charges I would make abroad as fraudulent even though I told my bank exactly where I would be.
  - In the case that a bank would want a model with a higher precision, I would recommend using The grid-searched decision tree or random forest
Both models perform similarly however it will depend on how much computational complexity one could handle.
    - If one is looking for a more complex and robust model, GridSearch Random Forest is the way to go.
- Finally, regarding features to focus on:
  - **The payment status within the last 2-3 months seems, by far, to be the most likely to predict whether or not one will default on their credit payment next month**.
