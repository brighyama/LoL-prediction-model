# Predicting Game Length of 2022 League of Legends Esports Matches using Early-Game Data

by Brighten Hayama (bhayama@ucsd.edu)

Our exploratory data analysis on this dataset can be found [here](https://brighyama.github.io/LoL-data-analysis/)

## Framing the Problem

We aim to predict the game length of competitive 2022 League of Legends matches, primarily using early-game (first 10-15 minutes) data. This is clearly a regression problem, as we wish to make a quantitative prediction. 

The information we will have after the first 15 minutes of a match will be enough for our model to predict the total gamelength. This means we should be able to have some idea of the outcome of a match before it ends.

The response variable in this case is the total game length. This was chosen because we care about evaluating the competitiveness of esports matches from our original EDA. If we see that early-game data, such as gold difference between each team at 15 minutes, has high correlation to the length of a particular match, then we will be able to answer our original question of whether or not gold difference in the early-game serves as a 'good' metric of competitiveness. It would also be interesting to see the accuracy of a regression model in the context of this problem because of the high-degree complexity of any single League of Legends game. 

In order to evaluate our prediction model, we use the correlation coefficient R², which generally allows us to tell how well our model predicts the outcome of the response variable. This is also a good metric for preventing the issue of overfitting our model, as we will get lower R² scores for models that contain too many predictors and random noise.

## Baseline Model

Our early prediction model used simple linear regression, and incorporated `golddiffat15` data, or the difference in gold between two opposing teams at the 15 minute mark. Because our dataset contains a pair of symmetric datapoints for each unique match, we transform the `golddiffat15` by its absolute value, to measure the relative distance or lead one team has over another. 

The second feature we use is `comeback_at_15`, a boolean column we added to the original dataset based on whether or not a team that had negative gold difference at 10 minutes ended up with positive gold difference at 15 minutes. Intuitively, it would seem that a match where one team maintained a lead in gold between 10 to 15 minutes would have shorter overall game length than a match where both teams are closer to each other. This feature was constructed directly from the original dataset and the column was kept as is.

This model seemed to perform rather poorly, only achieving a training score of 0.175 and testing score of 0.185. This is an indication that gold difference at 15 minutes and our metric of competitiveness in the early game isn't entirely correlated with the overall game length. 

## Final Model

We added several new features similar to `golddiffat15`, and transformed each using their absolute value and quantile to account for outliers: `golddiffat10`, `xpdiffat10`, `xpdiffat15`, `csdiffat10`, and `csdiffat15`, which aim to account for the high complexity of match 'winningness' using more factors than gold. Each of these new features are somewhat related to each other and affect how large of a lead one team might have over its opponent.

We also include a OneHotEncoding transform of the `league` that a particular match took place in, since difference leagues might have varying early game or late game metas and playstyles that affect overall game length.

In order to improve upon the accuracy of the baseline model, we choose to use a Random Forest Regressor to make many attempts toward prediction and utilize averaging to control overfitting when we incorporate more features. 

To tune the hyperparameters of our Random Forest Regressor, we used GridSearchCV to test varying combinations of max_depth and n_estimators, each of which deal with underfitting/overfitting. We found, using 5-fold GridSearchCV, that the best hyperparameters for our model are 60 max_depth and 600 n_estimators.

Our final model experienced a huge improvement in R² score, seeing a training accuracy of 0.964 and testing accuracy of 0.741. We note that the large difference between training and testing accuracy could be due to overfitting, as well as the general unpredictability of League of Legends matches using such simple models. 

## Fairness Analysis

To perform an analysis of fairness, we wish to see if our model predicts game lengths of matches from the 2022 World Championship (WCS) compared to all other leagues. From our EDA, we hypothesized that WCS games are more competitive than other games, meaning they would intuitively have longer overall game lengths in this context. Our evaluation metric is the R² score to determine the testing accuracy of predictions. 

Then, we want to use a permutation test on our two chosen groups. Our null hypothesis is that our model is fair; its precision for WCS league and all other leagues are roughly the same, and any differences are due to random chance. Our alternative hypothesis is that our model is unfair; its precision for WCS league is lower than its precision for any other league.

Our choice of test statistic is the difference in R² scores of our predictions, and significance level of 0.05. After performing a permutation test for 100 repetitions, we obtain a P-value of roughly 0.4, meaning we fail to reject our null hypothesis and can conclude with 95% confidence that our model is fair for matches within WCS versus any other league. 