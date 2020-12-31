# Abstract 

Predicting game results has always been popular in dierent sports elds and with the
development of technology. eSports has become a big part of young people's lives with
growing trends in players from casual gameplay to professional gameplay. One of the
most popular methods of making predictions is by utilizing machine learning methods
and algorithms. By applying machine learning models, complex player behavior can be
analyzed for predicting individual and long-term match results. As such, to solve this
problem of this work, Logistic Regression (LR), Random Forest (RF), and LightGBM
(LGBM) are used to predict the possibilities of a team winning of each round of Counter
Strike: Global Oensive games. Counter Strike is a popular rst-person shooting game
that has a rich community and plenty of accessible data and results.
There are a seemingly endless number of features, events, and environmental factors
that have an in
uence or correlation on gameplay and match results. In this thesis work,
instead of player skills or team ratings, only the features that represent the current, realtime
game situation (e.g., number of alive players) or features related to spatial data are
investigated for modeling and analyzing results via machine learning models. Distance
is considered to be a direct way of showing position relations, which ultimately, have a
large in
uence on match results. Therefore, dierent features related to player-to-player
distances and player-to-bomb distances were used to represent the positional relations in
the games. Two path planning algorithms, A* and Floyd-Warshall, were used to obtain
distances so that they are re
ective to actual player distances in the matches.
For the three machine learning algorithms and two path planning algorithms, the
data are trained one time without hyper-parameter tuning and one time with tuning.
Consequently, there are a total of 12 predictions for a single set of data to compare with
the best accuracy's being reported as: 88.1% for LGBM, 87.9% for RF, and 85.9% for
LR and an observation that LGBM is more sensitive to parameter tuning.
This thesis veries that the round winning situation can be successfully predicted
with satisfactory results. Furthermore, ensemble machine leaning methods work better
than Logistic Regression for this specic problem and setup. Lastly, in this work, the
Floyd-Warshall algorithm was shown to work better with ensemble methods while A* is
better with Logistic Regression.
