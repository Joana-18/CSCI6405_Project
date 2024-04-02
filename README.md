# CSCI6405_Project

The data used in this project is in **/data**. All samples were extracted using Reddit's Developer API.

The **/src** folder includes all code:
- **/analysis_transformations**: two notebooks for data exploration and transformation/cleaning steps;
- **/experiments**: code for experiments and corresponding results in .csv
  -  **/feature_importance**: a script to perform permutation feature importance on the best models and a notebook to plot the most important attributes
  -  **/hypotheses**: a notebook with all hypotheses testing steps
  -  **/predictions**:
    - **/no_text**: predictive algorithms with/without a MultiOutputClassifier and **without** text-related features
    - **/tfidf**: predictive algorithms with/without a MultiOutputClassifier and **with** text-related features
- **/extraction**: code for the crawling process
