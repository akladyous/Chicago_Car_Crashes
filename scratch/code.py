#------------------------------------------------------------------------------------------------------------
#Weighted Decision Tree
#------------------------------------------------------------------------------------------------------------
class_weight_dct = dict(
    zip(
        np.unique(y),
        compute_class_weight(class_weight=class_weight_dct, classes=np.unique(y), y=np.ravel(np.array(y)))
    )
)

tree_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best'],
    'max_depth': range(10,70,10),
    'max_features': ['sqrt', 'log2', None],
    'max_leaf_nodes': range(50,200,50),
    'class_weight': ["balanced", class_weight_dct]
#     'ccp_alpha': [.004, 0.008, .01 ,.015, .02]
}

cls_tree = DecisionTreeClassifier(random_state=264)
gs = GridSearchCV(estimator=cls_tree, param_grid=tree_params, n_jobs=4)
model1 = gs.fit(x_train, y_train)
gs.best_params_

model1_y_train_pred = gs.predict(x_train)
print(classification_report(y_train, model1_y_train_pred))



cls_rf.estimators_[0].max_features_, cls_rf.estimators_[0].ccp_alpha
cls_rf.estimators_[0].tree_.max_depth
[(est.get_depth(), est.tree_.max_depth, est.get_n_leaves()) for est in cls_rf.estimators_]
#------------------------------------------------------------------------------------------------------------
#  CROSS VALIDATION
#------------------------------------------------------------------------------------------------------------
cls_tree = DecisionTreeClassifier(random_state=264)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=264)
scores = cross_val_score(cls_tree, x_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
gs = GridSearchCV(estimator=cls_tree, param_grid=params, n_jobs=4)
model2 = gs.fit(x_train, y_train)
model2.best_estimator_
model2.best_params_
model2.best_score_
gs.best_estimator_

model2_y_train_pred = tree_grid.predict(x_train)
roc_auc_score(y_train, model2_y_train_pred)
Metrics(tree_grid, x_train, y_train)


#------------------------------------------------------------------------------------------------------------
#  PIPELINE
#------------------------------------------------------------------------------------------------------------
pipe_num_imputer       = Pipeline([
    ('Imputer', SimpleImputer(missing_values=np.NaN, strategy='most_frequent')) ])
pipe_cat_imputer       = PipeLine([
    ('Imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='UNKNOWN')) ])
pipe_date_transformer  = PipeLine ([ ('Date_Transformer', Date_Transformer()) ])
pip_ohe_transformer    = PipeLine([ ('OneHot', OneHotEncoder(sparse=False, drop='first') ])
pipe_speed_transformer = PipeLine([ ('speed', ) SpeedTransformer() ])


pipe_preprocessor = ColumnSelectorTransformer(
    transformer = [
        ('date', preprocess_dates, dates_cols),
        ('num', preprocess_numeric, numeric_cols),
        ('cat', pipe_cat_imputer, category_cols)
        ('cat X', pipe_cat_imputer, category_cols),
        ('cat y', pipe_cat_imputer, target_cols)
        ('speed', pipe_speed_transformer, ['POSTED_SPEED_LIMIT'])
        ])

pipe_cls = PipeLine(steps=[
    ('tree', DecisionTreeClassifier(**dc_param)),
    ('random_forest', RandomForestClassifier(**rf_best_params)),
    ('gradient_boosting', GradientBoostingClassifier(**gbc_param))
    ])

Pipe_model = PipeLine(steps=[
    ('preprocessor', pipe_preprocessor),
    ('classifiers', pipe_cls)
])

Pipe_model.fit_transform(df.drop(labels=['target']), df['target'])


#------------------------------------------------------------------------------------------------------------
#  FEATURE SELECTION
#------------------------------------------------------------------------------------------------------------
def features_select(X=x_train, y=y_train):
    cols = X.columns
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X, y)
    x_train_fs = fs.transform(X)
    

    plt.figure(figsize=(7,7))
    plt.bar(cols, fs.scores_)
    plt.show()
    
    return x_train_fs, fs

x_train, xls_kbest = features_select(x_train[x_train.columns.difference(['POSTED_SPEED_LIMIT'])], y_train)
xls_kbest.scores_
#------------------------------------------------------------------------------------------------------------
#  https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
#------------------------------------------------------------------------------------------------------------
def model_fit(cls, X=x_train, y=y_train, PerformCV=True, PrintFeaturesImportance=True, CV_fold=5):
    
    model = cls.fit(X, y)
    y_pred = cls.predict(X)
    y_prob = cls.predict_proba(X)[:,1]
    
    if PerformCV :
        cv_score = cross_val_score(cls, X, y, cv=CV_fold, scoring='roc_auc')
    
    print('\mModel Report')
    print(f"{'Accuracy':25} {accuracy_score(y, y_pred)}")
    print(f"{'AUC_ROC Score':25} {roc_auc_score(y, y_pred)}")
    


def modelfit(alg, x_train, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'])
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, dtrain[predictors], dtrain['Disbursed'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

#------------------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION
#------------------------------------------------------------------------------------------------------------

lr = LogisticRegression(max_iter=5000, n_jobs=4, random_state=264)
lr_param={
    'solver': ['newton-cg', 'lbfgs'],
    'penalty': ['l2'],
    'C': [100, 10, 1.0, 0.1, 0.01],
    'class_weight': [class_weight_dct, 'balanced']
}
grid_search_lr = GridSearchCV(estimator=lr, param_grid=lr_param, n_jobs=1, cv=3, scoring='roc_auc')
model4 = grid_search_lr.fit(x_train, np.ravel(np.array(y_train)))
model4_y_train_pred = grid_search_lr.predict(x_train)
lr_best_params = grid_search_lr.best_params_
print(lr_best_params)

print(f"{'Best Score':20} {grid_search_lr.best_score_}")
print(f'ROC_AUC Test: {roc_auc_score(y_train, model4_y_train_pred)}')
grid_search_lr.cv_results_['mean_test_score']

means = grid_search_lr.cv_results_['mean_test_score']
stds = grid_search_lr.cv_results_['std_test_score']
params = grid_search_lr.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))