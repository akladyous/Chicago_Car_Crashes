import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (confusion_matrix, plot_confusion_matrix,classification_report,
                             accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score)
import matplotlib.pyplot as plt 

class Dframe:
    def __init__(self, dataframe):
        self._df = dataframe
    @property
    def num_columns(self):
        return self._df.select_dtypes(include=np.number).columns.tolist()
    @property
    def int_columns(self):
        return self._df.select_dtypes(include=[np.int,np.int0,np.int16,np.int8,np.int16,np.int32,np.int64]).columns.tolist()
    @property
    def float_columns(self):
        return self._df.select_dtypes(include=['float64',np.float,np.float16,np.float32,np.float64]).columns.tolist()
    @property
    def object_columns(self):
        #pd.api.types.is_object_dtype(df['ROAD_DEFECT'])
        return self._df.select_dtypes(include=['object']).columns.tolist()
    @property
    def NaN_num_columns(self):
        series = self._df[self.float_columns].isnull().any()
        return series.index[series.values==True].tolist()
    @property
    def NaN_obj_colmns(self):
        series = self._df[self.object_columns].isnull().any()
        return series.index[series.values==True].tolist()
    
    def null_cols(self):
        total_nan = 0
        print('Feature                        Counts  % ')
        print('-----------------------------  ------- ---')
        for i in self._df.columns:
            total_nan = self._df[i].isnull().sum()
            if total_nan > 0:
                print(f"{i:<30} {total_nan:<7,} {(total_nan / self._df.shape[0]*100):<,.2f}")

    def df_info(self):
        df_nans = self._df.isnull().values.sum()
        df_rows = self._df.shape[0]
        df_columns = self._df.shape[1]
        df_total_items = self._df.shape[0] * self._df.shape[1]

        print(f"{'Data Entries':<34} {self._df.shape[0]}")
        print(f"{'Data Columns':<34} {self._df.shape[1]}")

        print(f"{'DataFrame items':<34} {df_total_items:,}")
        print(f"{'DataFrame Null':<34} {df_nans}  Null values")
        print(f"{'DataFrame contain':<34} {round(df_nans / df_total_items * 100, 2)}% Null Values \n")

        print(' #   Column                        Null Count &  Percent   Dtype')  
        print('---  ------                        ---------------------   -------')
        
        for x in range(self._df.columns.size):
            print(f"{x:<4} {self._df.columns[x]:<30}{self._df[self._df.columns[x]].isna().sum():<8,}\
            {(self._df[self._df.columns[x]].isna().sum() / (df_rows) * 100):<8,.2f}% {str(self._df[self._df.columns[x]].dtype):<5}")

# -------------------------------------------------------------------------------------------------
# Speed Transormer: estimators
# -------------------------------------------------------------------------------------------------
class SpeedTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, parameter='None'):
            self.parameter = parameter
        def fit(self, X, y=None, **fit_params):
            return self
        def transform(self, X, **transform_params):
            X_np_array = X.to_numpy()
            with np.nditer(X_np_array, op_flags=['readwrite']) as rows:
                for row in rows:
                    if 0 <= row <= 15:
                        row[...] = 15
                    elif 16 <= row <= 20:
                        row[...] = 20
                    elif 21 <= row <= 30:
                        row[...] = 30
                    elif 31 <= row <= 45:
                        row[...] = 45
                    elif 46 <= row <= 55:
                        row[...] = 55
                    elif 56 <= row <= 65:
                        row[...] = 65
                    elif 66 <= row <= 100:
                        row[...] = 70
            return pd.DataFrame(X_np_array).astype(np.str)
# -------------------------------------------------------------------------------------------------
# Date (dt64) Transormer: estimators
# -------------------------------------------------------------------------------------------------
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, parameter='None'):
        self.parameter = parameter
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, **transform_params):
        DateTransformed = pd.to_datetime(X)
        return DateTransformed
# -------------------------------------------------------------------------------------------------
# Column Selector Transformer: estimators
# -------------------------------------------------------------------------------------------------
class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_list=[]):
        self.columns_list = columns_list
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, **transform_params):
        SelectedColumns = X[self.columns_list].copy()
        #SelectedColumns = set(X.columns.tolist()).difference((set(self.columns_list)))
        return SelectedColumns
#------------------------------------------------------------------------------------------------------------
# features Selection - SKLEARN
#------------------------------------------------------------------------------------------------------------
def select_features(CLS_name, Threshold, X_TRAIN, Y_TRAIN, X_TEST):
    
    selector   = SelectFromModel(estimator=CLS_name, threshold=Threshold)
    selector.fit(X_TRAIN, Y_TRAIN)
    
    x_train_fs = selector.transform(X_TRAIN)
    x_test_fs  = selector.transform(X_TEST)
    X_columns  = selector.get_support()
    
    return x_train_fs, x_test_fs, X_columns, selector
#------------------------------------------------------------------------------------------------------------
# Target Categorize
#------------------------------------------------------------------------------------------------------------
class Target_Grouper(BaseEstimator, TransformerMixin):
    def __init__(self, parameter='None'):
        self.parameter = parameter
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, **transform_params):
        for idx, val in X.iteritems():
            #DISREGARDING
            if 'DISREGARDING' in val:
                X.iat[idx] = 'DISREGARDING'
            #DISTRACTION
            elif 'DISTRACTION' in val:
                X.iat[idx] = 'DISTRACTION'
            elif 'CELL' in val:
                X.iat[idx] = 'DISTRACTION'
            elif 'PHONE' in val:
                X.iat[idx] = 'DISTRACTION'
            elif 'TEXTING' in val:
                X.iat[idx] = 'DISTRACTION'
            #IMPROPER_MANEUVER
            elif 'IMPROPER OVERTAKING/PASSING' in val:
                X.iat[idx] = 'IMPROPER_MANEUVER'
            elif 'IMPROPER BACKING' in val:
                X.iat[idx] = 'IMPROPER_MANEUVER'
            elif 'IMPROPER LANE USAGE' in val:
                X.iat[idx] = 'IMPROPER_MANEUVER'
            elif 'IMPROPER TURNING/NO SIGNAL' in val:
                X.iat[idx] = 'IMPROPER_MANEUVER'
            #DRIVING_SKILLS/KNOWLEDGE/EXPERIENCE
            elif 'DRIVING' in val:
                X.iat[idx] =  'DRIVING_SKILLS/KNOWLEDGE/EXPERIENCE'
            elif 'FOLLOWING TOO CLOSELY' in val:
                X.iat[idx] = 'DRIVING_SKILLS/KNOWLEDGE/EXPERIENCE'
            #ROAD_CONDITION
            elif 'ROAD ENGINEERING/SURFACE/MARKING DEFECTS' in val:
                X.iat[idx] = 'ROAD_CONDITION'
            elif 'ROAD CONSTRUCTION/MAINTENANCE' in val:
                X.iat[idx] = 'ROAD_CONDITION'
            #ROAD_CONDITION
            elif 'OBSTRUCTED CROSSWALKS' in val:
                X.iat[idx] = 'ROAD_CONDITION'
            elif 'VISION OBSCURED' in val:
                X.iat[idx] = 'ROAD_CONDITION'
            #ALCOHOL_DRUGS
            elif 'ALCOHOL' in val:
                X.iat[idx] =  'ALCOHOL_DRUGS'
            elif 'DRINKING' in val:
                X.iat[idx] = 'ALCOHOL_DRUGS'
            #ANIMAL
            elif 'ANIMAL' in val:
                X.iat[idx] = 'ANIMAL'
        return X
#------------------------------------------------------------------------------------------------------------
#check imbalanced
#------------------------------------------------------------------------------------------------------------
def check_imbalanced(y, verbose=False):
    negative, positive = np.bincount(np.ravel(np.array(y)).astype(np.int64))
    total = positive + negative
    positive_percent = total / positive
    negative_percent = total / negative
    if verbose:
        print(f"{'Total Samples':15} {total}")
        print(f"{'Total Positive':15} {positive:} {(positive*100/total):.2f}%")
        print(f"{'Total Negative':15} {negative:} {(negative*100/total):.2f}%")
    return total, positive, negative
#------------------------------------------------------------------------------------------------------------
#Weighted Decision Tree
#------------------------------------------------------------------------------------------------------------
def get_class_weight(X, Y):
    class_weight = dict()
    class_weight = dict(
        zip(
            np.unique(Y),
            compute_class_weight(class_weight='balanced', classes=np.unique(Y), y=np.ravel(np.array(Y)))
            ))
    return class_weight
#------------------------------------------------------------------------------------------------------------
#print out Mertics
#------------------------------------------------------------------------------------------------------------
def model_metric(clf, X,y):
    y_pred = clf.predict(X)
    print(f"{'ROC_AUC SCORE':15} {roc_auc_score(y, y_pred)}")
    print(f"{'ACCURACY SCORE':15} {accuracy_score(y, y_pred)}")
    return y_pred
#------------------------------------------------------------------------------------------------------------
#print out Mertics
#------------------------------------------------------------------------------------------------------------
def Metrics(clf, X, y):
    y_pred = clf.predict(X)
    
    my_metrics = (
        (accuracy_score, 'accuracy_score'),
        (recall_score, 'recall_score'),
        (precision_score, 'precision_score'),
        (f1_score, 'f1_score')
    )
    
    for f, name in my_metrics:
        print(name.title())
        print(f(y, y_pred))
        print()
        
    plot_confusion_matrix(clf, X, y)
    plt.grid(False)
    plt.show()
#------------------------------------------------------------------------------------------------------------
#WAlgorith Scoring
#------------------------------------------------------------------------------------------------------------
def algo_scoring():
    ml_algo_name  = list()
    ml_algo_score = list()
    ml_algo = [
        SVC(probability=True, random_state=264),
        KNeighborsClassifier(weights='distance', n_jobs=4),
        GaussianNB(),
        DecisionTreeClassifier(class_weight='balanced', random_state=264),
        AdaBoostClassifier(random_state=264),
        RandomForestClassifier() ]

    for algo in ml_algo:
        ml_algo_name.append(algo.__class__.__name__)
        cv_score = cross_val_score(algo, x_train, y_train, scoring='accuracy', cv=2, n_jobs=4).mean()
        ml_algo_score.append(cv_score)

    return pd.DataFrame(list(zip(ml_algo_name, ml_algo_score)), columns=['CLASSIFIER', 'ACCURACY'])


def bagging(X_train, X_test, y_train, y_test,n_est):
    n_est=51
    estimators=range(1,n_est)
    decision_clf = DecisionTreeClassifier()
    scores1 = list()
    scores2 = list()
    
    for est in estimators:
        bagging_clf = BaggingClassifier(decision_clf, n_estimators=est, max_samples=0.67,max_features=0.67, 
                                    bootstrap=True, random_state=9)
        bagging_clf.fit(X_train, y_train)
        # test line
        y_pred_bagging1 = bagging_clf.predict(X_test)
        score_bc_dt1 = accuracy_score(y_test, y_pred_bagging1)
        scores1.append(score_bc_dt1)
        # train line
        y_pred_bagging2 = bagging_clf.predict(X_train)
        score_bc_dt2 = accuracy_score(y_train, y_pred_bagging2)
        scores2.append(score_bc_dt2)
    
    plt.figure(figsize=(10, 6))
    plt.title('Bagging Info')
    plt.xlabel('Estimators')
    plt.ylabel('Scores')
    plt.plot(estimators,scores1,'g',label='test line', linewidth=3)
    plt.plot(estimators,scores2,'c',label='train line', linewidth=3)
    plt.legend()
    plt.show()
    