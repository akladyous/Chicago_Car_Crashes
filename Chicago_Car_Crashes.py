import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (confusion_matrix, plot_confusion_matrix,classification_report,
                             accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score)
import matplotlib.pyplot as plt 

class Dframe:
    def __init__(self, dataframe):
        self._df = dataframe
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
    def NaN_float_columns(self):
        series = self._df[self.float_columns].isnull().any()
        return series.index[series.values==True].tolist()
    @property
    def NaN_object_colmns(self):
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

class SpeedTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, ColumnName):
            self.ColumnName = ColumnName
        def fit(self, X, y=None, **fit_params):
            return self
        def transform(self, X, **transform_params):
            X_np_array = X[self.ColumnName].to_numpy()
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
            return X_np_array


class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_list=[]):
        self.columns_list = columns_list
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, **transform_params):
        SelectedColumns = X[self.columns_list].copy()
        #SelectedColumns = set(X.columns.tolist()).difference((set(self.columns_list)))
        return SelectedColumns
            
class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, DateColumnName):
        self.DateColumnName = DateColumnName
    def fit(self, X, y=None, **fit_params):
        return self
    def transform(self, X, **transform_params):
        DateTransformed = pd.to_datetime(X[self.DateColumnName])
        return DateTransformed

def y_classes(x):
    if 'DISREGARDING' in x:
        return 'DISREGARDING'
    elif 'DISTRACTION' in x:
        return 'DISTRACTION'
    elif 'CELL' in x:
        return 'DISTRACTION'
    elif 'PHONE' in x:
        return 'DISTRACTION'
    elif 'TEXTING' in x:
        return 'DISTRACTION'
    elif 'IMPROPER' in x:
        return 'IMPROPER_MANEUVER'
    elif 'CARELESS' in x:
        return 'IMPROPER_MANEUVER'
    elif 'FOLLOWING TOO CLOSELY' in x:
        return 'IMPROPER_MANEUVER'
    elif 'FAILING TO YIELD' in x:
        return 'IMPROPER_MANEUVER'
    elif 'TURNING RIGHT ON RED' in x:
        return 'IMPROPER_MANEUVER'
    elif 'FAILING TO REDUCE SPEED' in x:
        return 'IMPROPER_MANEUVER'
    elif 'PASSING STOPPED SCHOOL' in x:
        return 'IMPROPER_MANEUVER'
    elif 'BICYCLE ADVANCING' in x:
        return 'IMPROPER_MANEUVER'
    elif 'MOTORCYCLE ADVANCING' in x:
        return 'IMPROPER_MANEUVER'
    elif 'DRIVING' in x:
        return 'DRIVING_SKILLS/KNOWLEDGE/EXPERIENCE'
    elif 'ROAD CONSTRUCTION' in x:
        return 'ROAD_CONDITION'
    elif 'ROAD ENGINEERING' in x:
        return 'ROAD_CONDITION'
    elif 'OBSTRUCTED CROSSWALKS' in x:
        return 'ROAD_CONDITION'
    elif 'VISION OBSCURED' in x:
        return 'ROAD_CONDITION'
    elif 'ALCOHOL' in x:
        return 'ALCOHOL_DRUGS'
    elif 'DRINKING' in x:
        return 'ALCOHOL_DRUGS'
    elif 'ANIMAL' in x:
        return 'ANIMAL'
    else:
        return x
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

class xMetrics:
    def __init__(self, y, y_pred):
        self.y = y
        self.y_pred = y_pred
    
    def get_metrics(self):
        all_metrics = (
        (accuracy_score, 'accuracy_score'),
        (precision_score, 'precision_score'),
        (recall_score, 'recall_score'),
        (f1_score, 'f1_score'),
        (roc_auc_score, 'roc_auc_score')
        )
        
        for m, n in all_metrics:
            #print(f"{n:20} {m(y, self._y_pred)}")
            print(accuracy_score(y, self.y_pred))
    def gg(self):
        self.Accuracy_Score()
    
    def Accuracy_Score(self):
        result = accuracy_score(self.y, self.y_pred)
        print(f"{'Accuracy Score':20} {result}")
    def Precision_Score(self):
        result = precision_score(self.y, self.y_pred)
        print(f"{'Precision Score':20} {result}")
    def Recall_Score(self):
        result = recall_score(self.y, self.y_pred)
        print(f"{'Recall Score':20} {result}")
    def F1_Score(self):
        result = f1_score(self.y, self.y_pred)
        print(f"{'F1 Score':20} {result}")    
    def ROC_AUC_Score(self):
        result = roc_auc_score(self.y, self.y_pred)
        print(f"{'ROC_AUC Score':20} {result}")    
    

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
    