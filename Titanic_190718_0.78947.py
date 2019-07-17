import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

class ExcelIO:
    def __init__(self, raw_train=None, raw_test=None, submit=None, path=None):
        self.raw_train, self.raw_test = raw_train, raw_test
        self.submit = submit
        self.path = path

    def import_csv(self):
        self.path = "C:/Users/Public/Program/Python/Project/Competition/Titanic/"
        self.raw_train = pd.read_csv(self.path + "train.csv")
        self.raw_test = pd.read_csv(self.path + "test.csv")

    def export_csv(self, df):
        self.submit = pd.DataFrame(df[['PassengerId', 'Survived']])
        self.submit.to_csv(self.path + "titanic_submit.csv", index=False)


excel = ExcelIO()
excel.import_csv()


class DescriptiveStatistics:
    def __init__(self, raw_train=None, raw_test=None, corr=None):
        self.raw_train, self.raw_test  = raw_train, raw_test
        self.corr = corr

    def get_correlation(self):
        self.corr = self.raw_train.corr()

    def print_shape(self):
        print('titanic_raw_train_data', self.raw_train.shape)

    def main(self):
        self.get_correlation()
        self.print_shape()


descriptor = DescriptiveStatistics(excel.raw_train, excel.raw_test)
descriptor.main()


class Scope():
    def __init__(self, raw_train=None, raw_test=None, train=None, test=None, low_card_cols=None, numeric_cols=None):
        self.raw_train, self.raw_test = raw_train, raw_test
        self.train, self.test = train, test
        self.low_card_cols = low_card_cols
        self.numeric_cols = numeric_cols

    def define_low_card_and_numeric_col(self):
        self.low_card_cols = [cname for cname in self.raw_train.columns if self.raw_train[cname].nunique() < 10 and
                              self.raw_train[cname].dtype == "object"]
        self.numeric_cols = [cname for cname in self.raw_train.columns if
                             self.raw_train[cname].dtype in ['int64', 'float64']]
        self.numeric_cols.remove('Survived')

    def get_data_in_scope(self):
        self.train = self.raw_train[self.low_card_cols + self.numeric_cols + ['Survived']]
        self.test = self.raw_test[self.low_card_cols + self.numeric_cols]
        print('titanic_train_data: Data_Scope', self.train.shape)

    def main(self):
        self.define_low_card_and_numeric_col()
        self.get_data_in_scope()


scoped = Scope(descriptor.raw_train, descriptor.raw_test)
scoped.main()


class Classifier:
    def __init__(self, raw_train=None, raw_test=None, train=None, test=None):
        self.raw_train, self.raw_test = raw_train, raw_test
        self.train, self.test = train, test

    def apply_pclass_2_str(self, x):  # Change PClass From Numeric to Character
        return {1: "A", 2: "B", 3: "C"}.get(x, 0)

    def apply_first_name(self, x):
        return x.split()[0]

    def apply_ticket_prefix(self, x):
        if x.isdigit(): return min(int(x[0]), 4)
        elif "A./" in x or 'A/' in x or 'A.5' in x: return "A"
        elif "C.A." in x or "CA" in x: return 'CA'
        elif "F.C.C" in x: return "FCC"
        elif "LINE" in x: return "LINE"
        elif "PC" in x: return "PC"
        elif "PP" in x: return "PP"
        elif "S.C." in x or "SC" in x: return "SC"
        elif "SOTON" in x: return "SOTON"
        elif "STON" in x: return "STON"
        elif "W./C" in x: return "WC"
        else: return 0

    def apply_break_down(self, df1, df2):
        df1['Pclass'] = df2['Pclass'].apply(self.apply_pclass_2_str)
        df1['Ticket_p'] = df2['Ticket'].apply(self.apply_ticket_prefix)
        df1['First_Name'] = df2['Name'].apply(self.apply_first_name)
        return df1

    def main(self):
        self.train = self.apply_break_down(self.train, self.raw_train)
        self.test = self.apply_break_down(self.test, self.raw_test)
        print('titanic_train_data: Extract', self.train.shape)


classified = Classifier(scoped.raw_train, scoped.raw_test, scoped.train, scoped.test)
classified.main()


class ColumnAddition:
    def __init__(self, train=None, test=None):
        self.train, self.test = train, test

    def add_family(self):
        self.train['Family'] = self.train['SibSp'] + self.train['Parch']
        self.test['Family'] = self.test['SibSp'] + self.test['Parch']

        for i in range(891):
            self.train.loc[i, 'Family_N'] = self.train.loc[i, 'First_Name'] + str(self.train.loc[i, 'Family'])

        for i in range(418):
            self.test.loc[i, 'Family_N'] = self.test.loc[i, 'First_Name'] + str(self.test.loc[i, 'Family'])

    def add_survive_order(self):
        # Survived Order: Child > Female > Single Female > Single Male > Male
        for i in range(891):
            if self.train.loc[i, 'Age'] <= 15:
                self.train.loc[i, 'Survived_Order'] = 1
            elif (self.train.loc[i, 'Sex'] == 'female') and (self.train.loc[i, 'Family']) > 0:
                self.train.loc[i, 'Survived_Order'] = 2
            elif (self.train.loc[i, 'Sex'] == 'female') and (self.train.loc[i, 'Family']) == 0:
                self.train.loc[i, 'Survived_Order'] = 3
            elif (self.train.loc[i, 'Sex'] == 'male') and (self.train.loc[i, 'Family']) == 0:
                self.train.loc[i, 'Survived_Order'] = 4
            elif (self.train.loc[i, 'Sex'] == 'male') and (self.train.loc[i, 'Family']) > 0:
                self.train.loc[i, 'Survived_Order'] = 5
            else:
                self.train.loc[i, 'Survived_Order'] = 999

        for i in range(418):
            if self.test.loc[i, 'Age'] <= 15:
                self.test.loc[i, 'Survived_Order'] = 1
            elif (self.test.loc[i, 'Sex'] == 'female') and (self.test.loc[i, 'Family']) > 0:
                self.test.loc[i, 'Survived_Order'] = 2
            elif (self.test.loc[i, 'Sex'] == 'female') and (self.test.loc[i, 'Family']) == 0:
                self.test.loc[i, 'Survived_Order'] = 3
            elif (self.test.loc[i, 'Sex'] == 'male') and (self.test.loc[i, 'Family']) == 0:
                self.test.loc[i, 'Survived_Order'] = 4
            elif (self.test.loc[i, 'Sex'] == 'male') and (self.test.loc[i, 'Family']) > 0:
                self.test.loc[i, 'Survived_Order'] = 5
            else:
                self.test.loc[i, 'Survived_Order'] = 999


added = ColumnAddition(classified.train, classified.test)
added.add_family()


class Imputation:
    def __init__(self, train=None, test=None, imputer=None):
        self.train, self.test = train, test
        self.imputer = imputer

    def imputate(self):
        self.imputer = Imputer()
        self.train['Age'] = self.imputer.fit_transform(self.train['Age'].to_frame())
        self.test['Age'] = self.imputer.transform(self.test['Age'].to_frame())


imputated = Imputation(added.train, added.test)
imputated.imputate()
added = ColumnAddition(imputated.train, imputated.test)
added.add_survive_order()


class ColumnDeletion:
    def __init__(self, train=None, test=None):
        self.train, self.test = train, test

    def drop_family(self):
        self.train = self.train.drop(columns=['Family'])
        self.test = self.test.drop(columns=['Family'])


deleted = ColumnDeletion(added.train, added.test)
deleted.drop_family()


class OneHotEncoder:
    def __init__(self, train=None, test=None, cat_train=None, cat_test=None):
        self.train, self.test = train, test
        self.cat_train, self.cat_test = cat_train, cat_test

    def encode(self):
        self.cat_train = pd.get_dummies(self.train)
        self.cat_test = pd.get_dummies(self.test)
        self.cat_train, self.cat_test = self.cat_train.align(self.cat_test, join='outer', axis=1)

    def fill_na_after_encode(self):
        self.cat_train.fillna(0, inplace=True)
        self.cat_test.fillna(0, inplace=True)
        print('titanic_cat_train_data: Final', self.cat_train.shape)

    def main(self):
        self.encode()
        self.fill_na_after_encode()


encoded = OneHotEncoder(deleted.train, deleted.test)
encoded.main()


class Model:
    def __init__(self, train=None, test=None, cat_train=None, cat_test=None, y=None, x=None,
                 train_y=None, train_x=None, val_y=None, val_x=None, model=None, pipeline=None, cvs=None):
        self.train, self.test = train, test
        self.cat_train, self.cat_test = cat_train, cat_test
        self.y, self.x = y, x
        self.train_y, self.train_x, self.val_y, self.val_x  = train_y, train_x, val_y, val_x
        self.model = model
        self.pipeline = pipeline
        self.cvs = cvs

    def set_parameter(self):
        # Not Consider PassengerID
        self.y = self.cat_train.Survived
        self.x = self.cat_train.loc[:, (self.cat_train.columns != 'Survived') &
                                       (self.cat_train.columns != 'PassengerID')]

    def split_train_valid(self):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
            self.x, self.y, train_size=0.7, test_size=0.3, random_state=0)

    def regress_by_random_forest(self):
        # No Gradient Boosting Model, Decision Tree Regressor
        self.model = RandomForestRegressor(n_estimators=50)
        self.model.fit(self.train_x, self.train_y)

    def calculate_cross_val_score_w_pipeline(self):
        self.pipeline = make_pipeline(Imputer(), RandomForestRegressor(n_estimators=50))
        self.cvs = cross_val_score(self.pipeline, self.train_x, self.train_y,
                                   scoring='neg_mean_absolute_error', cv=5) * -1
        print('Cross Validation Score: ', self.cvs)

    def predict_result(self):
        self.cat_test = self.cat_test.drop(['Survived'], axis=1)
        self.cat_test['Survived'] = self.model.predict(self.cat_test)
        self.cat_test['Survived'] = self.cat_test['Survived'].round(0).astype(int)

    def main(self):
        self.set_parameter()
        self.split_train_valid()
        self.regress_by_random_forest()
        self.calculate_cross_val_score_w_pipeline()
        self.predict_result()


model_df = Model(encoded.train, encoded.test, encoded.cat_train, encoded.cat_test)
excel.export_csv(model_df.cat_test)
