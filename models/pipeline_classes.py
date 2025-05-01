# Основные библиотеки
import pandas as pd
import numpy as np

# Библиотека и отдельные её модули для оптимизации параметров пайплайна
import optuna
from optuna.samplers import TPESampler

# Импортируем пайплайн
from sklearn.pipeline import (Pipeline, make_pipeline)

# Импорт импутера
from sklearn.impute import KNNImputer

# Импорт модуля для разбивки данных на тренировочную и тестовую выборки
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

# Импорт модулей для кодирования и масштабирования
from sklearn.preprocessing import (StandardScaler, 
                                   MinMaxScaler,
                                   MaxAbsScaler,
                                   RobustScaler, 
                                   Normalizer, 
                                   PowerTransformer, 
                                   QuantileTransformer, 
                                   FunctionTransformer)

# Импорт модулей для создания пайплайна
from sklearn.compose import (ColumnTransformer,
                             make_column_selector)

# Импорт модуля для стэкинга
from sklearn.ensemble import StackingClassifier

# Импорт моделей машинного обучения
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Импорт StackingClassifier для использования ансамбля моделей
from sklearn.ensemble import StackingClassifier

# Импорт метрик
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             f1_score, 
                             precision_score, 
                             recall_score, 
                             confusion_matrix, 
                             roc_curve,
                             make_scorer, 
                             log_loss,
                             auc)

# Импорт модулей из catboost
from catboost import (CatBoostClassifier, 
                      CatBoostRegressor, 
                      Pool, 
                      cv)

# Модули используются для создания класса-пустышки
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin




# Задаём константы
RANDOM_STATE = 38
TEST_SIZE = 0.25




'''
    Создаём класс-пустышку.
    Нужен для того, что бы ступени пайплайна не выдавали ошибку при вызове fit и predict, при значении paththrough.
'''
class Passthrough(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X




'''
    Класс для изменения и удаления признаков, а также создания новых признаков
'''
class FeatureEngineering:
    def __init__(self):
        pass

    # Основной метод, проводящий все преобразования
    def run(self, df):
        self.df = df.copy()
        return self.df




'''
    Создаём класс для использования библиотеки optuna в котором прописываем пайплайн, 
    вместе с переменными параметрами, которые будем подбирать.
    Это базовый класс, от него будут наследоваться классы для регрессии и клсссификации
'''
class Objective:
    def __init__(self, X_train, y_train, cv_num, wave='first'):
        self._X_train = X_train
        self._y_train = y_train
        self.cv_num = cv_num
        self.wave = wave
        
        self.model_name = None
        self.best_pipeline = None
        self.score = None
        self.n_catboost_iters = None
        
        self.feature_engineering = FeatureEngineering()
        self.tree_models_list = ['DecisionTreeRegressor', 'DecisionTreeClassifier', 
                                 'RandomForestRegressor', 'RandomForestClassifier']
    
    
    # Основная функция, вызывается при запуске подбора параметров методом study.optimize
    def __call__(self, trial):
        self.create_feature_engineering_params(trial)
        self.create_imputer_params(trial)
        self.create_preprocessor_params(trial)
        self.create_model_with_model_params(trial)
        self.create_pipeline(trial)
        
        # Для классификации стратифицируем кроссвалидацию для регрессии нет
        if isinstance(self.kf, KFold):
            split_args = (self._X_train,)
        elif isinstance(self.kf, StratifiedKFold):
            split_args = (self._X_train, self._y_train)
        
        # Запускаем кроссвалидацию вручную, т.к. кроссвалидация catboost не поддерживает работу с пайплайном
        self.n_catboost_iters = None
        met_cv_list = []
        n_iterations_list = []
        for i, (train_index, val_index) in enumerate(self.kf.split(*split_args)):
            X_train, X_val = self._X_train.iloc[train_index], self._X_train.iloc[val_index]
            y_train, y_val = self._y_train.iloc[train_index], self._y_train.iloc[val_index]
            X_train.reset_index(drop=True, inplace=True)
            y_train.reset_index(drop=True, inplace=True)
            
            # Для catboost отдельно готовим пулы и обучаем модель
            if self.model_name == 'CatBoostClassifier':
                # Сначала тдельно преобразовываем данные, потом обучаем модель на преобразованных пулах
                X_train_pre = X_train.copy()
                X_val_pre = X_val.copy()
                y_train_pre = y_train.copy()
                y_val_pre = y_val.copy()
                
                # Подготовить днные одним действием не получится из-за использования ImbPipeline 
                X_train_pre = self._pipe_final.named_steps['feature_engineering'].fit_transform(X_train_pre)
                X_val_pre = self._pipe_final.named_steps['feature_engineering'].transform(X_val_pre)
                X_train_pre = self._pipe_final.named_steps['imputer'].fit_transform(X_train_pre)
                X_val_pre = self._pipe_final.named_steps['imputer'].transform(X_val_pre)
                X_train_pre = self._pipe_final.named_steps['num'].fit_transform(X_train_pre)
                X_val_pre = self._pipe_final.named_steps['num'].transform(X_val_pre)
                
                # Создаём пулы из преобразованных данных
                train_pool = Pool(data=X_train_pre, label=y_train_pre)
                val_pool = Pool(data=X_val_pre, label=y_val_pre)
                
                # Отдельно обучаем модель на уже преобразованных данных
                self._pipe_final.named_steps['model'].fit(train_pool, 
                                                          eval_set=val_pool, 
                                                          use_best_model=True,
                                                          early_stopping_rounds=50)
                # Получаем количество итераций
                n_iterations_list.append(self._pipe_final.named_steps['model'].get_best_iteration())
                
            # Просто обучаем пайплайни, если модель не catboost
            else:
                self._pipe_final.fit(X_train, y_train)

            # Получаем предсказания для одной кроссвалидационной итерации и помещаем их в список
            met = self.score(self._pipe_final, X=X_val, y_true=y_val)
            met_cv_list.append(met)
        
        # Сохраняем среднее количество итераций в одном фолде для CatBoost
        if self.model_name == 'CatBoostClassifier':
            self.n_catboost_iters = round(np.mean(n_iterations_list))
            trial.set_user_attr('iterations', self.n_catboost_iters)
        
        # Возвращаем среднюю метрику по всем фолдам кроссвалидации
        score_value = np.array(met_cv_list).mean()
        return score_value
    
    
    # Задаём тренировочную выборку
    def set_train_and_test_data(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train

    
    # Метод для изменения и удаления признаков, а также создания новых признаков
    def do_feature_engineering(self, df, **kwargs):
        df = self.feature_engineering.run(df, **kwargs)
        return df
    
    
    # Метод для создания флагов для feature_engineering (для подбора параметров)
    def create_feature_engineering_params(self, trial):
        if self.wave == 'first':
            self._feature_engineering_param ={}
        elif self.wave == 'second':
            self._feature_engineering_param = {}

            
    # Подбор параметров для KNNImputer
    def create_imputer_params(self, trial):
        self._imputer_params = {
            'n_neighbors': trial.suggest_int('imputer_neighbors', 5, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance'])
        }
        self._imputer = KNNImputer(**self._imputer_params)
    
    
    # Метод для выбора способа кодирования и масштабирования (для подбора параметров)
    def create_preprocessor_params(self, trial):
        self._num_encoding_param = {
            'num_encoding': trial.suggest_categorical('num_encoding', ['MinMaxScaler', 
                                                                       'StandardScaler', 
                                                                       'RobustScaler', 
                                                                       'MaxAbsScaler',
                                                                       'PowerTransformer standardize=True',
                                                                       'PowerTransformer standardize=False',
                                                                       'QuantileTransformer',
                                                                       'passthrough']),
        }

        if self._num_encoding_param['num_encoding'] == 'MinMaxScaler':
            self._num_encoding = MinMaxScaler()
        elif self._num_encoding_param['num_encoding'] == 'StandardScaler':
            self._num_encoding = StandardScaler()
        elif self._num_encoding_param['num_encoding'] == 'RobustScaler':
            self._num_encoding = RobustScaler()
        elif self._num_encoding_param['num_encoding'] == 'MaxAbsScaler':
            self._num_encoding = MaxAbsScaler()
        elif self._num_encoding_param['num_encoding'] == 'PowerTransformer standardize=True':
            self._num_encoding = PowerTransformer(method='yeo-johnson', standardize=True)
        elif self._num_encoding_param['num_encoding'] == 'PowerTransformer standardize=False':
            self._num_encoding = PowerTransformer(method='yeo-johnson', standardize=False)
        elif self._num_encoding_param['num_encoding'] == 'QuantileTransformer':
            n_quantiles = trial.suggest_int('n_quantiles', 50, 500)
            output_distribution = trial.suggest_categorical('output_distribution', ['uniform', 'normal'])
            self._num_encoding = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)
        else:
            self._num_encoding = Passthrough()

    
    # Метод для создания пайплайна
    def create_pipeline(self, trial):
        self._pipe_final = Pipeline(
            [
                ('feature_engineering', FunctionTransformer(self.do_feature_engineering, kw_args={**self._feature_engineering_param})),
                ('imputer', self._imputer),
                ('num', self._num_encoding),
                ('model', self._model)
            ]
        )
    
    
    # Метод для создания копии пайплайна с новой необученной моделью и измененными параметрами
    def change_model_params(self, pipeline, new_params):
        # Создаем копию пайплайна без копирования самой модели
        new_pipeline = Pipeline([
            (name, step) for name, step in pipeline.steps if name != 'model'
        ])

        # Извлекаем текущие параметры модели
        current_model = pipeline.named_steps['model']
        current_params = current_model.get_params()

        # Обновляем параметры новыми значениями
        current_params.update(new_params)

        # Создаем новый экземпляр модели с обновленными параметрами
        if self.model_name == 'CatBoostClassifier':
            new_model = CatBoostClassifier(**current_params)

        # Добавляем новую модель в пайплайн
        new_pipeline.steps.append(('model', new_model))

        return new_pipeline
    
    
    # Метод автоматически вызывается после каждого триала, 
    # нужна для сохранения пайплайна с лучшей комбинацией параметров
    def callback(self, study, trial):
        if study.best_trial.number == trial.number:
            self.best_pipeline = self._pipe_final
            
            # Заменяем количество итераций на среднее по кроссвалидации для Catboost
            if self.model_name == 'CatBoostClassifier':
                new_params = {'iterations': self.n_catboost_iters}
                self.best_pipeline = self.change_model_params(self.best_pipeline, new_params)
                
                X_train = self._X_train.copy()
                y_train = self._y_train.copy()
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                
                self.best_pipeline.fit(X_train, y_train)




'''
    Создаём класс библиотеки optuna в котором прописываем пайплайн, 
    вместе с переменными параметрами, которые будем подбирать для классификации
'''
class ObjectiveClassifier(Objective):
    def __init__(self, X_train, y_train, cv_num, wave):
        super().__init__(X_train, y_train, cv_num, wave)
        self.kf = StratifiedKFold(n_splits=cv_num, shuffle=True, random_state=RANDOM_STATE)
    
    
    # Задаём тренировочную выборку
    def set_model(self, model_name):
        if model_name == 'SVC':
            self.model_name = 'SVC'
        elif model_name == 'KNNClassifier':
            self.model_name = 'KNNClassifier'
        elif model_name == 'LogisticRegression':
            self.model_name = 'LogisticRegression'
        elif model_name == 'DecisionTreeClassifier':
            self.model_name = 'DecisionTreeClassifier'
        elif model_name == 'RandomForestClassifier':
            self.model_name = 'RandomForestClassifier'
        elif model_name == 'CatBoostClassifier':
            self.model_name = 'CatBoostClassifier'
        elif model_name == 'MLPClassifier':
            self.model_name = 'MLPClassifier'
        elif model_name == 'KerasNN':
            self.model_name = 'KerasNN'
        else:
            print('Модель не была установлена! Такой модели нет в списке')


    # Создаём обёртку для метрики f1, что бы она считалась с учётом порога
    def f1_with_threshold(self, y_true, y_pred_proba, threshold):
        y_pred = (y_pred_proba > threshold).astype(int)
        return f1_score(y_true, y_pred)

    # Второй метод для создания обёртки f1
    def custom_f1_scorer(self, threshold):
        return make_scorer(self.f1_with_threshold, greater_is_better=True, threshold=threshold)#, needs_proba=True)

    
    # Задаём метрику для поиска лучшей модели
    def set_score(self, score):
        self.score_name = score
        if self.score_name == 'roc_auc':
            self.score = make_scorer(roc_auc_score, greater_is_better=True, multi_class='ovr', needs_proba=True)
        elif self.score_name == 'accuracy':
            self.score = make_scorer(accuracy_score, greater_is_better=True)
        elif self.score_name == 'f1':
            self.score = self.custom_f1_scorer(self.threshold)
        

    # Метод для создания модели и параметров модели (для подбора параметров)
    def create_model_with_model_params(self, trial):
        if self.model_name == 'SVC':
            self._svc_params ={
                'random_state': RANDOM_STATE,
                'probability': True,
                'C': trial.suggest_float('C', 0.01, 50),
                'kernel': trial.suggest_categorical('kernel', ['poly', 'rbf', 'sigmoid'])
            }
            self._model = SVC(**self._svc_params)
        
        elif self.model_name == 'KNNClassifier':
            self._knn_params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 15, 75),
                'p': trial.suggest_int('p', 1, 2),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 
                                                                     'ball_tree', 
                                                                     'kd_tree', 
                                                                     'brute'])
            }
            self._model = KNeighborsClassifier(**self._knn_params)
        
        elif self.model_name == 'LogisticRegression':
            self._logistic_regression_params ={
                'random_state': RANDOM_STATE,
                'C': trial.suggest_float('C', 0.01, 50)
            }
            solver_penalty_comb = trial.suggest_categorical('solver_and_penalty', 
                                                            ['lbfgs + l2', 'lbfgs + None', 
                                                             'liblinear + l1', 'liblinear + l2',
                                                             'sag + l2', 'sag + None',
                                                             'saga + l1', 'saga + l2', 'saga + None'])
            if solver_penalty_comb == 'lbfgs + l2':
                self._logistic_regression_params['solver'] = 'lbfgs'
                self._logistic_regression_params['penalty'] = 'l2'
            elif solver_penalty_comb == 'lbfgs + None':
                self._logistic_regression_params['solver'] = 'lbfgs'
                self._logistic_regression_params['penalty'] = None
            if solver_penalty_comb == 'liblinear + l1':
                self._logistic_regression_params['solver'] = 'liblinear'
                self._logistic_regression_params['penalty'] = 'l1'
            elif solver_penalty_comb == 'liblinear + l2':
                self._logistic_regression_params['solver'] = 'liblinear'
                self._logistic_regression_params['penalty'] = 'l2'
            elif solver_penalty_comb == 'sag + l2':
                self._logistic_regression_params['solver'] = 'sag'
                self._logistic_regression_params['penalty'] = 'l2'
            elif solver_penalty_comb == 'sag + None':
                self._logistic_regression_params['solver'] = 'sag'
                self._logistic_regression_params['penalty'] = None
            elif solver_penalty_comb == 'saga + l1':
                self._logistic_regression_params['solver'] = 'saga'
                self._logistic_regression_params['penalty'] = 'l1'
            elif solver_penalty_comb == 'saga + l2':
                self._logistic_regression_params['solver'] = 'saga'
                self._logistic_regression_params['penalty'] = 'l2'
            elif solver_penalty_comb == 'saga + None':
                self._logistic_regression_params['solver'] = 'saga'
                self._logistic_regression_params['penalty'] = None
            self._model = LogisticRegression(**self._logistic_regression_params)
        
        elif self.model_name == 'DecisionTreeClassifier':
            self._decisiontree_params = {
                'random_state': RANDOM_STATE,
                'max_depth': trial.suggest_int('max_depth', 2, 20),
                'max_features': trial.suggest_categorical('max_features', ['log2', 'sqrt', None]),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random'])
            }
            self._model = DecisionTreeClassifier(**self._decisiontree_params)
        
        elif self.model_name == 'RandomForestClassifier':
            self._random_forest_params = {
                'random_state': RANDOM_STATE,
                'max_depth': trial.suggest_int('max_depth', 2, 15),
                'max_features': trial.suggest_int('max_features', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 20),
                'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini','entropy'])
            }
            self._model = RandomForestClassifier(**self._random_forest_params)

        elif self.model_name == 'CatBoostClassifier':
            self._catboost_params = {
                'random_seed': RANDOM_STATE,
                'iterations': 5000,
                'verbose': False,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 2, 15),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 1e2, log=True),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'auto_class_weights': trial.suggest_categorical('auto_class_weights', ['Balanced', None]),
                'border_count': trial.suggest_int('border_count', 32, 254),
                'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 2, 10),
                'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient'])
            }
            self._model = CatBoostClassifier(**self._catboost_params)

        elif self.model_name == 'MLPClassifier':
            self._mlp_params = {
                'random_state': RANDOM_STATE,
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (50, 50)]),
                'activation': trial.suggest_categorical('activation', ['relu']),
                'solver': trial.suggest_categorical('solver', ['adam']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-3, log=True),
                'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling'])
            }
            self._model = MLPClassifier(**self._mlp_params)




'''
    Класс для запуска подбора лучших параметров пайплайна
'''
class Study:
    def __init__(self, X_train, y_train, X_test, y_test, cv_num, wave='first'):
        # Создаём словари для сохранения лучших пайплайнов и результатов
        self.wave = wave
        self.best_piplines_dict = {}
        self.models_res_dict = {}
        
        # Создаём необходимиые переменные (устанавливаем дефолтные значения)
        self.score = None
        self._cv_num = cv_num
        self._trials_num = 5
        self._show_trials_num = 1
        
        # Создаём переменные для тренировочной и тестовой выборок
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        
        # Создаём объект класса с настроемым для поиска параметров пайплайном
        self._objective = ObjectiveClassifier(self._X_train, self._y_train, self._cv_num, self.wave)
    
    
    # Задаём список моделей
    def set_models_list(self, models_list):
        self.models_list = models_list
        
    # Задаём метрику
    def set_model_score(self, score):
        self.score = score

    # Задаём количество поисков для одной модели
    def set_trials_num(self, trials_num):
        self._trials_num = trials_num

    # Задаём количество выводимых на экран результатов поиска
    def set_show_trials_num(self, show_trials_num):
        self._show_trials_num = show_trials_num
        
    # Задаём тренировочную и тестовую выборки
    def set_dataframe(self, X_train, y_train, X_test, y_test):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
    
    
    # Функция для запуска процесса подбора гиперпараметров для пайплайна
    def run(self):
        # Создаём датафрейм, для вывода на экрран результата всех моделей
        self._study_info = pd.DataFrame(columns=['model', f'{self.score}_cv', f'{self.score}_test'])
        self._study_info.columns.name = 'models_best_results'
        
        # Добавляем в объект класса Objective тренировочную выборку и указываем метрику
        self._objective.set_train_and_test_data(self._X_train, self._y_train)
        self._objective.set_score(self.score)
        
        # Проходим цыклом по всем моделям
        for model_name in self.models_list:
            self._objective.set_model(model_name)
            
            # Создаём объект класса potuna для подбора параметров
            study = optuna.create_study(direction='maximize', 
                                        study_name='test_models', 
                                        sampler=TPESampler(seed=RANDOM_STATE))
            
            # Запускем процесс подбора параметров
            study.optimize(self._objective, 
                           n_trials=self._trials_num,
                           callbacks=[self._objective.callback], 
                           n_jobs=1, 
                           show_progress_bar=True)
            
            # Создаём датафрейм с результатами триалов, сохраняем в словарь, выводим на экран
            model_res = self.create_model_res_dataframe(model_name, study.trials_dataframe())
            # Убираем "user_attrs_" из назваания добавленного для catboost параматра с количеством итераций
            if model_name == 'CatBoostClassifier':
                model_res.rename(columns=lambda x: x.replace('user_attrs_', ''), inplace=True)
            model_res = model_res.reset_index(drop=True)
            
            # Сохраняем датафрейм с результатами триалов в словарь
            self.models_res_dict[model_name] = model_res
            
            # Выводим датафрейм с результатами триалов на экран (при необходимости)
            if self._show_trials_num:
                model_res_show = model_res.copy()
                model_res_show.index = range(1, len(model_res_show) + 1)
                display(model_res_show.head(self._show_trials_num))

            # Обучаем лучший пайплайн
            self._objective.best_pipeline.fit(self._X_train.reset_index(drop=True), 
                                              self._y_train.reset_index(drop=True))
            
            # Добавляем лучший пайплайн, уже обученный, в словарь
            self.best_piplines_dict[model_name] = self._objective.best_pipeline
            
            # Получаем предсказания на тестовой выборке
            self._y_pred = self._objective.best_pipeline.predict(self._X_test)
            
            # Для классификации предсказываем вероятность
            self._y_proba = self._objective.best_pipeline.predict_proba(self._X_test)

            
            # Записываем лучшие метрики классификации в датафрейм
            if self.score == 'roc_auc':
                self._study_info.loc[-1] = [model_name, model_res.loc[0, self.score], 
                                            roc_auc_score(self._y_test, self._y_proba, multi_class='ovr')]
            elif self.score == 'accuracy':
                self._study_info.loc[-1] = [model_name, model_res.loc[0, self.score], 
                                            accuracy_score(self._y_test, self._y_pred)]
            elif self.score == 'f1':
                self._study_info.loc[-1] = [model_name, model_res.loc[0, self.score], 
                                            f1_score(self._y_test, self._y_pred)]
            
            self._study_info = self._study_info.reset_index(drop=True)
        
        # Сортируем значения в датафрейме с итоговыми результатами и выводим на экран
        self._study_info = self._study_info.reset_index(drop=True)
        self._study_info = self._study_info.sort_values(by=f'{self.score}_cv', ascending=False)
        print()
        self._study_info = self._study_info.reset_index(drop=True)
        study_info_show = self._study_info.copy()
        study_info_show.index = range(1, len(study_info_show) + 1)
        display(study_info_show)

        
    # Функция для преобразования датафрейма для более наглядного отображения параметров модели
    def create_model_res_dataframe(self, model_name, model_res):
        # Удаляем неинформативные колонки, меняем названия, сортируем значения
        model_res = model_res.sort_values(by='value', ascending=False)
        model_res.columns.name = model_name
        model_res = model_res.reset_index(drop=True)
        model_res['duration'] = model_res['duration'].astype(str).str.replace('0 days ', '')
        model_res = model_res.drop(columns=['number', 'datetime_start', 'datetime_complete', 'state'])
        model_res.columns = model_res.columns.str.replace('params_', '')

        # Расставляем колонки в нужном порядке
        cols = model_res.columns.tolist()
        rest_cols = cols.copy()
        rest_cols.remove('value')
        rest_cols.remove('duration')
        rest_cols.remove('num_encoding')

        new_cols = ['value', 
                    'duration', 
                    'num_encoding'] + rest_cols

        model_res = model_res[new_cols]

        # Меняем название колонки с результатом на нашу метрику
        model_res = model_res.rename(columns={'value': self.score})
        
        return model_res