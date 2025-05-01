# Основные библиотеки
import pandas as pd

# Импортируем библиотеку shap, для анализа важности признаков
import shap

# Импортируем библиотеки для построения графиков
import matplotlib.pyplot as plt
import seaborn as sns





'''
    Класс для анализа важности признаков с помошью shap
'''
class ShapShow:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.explainer = self.create_explainer()
        self.shap_values = self.compute_shap_values()

    def create_explainer(self):
        # Создаём PermutationExplainer для совместимости с составными моделями
        return shap.PermutationExplainer(self.model.predict_proba, self.X_test)

    def compute_shap_values(self):
        # Рассчитываем SHAP значения
        return self.explainer(self.X_test)

    def plot_feature_importance_bar(self, max_display=20):
        # Рассчитываем средние абсолютные значения SHAP для каждого признака по всем классам
        mean_shap_values = abs(self.shap_values.values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Importance': mean_shap_values.mean(axis=1)
        }).sort_values(by='Importance', ascending=False)

        # Строим барплот
        feature_importance.head(max_display).plot(
            kind='barh', x='Feature', y='Importance', legend=False, figsize=(8, 3), color='hotpink'
        )
        plt.title("Feature Importance (Aggregated Across Classes)")
        plt.xlabel("mean(|SHAP value|)")
        plt.ylabel("Feature")
        plt.gca().invert_yaxis()
        plt.show()
    
    def plot_beeswarm_multiclass(self, max_display=10):
        # Получаем предсказанные вероятности для каждого класса
        proba = self.model.predict_proba(self.X_test)
        n_classes = proba.shape[1]

        # Создаём градиент цвета для точек
        colors = np.dot(proba, np.eye(n_classes))  # Матрица смешивания цветов

        # Строим пчелиный рой
        shap.summary_plot(
            self.shap_values.values, self.X_test,
            max_display=max_display, color_bar=False, color=colors
        )

        # Легенда (например, квадрат с цветами для каждого класса)
        plt.figure(figsize=(4, 3))
        for i, color in enumerate(np.eye(n_classes)):
            plt.scatter([], [], color=color, label=f'Class {i}')
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, title="Classes")
        plt.axis('off')
        plt.show()