from sklearn.model_selection import cross_validate
import numpy as np

def cross_validate_model(model, X, y, cv=5):
    """
    Ну и ну, вы это читаете, я приятно удивлен...
    Кросс-валидация модели RecSys
    
    Args:
        model: обученная модель
        X: матрица взаимодействий
        y: целевые переменные
        cv: кол-во фолдов
    
    Returns:
        dict: метрики кросс-валидации
    """
    scoring = {
        'precision': 'precision_macro',
        'recall': 'recall_macro', 
        'f1': 'f1_macro'
    }
    
    cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    return {
        'mean_precision': np.mean(cv_scores['test_precision']),
        'std_precision': np.std(cv_scores['test_precision']),
        'mean_recall': np.mean(cv_scores['test_recall']),
        'std_recall': np.std(cv_scores['test_recall']),
        'mean_f1': np.mean(cv_scores['test_f1']),
        'std_f1': np.std(cv_scores['test_f1'])
    }

def evaluate_model_stability(model, X, y, n_runs=10):
    """
    Оценка стабильности модели на множественном запуске
    """
    metrics_history = []
    
    for i in range(n_runs):
        # Щепотка шумихи для оценки стабильности
        X_noisy = X + np.random.normal(0, 0.01, X.shape)
        metrics = cross_validate_model(model, X_noisy, y, cv=3)
        metrics_history.append(metrics)
    
    return metrics_history