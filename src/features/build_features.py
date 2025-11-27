import pandas as pd
from sklearn.preprocessing import StandardScaler

# Так как источник пока неизвестен, ставлю болванку
def normalize_features(user_item_matrix):
    """Нормализация пользовательских признаков"""
    scaler = StandardScaler()
    return scaler.fit_transform(user_item_matrix)

def add_temporal_features(ratings_df):
    """Добавление временных признаков"""
    ratings_df['day_of_week'] = pd.to_datetime(ratings_df['timestamp']).dt.dayofweek
    ratings_df['month'] = pd.to_datetime(ratings_df['timestamp']).dt.month
    return ratings_df