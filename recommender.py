import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Datos de ejemplo: usuarios, películas y calificaciones (1-5)
data = {
    'user_id': ['user1', 'user1', 'user1', 'user2', 'user2', 'user2', 'user3', 'user3', 'user3', 'user4', 'user4', 'user4', 'user5', 'user5', 'user5'],
    'item_id': ['movie1', 'movie2', 'movie3', 'movie1', 'movie2', 'movie4', 'movie2', 'movie3', 'movie5', 'movie1', 'movie3', 'movie6', 'movie4', 'movie5', 'movie6'],
    'rating': [5, 3, 4, 4, 2, 5, 5, 2, 3, 2, 3, 4, 3, 5, 4]
}

# Convertir los datos en un DataFrame
df = pd.DataFrame(data)
print("Datos de calificaciones de usuarios y películas:")
print(df)

# Cargar los datos en la biblioteca surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Entrenar el modelo utilizando SVD
trainset = data.build_full_trainset()
algo = SVD()

# Realizar la validación cruzada
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Entrenar el modelo en el conjunto completo de datos
algo.fit(trainset)

# Hacer predicciones para un usuario específico
user_id = 'user1'
item_id = 'movie4'
pred = algo.predict(user_id, item_id)
print(f'Predicción de calificación para {user_id} y {item_id}: {pred.est:.2f}')

# Generar recomendaciones para un usuario específico
def get_recommendations(user_id, df, algo, n=5):
    # Obtener todos los ítems (películas) únicos
    all_items = df['item_id'].unique()
    # Obtener los ítems que el usuario ya ha calificado
    user_items = df[df['user_id'] == user_id]['item_id'].unique()
    # Filtrar los ítems que el usuario no ha calificado
    items_to_predict = [item for item in all_items if item not in user_items]
    # Predecir calificaciones para los ítems no calificados
    predictions = [algo.predict(user_id, item).est for item in items_to_predict]
    # Crear un DataFrame con las predicciones
    recommendations = pd.DataFrame({'item_id': items_to_predict, 'predicted_rating': predictions})
    # Ordenar las recomendaciones por calificación prevista
    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)
    return recommendations.head(n)

# Obtener recomendaciones para user1
recommendations_for_user1 = get_recommendations('user1', df, algo, n=5)
print(f'Recomendaciones para user1:')
print(recommendations_for_user1)
