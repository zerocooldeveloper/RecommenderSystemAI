from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Implementación de un Sistema de Recomendación', 0, 1, 'C')
        self.cell(0, 10, 'Usando IA para Recomendaciones Personalizadas', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Introducción
pdf.chapter_title('Introducción')
intro = (
    "En este documento, explicaremos paso a paso cómo implementar un sistema de recomendación "
    "utilizando técnicas de inteligencia artificial. Utilizaremos Python y la biblioteca "
    "'surprise' para construir un modelo de recomendación basado en el algoritmo de matriz de "
    "factorización SVD (Singular Value Decomposition).\n"
)
pdf.chapter_body(intro)

# Datos y Preparación
pdf.chapter_title('Datos y Preparación')
data_prep = (
    "Primero, definimos un conjunto de datos de ejemplo que contiene las calificaciones que "
    "los usuarios han dado a varias películas. Convertimos estos datos en un DataFrame de pandas "
    "para facilitar su manipulación.\n\n"
    "Código:\n"
    "```python\n"
    "import pandas as pd\n"
    "data = {\n"
    "    'user_id': ['user1', 'user1', 'user1', 'user2', 'user2', 'user2', 'user3', 'user3', "
    "'user3', 'user4', 'user4', 'user4', 'user5', 'user5', 'user5'],\n"
    "    'item_id': ['movie1', 'movie2', 'movie3', 'movie1', 'movie2', 'movie4', 'movie2', "
    "'movie3', 'movie5', 'movie1', 'movie3', 'movie6', 'movie4', 'movie5', 'movie6'],\n"
    "    'rating': [5, 3, 4, 4, 2, 5, 5, 2, 3, 2, 3, 4, 3, 5, 4]\n"
    "}\n"
    "df = pd.DataFrame(data)\n"
    "print('Datos de calificaciones de usuarios y películas:')\n"
    "print(df)\n"
    "```\n\n"
    "Explicación:\n"
    "- `import pandas as pd`: Importa la biblioteca pandas para manipular los datos.\n"
    "- `data`: Define un diccionario con los datos de ejemplo.\n"
    "- `pd.DataFrame(data)`: Convierte el diccionario en un DataFrame de pandas.\n"
    "- `print(df)`: Muestra los datos en formato de tabla.\n"
)
pdf.chapter_body(data_prep)

# Biblioteca `surprise`
pdf.chapter_title('Biblioteca `surprise`')
surprise_intro = (
    "Usamos la biblioteca `surprise` para construir el modelo de recomendación. Primero, cargamos "
    "los datos en un formato que `surprise` pueda entender.\n\n"
    "Código:\n"
    "```python\n"
    "from surprise import Dataset, Reader\n"
    "reader = Reader(rating_scale=(1, 5))\n"
    "data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)\n"
    "```\n\n"
    "Explicación:\n"
    "- `from surprise import Dataset, Reader`: Importa las clases necesarias de la biblioteca `surprise`.\n"
    "- `Reader(rating_scale=(1, 5))`: Define la escala de calificaciones.\n"
    "- `Dataset.load_from_df(...)`: Carga los datos del DataFrame en el formato requerido por `surprise`.\n"
)
pdf.chapter_body(surprise_intro)

# Entrenamiento del Modelo
pdf.chapter_title('Entrenamiento del Modelo')
model_training = (
    "Entrenamos un modelo de recomendación utilizando el algoritmo SVD. Primero, construimos el "
    "conjunto de entrenamiento completo y luego entrenamos el modelo. También realizamos una "
    "validación cruzada para evaluar el rendimiento del modelo.\n\n"
    "Código:\n"
    "```python\n"
    "from surprise import SVD\n"
    "from surprise.model_selection import cross_validate\n"
    "trainset = data.build_full_trainset()\n"
    "algo = SVD()\n"
    "cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)\n"
    "algo.fit(trainset)\n"
    "```\n\n"
    "Explicación:\n"
    "- `from surprise import SVD`: Importa el algoritmo SVD.\n"
    "- `from surprise.model_selection import cross_validate`: Importa la función de validación cruzada.\n"
    "- `data.build_full_trainset()`: Construye el conjunto de entrenamiento completo.\n"
    "- `algo = SVD()`: Crea una instancia del algoritmo SVD.\n"
    "- `cross_validate(...)`: Realiza la validación cruzada para evaluar el modelo.\n"
    "- `algo.fit(trainset)`: Entrena el modelo con el conjunto de entrenamiento.\n"
)
pdf.chapter_body(model_training)

# Predicciones
pdf.chapter_title('Predicciones')
predictions = (
    "Una vez entrenado el modelo, podemos hacer predicciones de calificaciones para usuarios y "
    "películas específicos. Aquí, predecimos la calificación que el usuario 'user1' daría a la "
    "película 'movie4'.\n\n"
    "Código:\n"
    "```python\n"
    "user_id = 'user1'\n"
    "item_id = 'movie4'\n"
    "pred = algo.predict(user_id, item_id)\n"
    "print(f'Predicción de calificación para {user_id} y {item_id}: {pred.est:.2f}')\n"
    "```\n\n"
    "Explicación:\n"
    "- `algo.predict(user_id, item_id)`: Predice la calificación que el usuario daría a la película.\n"
    "- `pred.est`: Obtiene la calificación predicha.\n"
)
pdf.chapter_body(predictions)

# Generación de Recomendaciones
pdf.chapter_title('Generación de Recomendaciones')
recommendations = (
    "Podemos generar recomendaciones para un usuario específico filtrando las películas que el "
    "usuario no ha calificado y prediciendo las calificaciones para esas películas. Luego, ordenamos "
    "las predicciones y seleccionamos las mejores recomendaciones.\n\n"
    "Código:\n"
    "```python\n"
    "def get_recommendations(user_id, df, algo, n=5):\n"
    "    all_items = df['item_id'].unique()\n"
    "    user_items = df[df['user_id'] == user_id]['item_id'].unique()\n"
    "    items_to_predict = [item for item in all_items if item not in user_items]\n"
    "    predictions = [algo.predict(user_id, item).est for item in items_to_predict]\n"
    "    recommendations = pd.DataFrame({'item_id': items_to_predict, 'predicted_rating': predictions})\n"
    "    recommendations = recommendations.sort_values(by='predicted_rating', ascending=False)\n"
    "    return recommendations.head(n)\n"
    "\n"
    "recommendations_for_user1 = get_recommendations('user1', df, algo, n=5)\n"
    "print(f'Recomendaciones para user1:')\n"
    "print(recommendations_for_user1)\n"
    "```\n\n"
    "Explicación:\n"
    "- `get_recommendations(...)`: Función para generar recomendaciones.\n"
    "- `all_items`: Lista de todas las películas únicas.\n"
    "- `user_items`: Lista de películas que el usuario ya ha calificado.\n"
    "- `items_to_predict`: Películas que el usuario no ha calificado.\n"
    "- `algo.predict(...)`: Predice las calificaciones para las películas no calificadas.\n"
    "- `recommendations.sort_values(...)`: Ordena las predicciones de mayor a menor.\n"
    "- `recommendations.head(n)`: Retorna las `n` mejores recomendaciones.\n"
)
pdf.chapter_body(recommendations)

# Conclusión
pdf.chapter_title('Conclusión')
conclusion = (
    "Este código implementa un sistema de recomendación utilizando técnicas de inteligencia artificial, "
    "específicamente el algoritmo de matriz de factorización SVD. El proceso incluye la preparación de los datos, "
    "la carga en el formato adecuado, el entrenamiento del modelo, la validación cruzada para evaluar el rendimiento "
    "y la generación de recomendaciones personalizadas.\n"
    "\n"
    "Este enfoque es similar a lo que utilizan plataformas como Netflix para proporcionar recomendaciones personalizadas "
    "a sus usuarios, mejorando significativamente la experiencia del usuario.\n"
)
pdf.chapter_body(conclusion)

# Guardar el archivo PDF
pdf.output('/Users/zerocool/Documents/Implementacion_Sistema_Recomendacion_Explicacion.pdf')
