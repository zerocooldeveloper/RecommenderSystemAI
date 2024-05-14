**Implementación de un Sistema de Recomendación Usando IA para Recomendaciones Personalizadas**

**Implementación de un Sistema de Recomendación**

Este documento explica cómo implementar y ejecutar un sistema de recomendación utilizando técnicas de inteligencia artificial. Utilizaremos Python y la biblioteca 'surprise' para construir un modelo de recomendación basado en el algoritmo de matriz de factorización SVD (Singular Value Decomposition).

**Archivos**

- recommender.py : Código fuente del sistema de recomendación.
- Implementacion\_Sistema\_Recomendacion\_Explicacion.pdf : Documento explicativo del código.

**Requisitos**

Para ejecutar este proyecto, necesitas tener instalado Python 3.x en tu sistema. Además, necesitas instalar las bibliotecas  pandas  y  scikit-surprise .

**Instalación**

**Paso 1: Clonar el Repositorio**

Primero, clona este repositorio en tu máquina local. Abre una terminal o símbolo del sistema y ejecuta el siguiente comando:

`   `bash

git clone https://github.com/zerocooldeveloper/RecommenderSystemAI.git

cd RecommenderSystemAI

**Paso 2: Crear y Activar un Entorno Virtual (opcional pero recomendado)**

Es  recomendable  utilizar  un  entorno  virtual  para  gestionar  las  dependencias  del  proyecto  sin interferir con otras instalaciones de Python en tu sistema.

\#### En Windows:

1. Crea el entorno virtual:       bash

   `   `python -m venv venv

2. Activa el entorno virtual:       bash

   `   `venv\Scripts\activate

   #### En macOS y Linux:

1. Crea el entorno virtual:       bash

   `   `python3 -m venv venv

2. Activa el entorno virtual:

`      `bash

`   `source venv/bin/activate

**Paso 3: Instalar las Dependencias**

Instala las bibliotecas necesarias utilizando  pip . Ejecuta el siguiente comando en la terminal:

`   `bash

pip install pandas scikit-surprise

**Paso 4: Ejecutar el Script**

Para  entrenar  el  modelo  y  generar  recomendaciones,  simplemente  ejecuta  el  script  recommender.py . Asegúrate de estar en el directorio del proyecto y ejecuta el siguiente comando:

`   `bash

python recommender.py

**Código del Script recommender.py**

A continuación se muestra el código completo del script  recommender.py :

`   `python

import pandas as pd

from surprise import Dataset, Reader, SVD

from surprise.model\_selection import cross\_validate

- Datos de ejemplo: usuarios, películas y calificaciones (1-5)

data = {

`    `'user\_id': ['user1', 'user1', 'user1', 'user2', 'user2', 'user2', 'user3', 'user3', 'user3', 'user4', 'user4', 'user4', 'user5', 'user5', 'user5'],

`    `'item\_id': ['movie1', 'movie2', 'movie3', 'movie1', 'movie2', 'movie4', 'movie2', 'movie3', 'movie5', 'movie1', 'movie3', 'movie6', 'movie4', 'movie5', 'movie6'],

`    `'rating': [5, 3, 4, 4, 2, 5, 5, 2, 3, 2, 3, 4, 3, 5, 4]

}

- Convertir los datos en un DataFrame

df = pd.DataFrame(data)

print("Datos de calificaciones de usuarios y películas:") print(df)

- Cargar los datos en la biblioteca surprise

reader = Reader(rating\_scale=(1, 5))

data = Dataset.load\_from\_df(df[['user\_id', 'item\_id', 'rating']], reader)

- Entrenar el modelo utilizando SVD trainset = data.build\_full\_trainset() algo = SVD()
- Realizar la validación cruzada

  cross\_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

- Entrenar el modelo en el conjunto completo de datos algo.fit(trainset)
- Hacer predicciones para un usuario específico

  user\_id = 'user1'

  item\_id = 'movie4'

  pred = algo.predict(user\_id, item\_id)

  print(f'Predicción de calificación para {user\_id} y {item\_id}: {pred.est:.2f}')

- Generar recomendaciones para un usuario específico

def get\_recommendations(user\_id, df, algo, n=5):

- Obtener todos los ítems (películas) únicos

`    `all\_items = df['item\_id'].unique()

- Obtener los ítems que el usuario ya ha calificado

`    `user\_items = df[df['user\_id'] == user\_id]['item\_id'].unique()

- Filtrar los ítems que el usuario no ha calificado

`    `items\_to\_predict = [item for item in all\_items if item not in user\_items]

- Predecir calificaciones para los ítems no calificados

`    `predictions = [algo.predict(user\_id, item).est for item in items\_to\_predict]

- Crear un DataFrame con las predicciones

`    `recommendations = pd.DataFrame({'item\_id': items\_to\_predict, 'predicted\_rating': predictions})

- Ordenar las recomendaciones por calificación prevista

`    `recommendations = recommendations.sort\_values(by='predicted\_rating', ascending=False)

`    `return recommendations.head(n)

- Obtener recomendaciones para user1

recommendations\_for\_user1 = get\_recommendations('user1', df, algo, n=5) print(f'Recomendaciones para user1:')

print(recommendations\_for\_user1)

**Conclusión**

Este código implementa un sistema de recomendación utilizando técnicas de inteligencia artificial, específicamente el algoritmo de matriz de factorización SVD. El proceso incluye la preparación de los datos, la carga en el formato adecuado, el entrenamiento del modelo, la validación cruzada para evaluar el rendimiento y la generación de recomendaciones personalizadas. Este enfoque es similar a lo que utilizan plataformas como Netflix para proporcionar recomendaciones personalizadas a sus usuarios, mejorando significativamente la experiencia del usuario.
