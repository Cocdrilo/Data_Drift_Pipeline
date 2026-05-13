import requests
import random

url = "http://localhost:8000/predict/batch"

instances = []
for _ in range(300):
    features = [random.uniform(5000.0, 10000.0) for _ in range(10)]
    instances.append({"features": features})

payload = {"instances": instances}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    print(f"Petición exitosa. {response.json().get('count')} registros anómalos inyectados.")
except requests.exceptions.RequestException as e:
    print(f"Error en la petición: {e}")