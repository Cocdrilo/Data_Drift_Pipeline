# Guía de Validación de Arquitectura: Bucle MLOps de Auto-Curación (Continuous Training)

## Fase 0: Inicialización del Ecosistema

Antes de comenzar la demostración, debemos asegurar que tanto la infraestructura local como el puente de CI/CD están activos.

### 1. Levantar los microservicios

Abre una terminal en la raíz del proyecto y arranca la orquestación.

```bash
docker compose up -d
```

### 2. Activar el Agente de Integración Continua (Self-Hosted Runner)

En una nueva pestaña de la terminal, navega a la carpeta donde instalaste el agente de GitHub y ejecútalo.

```bash
# Cambia la ruta por la carpeta donde descargaste el runner
cd ~/actions-runner 
./run.sh
```

**Indicador de éxito:** La terminal mostrará `Connected to GitHub` y `Listening for Jobs`.

---

## Fase 1: Comprobación del Estado Base (Baseline)

Demostración de que el sistema se encuentra en estado óptimo con los datos de referencia iniciales.

### 1. Verificar Grafana

Accede a `http://localhost:3000`. Muestra el panel principal de monitorización.

- **Resultado esperado:** El nivel de deriva (*Drift Level*) está en `0` (**Sano/Verde**).

### 2. Verificar Prometheus

Accede a `http://localhost:9090/alerts`.

- **Resultado esperado:** La regla `CriticalDataDrift` aparece en estado `Inactive` (**Verde**).

---

## Fase 2: Simulación de Entorno Hostil (Data Drift)

Forzamos un cambio drástico en la distribución de los datos de entrada para evaluar la reactividad del sistema.

### 1. Inyectar ruido estadístico

En una terminal en la raíz del proyecto, ejecuta el script de ataque.

```bash
python src/force_drift.py
```

> Este script envía 300 peticiones anómalas a la API, contaminando el dataset de producción local `production_dataset.csv`.

---

## Fase 3: Detección y Escalamiento de Alertas

El sistema debe identificar la anomalía y escalar el problema sin intervención humana.

### 1. Detección del Monitor

Espera a que el contenedor `monitor` realice su siguiente ciclo de evaluación (o fórza su ejecución con `docker compose restart monitor`).

#### Observar Grafana

- El nivel de deriva saltará a `2` (**Crítico/Rojo**).

### 2. Escalamiento Temporal (Alertmanager)

Observa la pestaña de alertas de Prometheus (`http://localhost:9090/alerts`).

- **Resultado esperado:** La regla `CriticalDataDrift` pasará a estado `Pending` y, tras el tiempo configurado, a estado `Firing` (**Rojo**).

- **Acción interna:** Alertmanager dispara el Webhook hacia nuestra API.

---

## Fase 4: Auto-Curación (Self-Healing)

El hito principal del proyecto. El sistema coordina el entrenamiento de un nuevo modelo adaptado a la nueva realidad de los datos.

### 1. Observar la ejecución remota

Ve a la pestaña **Actions** en tu repositorio de GitHub.

- **Resultado esperado:** Verás que el workflow `Continuous Training Pipeline (CT)` se ha iniciado automáticamente.

### 2. Observar la ejecución local

Vuelve a la pestaña de tu terminal donde está corriendo `./run.sh`.

- **Resultado esperado:** Verás a GitHub dándole órdenes en tiempo real a tu ordenador. Observarás cómo se ejecuta `train.py`, cómo MLflow registra la nueva versión del modelo, y cómo se limpia el `production_dataset.csv`.

### 3. Recarga en Caliente (Hot Reload)

El propio workflow enviará una señal a tu API para que cargue el nuevo modelo en memoria sin detener el servicio.

### 4. Despliegue Continuo (CD)

Finalmente, GitHub empaquetará el sistema actualizado y subirá la nueva imagen a GitHub Packages (`ghcr.io/...`).

---

## Fase 5: Resolución y Retorno a la Normalidad

Validación de que el sistema ha asimilado los nuevos patrones y se ha estabilizado.

### 1. Verificación final

Espera al siguiente ciclo del Monitor (o reinícialo manualmente).

#### Observar Grafana

- Al haberse entrenado con los datos anómalos y actualizado la referencia, el Monitor considerará que la crisis ha pasado.
- El gráfico bajará automáticamente a `0` (**Sano/Verde**).

#### Observar Prometheus

- La alerta volverá a estado `Inactive`.
