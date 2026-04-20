# Parcial Ingeniería de Datos: Sector Minero Energético ⚡

[![Python Development](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit UI](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ErikSantiago-G/Parcial-Inge.git)

¡Bienvenido al ecosistema analítico para la optimización de métricas del sector Energético y Minero de Colombia! 

Este proyecto está formulado para la extracción unificada (Data Warehouse) a partir de scripts SQL / SQLite y su representación estructurada en cuadros de mando avanzados. Evaluamos métricas como la densidad de inversión, tipos de industria (Solar, Hidrógeno, Eólica, etc.) y realizamos modelado predictivo de viabilidad para presupuestos empresariales.

## 🚀 Instalación y Despliegue

La plataforma se despliega en un entorno local ejecutando Streamlit. Instala las dependencias vía pip:

```bash
# 1. Instalar librerías
pip install -r requirements.txt

# 2. Inicializar la plataforma
streamlit run app.py
```

## 🏗️ Arquitectura del Proyecto

Para abolir esquemas "Spaghetti Code", el código ha sido distribuido en roles puras:
- **`app.py`:** Core del dashboard. Estructurado en funciones modulares `load_data()`, `train_model()`, `show_kpis()`, `show_charts()` y `show_map()`.
- **`index.html` & `styles.css`:** Interfaz web satélite (Landing Page) que expone e instruye el proceso transgeneracional de Transformación y Extracción de Datos (ETL + EDA).
- **`dataset_descripcion.md`:** Documento complementario definiendo diccionarios de variables provenientes originalmente de la base de datos `all_seasons.csv`.
- **`Inge.py`:** El script legado y original donde las peticiones analíticas bases están cimentadas. (Punto de inflexión lógico).

## 🔮 Funcionalidad Predictiva

Dentro del Panel Interactivo existe un subsistema embebido entrenado en un `RandomForestRegressor`. Este interpreta el *Tipo de Energía*, la *Sectorización Industrial* y el *Asentamiento Geográfico* calculando dinámicamente el presupuesto estimativo probable (`monto_inversion`) con métricas R2 validando en tiempo de compilación.

---
**Recursos Github:** [Repositorio Oficial (ErikSantiago-G)](https://github.com/ErikSantiago-G/Parcial-Inge.git)
