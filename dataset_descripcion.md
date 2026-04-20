# Información y Diccionario de Datos

## 📌 Descripción del Dataset
El dataset del **Sector Minero Energético de Colombia** contiene información estructurada simulada de inversiones en múltiples tipos de energía e industrias ubicadas en el país. Sus tablas, originalmente estructuradas en SQL relacionales, se importan para propósitos analíticos orientados a BI (Business Intelligence) y minería de datos (exploración predictiva).

**Fuente Original:** Extranjero / Kaggle / Simulación Académica (Base de Datos SQLite).

---

## 📖 Diccionario de Variables (Tabla Consolidada)

Bajo la estructura del Data Warehouse generado en el `EDA` y el `ETL` dentro del script base:

| Variable | Tipo de Dato (Pandas) | Descripción | Origen / Tabla |
| :--- | :---: | :--- | :--- |
| **`id_proyecto`** | `int64` | Identificador único de cada proyecto energético. | `proyectos` |
| **`nombre_proyecto`** | `object` (String) | El nombre descriptivo del proyecto en ejecución. | `proyectos` |
| **`ubicacion`** | `object` (String) | Región, ciudad o departamento de Colombia del proyecto. | `proyectos` |
| **`tipo_energia`** | `int64` | Relación foránea del ID energético del proyecto. | `proyectos` |
| **`id_tipo`** | `int64` | Identificador primario para la categoría de energía. | `tipos_energia` |
| **`tipo_energia_descripción`** | `object` (String) | Nombre de la energía producida (Ej. Eólica, Solar, Geotermia). | `tipos_energia` |
| **`id_inversion`** | `int64` | Identificador único del registro de aportes / financiamiento. | `inversiones` |
| **`monto_inversion`** | `float64` | Cantidad total invertida en el proyecto (USD / COP). | `inversiones` |
| **`id_empresa`** | `int64` | ID de la entidad corporativa dueña del proyecto. | `empresas` |
| **`nombre_empresa_asociada`** | `object` (String) | Razón social corporativa de los inversionistas. | `empresas` |
| **`industria`** | `object` (String) | Naturaleza industrial / extractiva de la empresa. | `empresas` |

> *Nota: Durante el ETL las filas y sentencias redundantes como `proyecto_id_inversion` o `proyecto_id_empresa` fueron descartadas para garantizar un esquema limpio y central.*
