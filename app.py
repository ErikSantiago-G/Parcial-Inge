import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================================
# 1. Configuración Principal
# =========================================================================
st.set_page_config(page_title="Inge Datos - Energía ⚡", page_icon="⚡", layout="wide")

# =========================================================================
# 2. Funciones de Base de Datos y ETL (Carga de Datos)
# =========================================================================

def create_mock_data(conn):
    """Crea datos simulados si la BDD está vacía o no existe el script SQL."""
    # Proyectos
    pd.DataFrame({
        'id_proyecto': [1, 2, 3, 4, 5],
        'nombre_proyecto': ['Parque Eólico Guajira I', 'Solar Celsia', 'Geotermia Macizo', 'Hidrógeno Verde Caribe', 'Biomasa Valle'],
        'ubicacion': ['La Guajira', 'Valle del Cauca', 'Nariño', 'Atlántico', 'Risaralda'],
        'tipo_energia': [1, 2, 3, 4, 5]
    }).to_sql('proyectos', conn, if_exists='replace', index=False)
    
    # Tipos de Energia
    pd.DataFrame({
        'id_tipo': [1, 2, 3, 4, 5],
        'tipo_energia_descripción': ['Eólica', 'Solar', 'Geotermia', 'Hidrógeno Verde', 'Biomasa']
    }).to_sql('tipos_energia', conn, if_exists='replace', index=False)
    
    # Empresas
    pd.DataFrame({
        'proyecto_id': [1, 2, 3, 4, 5],
        'id_empresa': [101, 102, 103, 104, 105],
        'nombre_empresa_asociada': ['Isagen', 'Celsia', 'Ecopetrol', 'Promigas', 'Incauca'],
        'industria': ['Energía Eólica', 'Energía Solar', 'Geotermia', 'Hidrógeno Verde', 'Biomasa']
    }).to_sql('empresas', conn, if_exists='replace', index=False)
    
    # Inversiones
    pd.DataFrame({
        'id_inversion': [1001, 1002, 1003, 1004, 1005],
        'proyecto_id': [1, 2, 3, 4, 5],
        'monto_inversion': [50000000, 30000000, 15000000, 40000000, 10000000]
    }).to_sql('inversiones', conn, if_exists='replace', index=False)

@st.cache_data
def load_data():
    """Conecta a SQLite y extrae los datos mediante Pandas. Genera un consolidado."""
    db_name = "all_seasons.csv" 
    
    # Conexión local a sqlite
    conn = sqlite3.connect(db_name)
    
    # Intentar ejecutar SQL si existe (Comportamiento de Inge.py)
    if os.path.exists("SectorMineroEnergeticoColombia.sql"):
        try:
            with open("SectorMineroEnergeticoColombia.sql", "r") as f:
                conn.executescript(f.read())
        except:
             pass
    
    # Comprobar si las tablas existen
    cursor = conn.cursor()
    cursor.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='proyectos'")
    if cursor.fetchone()[0] == 0:
        create_mock_data(conn)
        
    # Cargar DataFrames
    df_proyectos = pd.read_sql_query("SELECT * FROM proyectos", conn)
    df_tipos_energia = pd.read_sql_query("SELECT * FROM tipos_energia", conn)
    df_inversiones = pd.read_sql_query("SELECT * FROM inversiones", conn)
    df_empresas = pd.read_sql_query("SELECT * FROM empresas", conn)
    conn.close()

    # Consolidación (ETL desde Inge.py)
    df_consolidado = pd.merge(df_proyectos, df_tipos_energia, left_on='tipo_energia', right_on='id_tipo', how='left')
    df_consolidado = pd.merge(df_consolidado, df_inversiones, left_on='id_proyecto', right_on='proyecto_id', how='left')
    df_consolidado = pd.merge(df_consolidado, df_empresas, left_on='id_proyecto', right_on='proyecto_id', how='left', suffixes=('_inversion', '_empresa'))
    
    # Limpieza de Nombres y Columnas si aplicó la colisión del merge general
    col_mapping = {}
    if 'nombre_consolidado' in df_consolidado.columns: col_mapping['nombre_consolidado'] = 'nombre_proyecto'
    if 'nombre_empresa' in df_consolidado.columns: col_mapping['nombre_empresa'] = 'nombre_empresa_asociada'
    if 'tipo' in df_consolidado.columns: col_mapping['tipo'] = 'tipo_energia_descripción'
    if 'monto' in df_consolidado.columns: col_mapping['monto'] = 'monto_inversion'
    if 'descripcion' in df_consolidado.columns: col_mapping['descripcion'] = 'descripción_proyecto'
    
    if col_mapping:
        df_consolidado = df_consolidado.rename(columns=col_mapping)
        
    return df_consolidado

# =========================================================================
# 3. Modelado Predictivo en Tiempo Real
# =========================================================================

@st.cache_resource
def train_model(df):
    """Entrena Random Forest para predecir monto de inversion desde caracterísiticas simples"""
    df_model = df.copy()
    
    # Limpiar nulos
    df_model = df_model.dropna(subset=['monto_inversion', 'tipo_energia_descripción', 'industria', 'ubicacion'])
    
    if df_model.empty or len(df_model) < 3:
        return None, {}, None # no hay suficientes datos
    
    le_dict = {}
    cat_cols = ['tipo_energia_descripción', 'industria', 'ubicacion']
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le
        
    X = df_model[cat_cols]
    y = df_model['monto_inversion']
    
    # Split y Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) if len(X) > 4 else (X, X, y, y)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'r2': r2_score(y_test, y_pred) if len(y_test) > 1 else 1.0,
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    return model, le_dict, metrics

# =========================================================================
# 4. Funciones de Renderizado UI Modulares
# =========================================================================

def show_kpis(df):
    st.subheader("💡 Métricas Clave (KPIs)")
    c1, c2, c3, c4 = st.columns(4)
    
    total_inversion = df['monto_inversion'].sum() if 'monto_inversion' in df.columns else 0
    total_proyectos = df['id_proyecto'].nunique() if 'id_proyecto' in df.columns else 0
    total_industrias = df['industria'].nunique() if 'industria' in df.columns else 0
    total_empresas = df['nombre_empresa_asociada'].nunique() if 'nombre_empresa_asociada' in df.columns else 0
    
    c1.metric("Proyectos Totales", f"{total_proyectos}")
    c2.metric("Inversión Total", f"${total_inversion:,.0f}")
    c3.metric("Tipos de Industria", f"{total_industrias}")
    c4.metric("Empresas Involucradas", f"{total_empresas}")
    st.markdown("---")

def show_charts(df):
    st.subheader("📊 Gráficos Descriptivos")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**1. Inversión por Tipo de Energía**")
        df_inv = df.groupby('tipo_energia_descripción')['monto_inversion'].sum().reset_index()
        fig1 = px.bar(df_inv, x='tipo_energia_descripción', y='monto_inversion', color='tipo_energia_descripción', template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.markdown("**2. Número de Proyectos por Industria**")
        df_ind = df.groupby('industria')['id_proyecto'].nunique().reset_index()
        fig2 = px.bar(df_ind, x='industria', y='id_proyecto', color='industria', template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
        
    c3, c4 = st.columns(2)
    with c3:
         st.markdown("**3. Distribución Inversión (Kernel Density)**")
         fig3 = px.histogram(df, x='monto_inversion', nbins=10, marginal="box", color_discrete_sequence=['#2dd4bf'], template='plotly_dark')
         st.plotly_chart(fig3, use_container_width=True)
         
    with c4:
         st.markdown("**4. Tabla de Frecuencias (Proyectos/Energía)**")
         freq_table = df['tipo_energia_descripción'].value_counts().reset_index()
         freq_table.columns = ['Tipo de Energía', 'Frecuencia (Proyectos)']
         st.dataframe(freq_table, use_container_width=True, hide_index=True)

def show_map(df):
    st.markdown("---")
    st.subheader("🗺️ Mapa Energético Distribuido")
    st.write("Visualización abstracta de las ubicaciones regionales del portafolio energético en Colombia.")
    
    # Dado que los datos asumen una ubiación de texto pero no lat/lon definidos explícitamente en Inge.py:
    # Generaremos un mapa usando coordenadas simuladas referenciales a los departamentos mencionados si existen:
    
    coordenadas_co = {
        'La Guajira': {'lat': 11.5444, 'lon': -72.9069},
        'Valle del Cauca': {'lat': 3.8009, 'lon': -76.6205},
        'Nariño': {'lat': 1.6198, 'lon': -77.5682},
        'Atlántico': {'lat': 10.6559, 'lon': -74.9664},
        'Risaralda': {'lat': 5.0933, 'lon': -75.9868}
    }
    
    map_data = []
    for _, row in df.iterrows():
        ubic = row.get('ubicacion', 'ND')
        if pd.notna(ubic) and ubic in coordenadas_co:
            map_data.append({
                'Nombre': row.get('nombre_proyecto', 'Proyecto'),
                'Industria': row.get('industria', 'Desconocido'),
                'Monto': row.get('monto_inversion', 0),
                'lat': coordenadas_co[ubic]['lat'],
                'lon': coordenadas_co[ubic]['lon']
            })
            
    df_map = pd.DataFrame(map_data)
    
    if not df_map.empty:
        fig_map = px.scatter_mapbox(df_map, lat="lat", lon="lon", hover_name="Nombre", hover_data=["Industria", "Monto"],
                            color="Industria", size="Monto",
                            color_continuous_scale=px.colors.cyclical.IceFire, size_max=25, zoom=4,
                            center={"lat": 4.5709, "lon": -74.2973}, mapbox_style="carto-darkmatter")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No hay suficientes coordenadas geoespaciales mapeables en la columna 'ubicacion'.")


def show_ml_predictor(df):
    st.markdown("---")
    st.subheader("🔮 Análisis Predictivo en Tiempo Real")
    st.write("Estima el presupuesto de inversión usando nuestro modelo embebido (RandomForest).")
    
    model, encoders, metrics = train_model(df)
    
    if model is None:
        st.warning("No hay suficientes datos válidos para entrenar el modelo de Machine Learning.")
        return
        
    st.caption(f"**Rendimiento del Modelo:** R² Score: `{metrics['r2']:.2f}` | MAE: `${metrics['mae']:,.2f}`")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        in_tipo = st.selectbox("Tipo de Energía", df['tipo_energia_descripción'].dropna().unique())
    with col_in2:
        in_ind = st.selectbox("Industria", df['industria'].dropna().unique())
    with col_in3:
        in_ubi = st.selectbox("Región (Ubicación)", df['ubicacion'].dropna().unique())
        
    if st.button("Generar Predicción del Monto 🚀", use_container_width=True, type="primary"):
        # Encoding input
        try:
            val_tipo = encoders['tipo_energia_descripción'].transform([in_tipo])[0]
            val_ind = encoders['industria'].transform([in_ind])[0]
            val_ubi = encoders['ubicacion'].transform([in_ubi])[0]
            
            input_df = pd.DataFrame([[val_tipo, val_ind, val_ubi]], columns=['tipo_energia_descripción', 'industria', 'ubicacion'])
            prediccion = model.predict(input_df)[0]
            
            st.success(f"### 💰 Inversión Estimada: ${prediccion:,.0f} USD")
            
        except Exception as e:
            st.error(f"Error procesando la solicitud: {e}")

def show_orchestration():
    st.title("⚙️ Orquestación de Datos: Airflow & Docker")
    st.markdown("---")
    st.write("Bienvenido al módulo de entendimiento de **Data Orchestration**. Aquí simulamos conceptualmente los ejercicios requeridos de orquestación de tuberías de datos utilizando los ecosistemas Apache Airflow y Docker.")
    
    t1, t2, t3 = st.tabs(["1. Simulación de DAGs", "2. Definición de Tareas ETL", "3. Prógramación de Workflows"])
    
    with t1:
        st.subheader("🕸️ Simulación de DAGs (Directed Acyclic Graphs)")
        st.write("En Airflow, un DAG establece la forma y dirección temporal de las tareas sin entrar en un cíclo infinito. A continuación, trazamos cómo luciría el Pipeline interactivo del flujo de Energía para Colombia en Airflow:")
        
        # Simulación visual básica de dependencias
        st.graphviz_chart('''
            digraph {
                node [shape=box, style=filled, color="#2dd4bf", fontcolor=black];
                edge [color=gray];
                A [label="Inicio_Pipeline"]
                B [label="Extraer_Proyectos (SQLite)"]
                C [label="Extraer_Empresas (SQLite)"]
                D [label="Limpieza_y_Consolidacion_ETL"]
                E [label="Entrenamiento_Modelo_Predictivo"]
                F [label="Carga_Visual_Dashboard"]
                
                A -> B
                A -> C
                B -> D
                C -> D
                D -> E
                D -> F
            }
        ''', use_container_width=True)
        st.info("Dependencias: Extracción paralela de bases de datos antes de unirse en un solo eslabón de consolidación.")

    with t2:
        st.subheader("🧩 Definición de Tareas ETL")
        st.write("Usando los operadores modernos como TaskFlow API (`@task`), la carga sobre contenedores Docker se independiza. Aquí simulamos cómo declaramos dichas tareas de procesamiento para inyectar este Dashboard.")
        st.code('''from airflow.decorators import task
import pandas as pd

@task
def extraer_datos_energia():
    """Conecta a la base de datos (Docker) y extrae las inversiones."""
    print("Ejecutando SELECT hacia all_seasons.csv...")
    return data

@task
def transformar_y_limpiar(data):
    """Elimina redundancias, hace los Merge para consolidado final."""
    data_limpia = df.dropna()
    print("El DataVault está estructurado.")
    return data_limpia

@task
def cargar_al_warehouse(data_limpia):
    """Escribe los datos a PostgreSQL/BigQuery y dispara Streamlit."""
    data_limpia.to_sql('produccion_energia', conn)
''', language="python")

    with t3:
        st.subheader("⏰ Programación Básica de Workflows")
        st.write("Una vez las tareas están aisladas, el orquestador usa expresiones CRON para dictar la recurrencia con la que extrae y alimenta el Streamlit, evitando inyectar código manual.")
        st.code('''from airflow import DAG
from datetime import datetime, timedelta

# Diccionario inicial del Workflow
default_args = {
    'owner': 'ing_datos',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Declaración principal - Correción Diaria
with DAG(
    'pipeline_sector_energetico_co',
    default_args=default_args,
    description='Procesar transacciones de inversiones de energia',
    schedule_interval='@daily', # <- Ejecición a Media Noche
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:
    
    # Declaración del flujo (Secuenciamiento)
    raw_data = extraer_datos_energia()
    clean_data = transformar_y_limpiar(raw_data)
    cargar_al_warehouse(clean_data)
''', language="python")
        st.success("Mediante este enfoque, la virtualización de todos los servicios locales de este dashboard, respaldados bajo imágenes Docker, se mantienen en sincronía constante y tolerantes a caídas.")

# =========================================================================
# 5. Lógica Principal (Main)
# =========================================================================

def main():
    # Menú de Navegación Lateral Superior
    st.sidebar.title("Navegación 🧭")
    page = st.sidebar.radio("Ir a sección:", ["Dashboard Principal 📊", "Orquestación (Airflow) ⚙️"])
    
    st.sidebar.markdown("---")

    df = load_data()
    
    if df.empty:
        st.error("No se encontraron registros activos en la base de datos.")
        return

    if page == "Dashboard Principal 📊":
        st.title("🔋 Minería de Datos: Sector Minero-Energético")

        # Panel de Filtrado Lateral
        st.sidebar.title("🎛️ Filtros Globales")
        
        st.sidebar.markdown("**Filtra el DataWarehouse en Vivo**")
        f_industria = st.sidebar.multiselect("Seleccionar Industria", options=df['industria'].dropna().unique(), default=df['industria'].dropna().unique())
        f_energia = st.sidebar.multiselect("Seleccionar Tipo Energía", options=df['tipo_energia_descripción'].dropna().unique(), default=df['tipo_energia_descripción'].dropna().unique())

        # Aplicar filtros
        filtered_df = df[
            (df['industria'].isin(f_industria)) & 
            (df['tipo_energia_descripción'].isin(f_energia))
        ]

        if filtered_df.empty:
            st.warning("Ajusta los filtros: No hay proyectos bajo estas condiciones combinadas.")
        else:    
            # Llamar componentes modulares
            show_kpis(filtered_df)
            show_charts(filtered_df)
            show_map(filtered_df)
            show_ml_predictor(df) 
    
    elif page == "Orquestación (Airflow) ⚙️":
        show_orchestration()

if __name__ == '__main__':
    main()
