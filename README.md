# Simulación 3D de flujos en quebradas

Prototipo en Python que simula y visualiza en 3D el escurrimiento de flujos por las quebradas de un modelo digital de elevación (DEM) real. Forma parte de una plataforma de visualización de quebradas, en desarrollo como insumo para un Sistema de Alerta Temprana (SAT) en el sector Lo Moscoso, comuna de Placilla (Región de O'Higgins, Chile).

## Qué hace

1. Lee un DEM en GeoTIFF con `rasterio`, lo submuestrea y descarta las celdas sin dato.
2. Elige al azar varias zonas de inicio entre las celdas sobre el percentil 75 de altura, alejadas de los bordes.
3. Lanza decenas de miles de partículas desde esos focos y las hace avanzar con una física simple, paralelizada en `taichi`:
   - gravedad;
   - fricción al contacto con el terreno;
   - deslizamiento según el gradiente local de elevación;
   - rebote en los bordes del dominio y en las celdas sin dato.
4. Muestra el terreno y la nube de partículas, coloreada por velocidad, en una escena 3D interactiva de `pyvista`, con botones para iniciar o pausar y para cerrar.

## Alcance

Es un modelo **cinemático simplificado y sin calibrar**. Sirve para visualizar trayectorias y zonas de concentración de flujo, no para estimar alcances, caudales ni velocidades reales. Los parámetros físicos (fricción, fuerza de deslizamiento, factor de gravedad) son adimensionales y están definidos como constantes al inicio de `simulacion_3d.py`.

La modelación hidráulica cuantitativa del estudio se hace aparte, con HEC-HMS y HEC-RAS 2D.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate      # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python simulacion_3d.py --dem ruta/al/dem.tif
```

Opciones:

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--dem` | (obligatorio) | Ruta al DEM en GeoTIFF |
| `--factor` | 10 | Submuestreo del DEM (1 = resolución completa) |
| `--particulas` | 40000 | Número de partículas |
| `--focos` | 4 | Número máximo de zonas de inicio |
| `--semilla` | aleatoria | Semilla para reproducir la misma elección de focos |

El DEM no se incluye en el repositorio. Para el estudio se usó el DEM ALOS PALSAR RTC, disponible en [ASF Data Search](https://search.asf.alaska.edu).

## Próximos pasos

- Permitir que el usuario defina los parámetros de activación de las quebradas (precipitación, focos, propiedades del flujo).
- Calibrar la física con los resultados de la modelación en HEC-RAS 2D.

## Autor

Nicolás Santana Martínez · [LinkedIn](https://www.linkedin.com/in/nicol%C3%A1s-santana2003)
