"""Prototipo de simulación 3D de flujos por partículas sobre un DEM real.

Lanza partículas desde varias zonas altas del relieve y las deja escurrir por
gravedad y por el gradiente del terreno. La física corre en paralelo con
Taichi y la visualización es 3D e interactiva con PyVista.

Es un modelo cinemático simplificado, sin calibrar: sirve para visualizar
trayectorias de flujo, no para estimar alcances, caudales ni velocidades reales.

Uso:
    python simulacion_3d.py --dem ruta/al/dem.tif
    python simulacion_3d.py --dem ruta/al/dem.tif --factor 5 --particulas 60000 --focos 6 --semilla 42
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pyvista as pv
import rasterio
import taichi as ti

# Parámetros físicos del prototipo (adimensionales, no calibrados).
DT = 0.02
GRAVEDAD = 9.81
FACTOR_GRAVEDAD = 50.0
FRICCION_SUELO = 0.30
FUERZA_DESLIZAMIENTO = 0.2
REBOTE_BORDE = -0.5
PERCENTIL_ZONAS_ALTAS = 75
MARGEN_CELDAS = 5
NODATA_UMBRAL = -100.0


@dataclass
class Configuracion:
    ruta_dem: str
    factor_reduccion: int
    n_particulas: int
    max_focos: int
    semilla: int | None


def leer_argumentos() -> Configuracion:
    parser = argparse.ArgumentParser(description="Simulación 3D de flujos por partículas sobre un DEM.")
    parser.add_argument("--dem", required=True, help="Ruta al DEM en formato GeoTIFF.")
    parser.add_argument("--factor", type=int, default=10, help="Submuestreo del DEM (1 = resolución completa).")
    parser.add_argument("--particulas", type=int, default=40_000, help="Número de partículas.")
    parser.add_argument("--focos", type=int, default=4, help="Número máximo de zonas de inicio.")
    parser.add_argument("--semilla", type=int, default=None, help="Semilla aleatoria para reproducibilidad.")
    args = parser.parse_args()
    return Configuracion(args.dem, args.factor, args.particulas, args.focos, args.semilla)


def cargar_dem(ruta: str, factor: int) -> tuple[np.ndarray, float]:
    """Lee el DEM, lo submuestrea y reemplaza los valores sin dato por la cota mínima."""
    with rasterio.open(ruta) as src:
        dem = src.read(1).astype(np.float32)[::factor, ::factor]
    dem[dem < NODATA_UMBRAL] = np.nan
    cota_min = float(np.nanmin(dem))
    return np.nan_to_num(dem, nan=cota_min), cota_min


def elegir_focos(dem: np.ndarray, cota_min: float, max_focos: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Elige zonas de inicio al azar entre las celdas sobre el percentil 75 de altura, lejos de los bordes."""
    filas, columnas = dem.shape
    validos = dem > cota_min + 1.0
    umbral_alto = np.percentile(dem[validos], PERCENTIL_ZONAS_ALTAS)
    y_altos, x_altos = np.where((dem >= umbral_alto) & validos)

    dentro = (
        (x_altos > MARGEN_CELDAS) & (x_altos < columnas - MARGEN_CELDAS)
        & (y_altos > MARGEN_CELDAS) & (y_altos < filas - MARGEN_CELDAS)
    )
    x_ok, y_ok = x_altos[dentro], y_altos[dentro]

    if len(x_ok) == 0:
        print("Aviso: las cumbres están muy al borde; se usan focos en el centro del DEM.")
        x_ok = np.array([columnas // 2, columnas // 2 + 5, columnas // 2 - 5])
        y_ok = np.array([filas // 2, filas // 2 + 5, filas // 2 - 5])

    n_focos = min(max_focos, len(x_ok))
    idx = rng.choice(len(x_ok), n_focos, replace=False)
    return x_ok[idx], y_ok[idx]


def posiciones_iniciales(dem: np.ndarray, cota_min: float, x_focos: np.ndarray, y_focos: np.ndarray,
                         n_particulas: int, rng: np.random.Generator) -> np.ndarray:
    """Reparte las partículas entre los focos, dispersas alrededor de cada uno y sobre el terreno."""
    filas, columnas = dem.shape
    pos = np.zeros((n_particulas, 3), dtype=np.float32)
    grupos = np.array_split(np.arange(n_particulas), len(x_focos))

    for (x0, y0), ids in zip(zip(x_focos, y_focos), grupos):
        px = np.clip(x0 + rng.uniform(-3.0, 3.0, len(ids)), 1, columnas - 2)
        py = np.clip(y0 + rng.uniform(-3.0, 3.0, len(ids)), 1, filas - 2)
        fuera = dem[py.astype(int), px.astype(int)] <= cota_min + 1.0
        px[fuera], py[fuera] = x0, y0
        pos[ids, 0] = px
        pos[ids, 1] = py
        pos[ids, 2] = dem[py.astype(int), px.astype(int)] + rng.uniform(5, 15, len(ids))
    return pos


def construir_simulador(dem: np.ndarray, cota_min: float, pos_inicial: np.ndarray):
    """Crea los campos de Taichi y devuelve (pos, rapidez, paso), donde paso() avanza un DT."""
    filas, columnas = dem.shape
    n = pos_inicial.shape[0]
    piso = cota_min + 1.0

    pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
    vel = ti.Vector.field(3, dtype=ti.f32, shape=n)
    rapidez = ti.field(dtype=ti.f32, shape=n)
    elevacion = ti.field(dtype=ti.f32, shape=(columnas, filas))
    elevacion.from_numpy(dem.T.copy())
    pos.from_numpy(pos_inicial)

    @ti.kernel
    def paso():
        for i in pos:
            vel[i][2] -= GRAVEDAD * FACTOR_GRAVEDAD * DT
            prev_x, prev_y = pos[i][0], pos[i][1]
            pos[i] += vel[i] * DT

            # Rebote en los bordes del dominio.
            if pos[i][0] <= 1.0:
                pos[i][0] = 1.1
                vel[i][0] *= REBOTE_BORDE
            if pos[i][0] >= columnas - 2.0:
                pos[i][0] = columnas - 2.1
                vel[i][0] *= REBOTE_BORDE
            if pos[i][1] <= 1.0:
                pos[i][1] = 1.1
                vel[i][1] *= REBOTE_BORDE
            if pos[i][1] >= filas - 2.0:
                pos[i][1] = filas - 2.1
                vel[i][1] *= REBOTE_BORDE

            gx, gy = int(pos[i][0]), int(pos[i][1])
            z_terreno = elevacion[gx, gy]

            # Rebote en celdas sin dato (fuera de la zona de estudio).
            if z_terreno <= piso:
                pos[i][0], pos[i][1] = prev_x, prev_y
                vel[i][0] *= REBOTE_BORDE
                vel[i][1] *= REBOTE_BORDE
                gx, gy = int(pos[i][0]), int(pos[i][1])
                z_terreno = elevacion[gx, gy]

            # Contacto con el terreno: fricción y deslizamiento según el gradiente local.
            if pos[i][2] < z_terreno:
                pos[i][2] = z_terreno
                vel[i][2] = 0.0
                vel[i][0] *= FRICCION_SUELO
                vel[i][1] *= FRICCION_SUELO

                z_izq = elevacion[gx - 1, gy]
                z_der = elevacion[gx + 1, gy]
                z_arr = elevacion[gx, gy - 1]
                z_aba = elevacion[gx, gy + 1]
                if z_izq <= piso:
                    z_izq = z_terreno
                if z_der <= piso:
                    z_der = z_terreno
                if z_arr <= piso:
                    z_arr = z_terreno
                if z_aba <= piso:
                    z_aba = z_terreno

                vel[i][0] += (z_izq - z_der) * FUERZA_DESLIZAMIENTO * DT
                vel[i][1] += (z_arr - z_aba) * FUERZA_DESLIZAMIENTO * DT

            rapidez[i] = vel[i].norm()

    return pos, rapidez, paso


def ejecutar_visualizacion(dem: np.ndarray, pos, rapidez, paso) -> None:
    """Abre la escena 3D con botones de iniciar/pausar y cerrar, y corre el ciclo de animación."""
    filas, columnas = dem.shape
    xx, yy = np.meshgrid(np.arange(columnas, dtype=np.float32), np.arange(filas, dtype=np.float32))
    terreno = pv.StructuredGrid(xx, yy, dem)
    nube = pv.PolyData(pos.to_numpy())
    nube["Velocidad"] = np.zeros(nube.n_points)

    plotter = pv.Plotter()
    plotter.add_mesh(terreno, color="saddlebrown", opacity=0.9, show_edges=False)
    plotter.add_mesh(nube, scalars="Velocidad", cmap="turbo", point_size=4, show_scalar_bar=True, clim=[0, 15])
    plotter.set_scale(zscale=0.006)
    plotter.view_yz()
    plotter.camera.azimuth = 15
    plotter.camera.elevation = 25
    plotter.camera.zoom(2.2)
    plotter.add_axes()

    estado = {"corriendo": False, "cerrar": False}

    def alternar(valor: bool) -> None:
        estado["corriendo"] = valor

    def cerrar(_valor: bool) -> None:
        estado["cerrar"] = True

    plotter.add_checkbox_button_widget(alternar, value=False, position=(20, 70), size=30,
                                       color_on="green", color_off="grey")
    plotter.add_text("Iniciar / Pausar", position=(60, 75), font_size=12, color="white")
    plotter.add_checkbox_button_widget(cerrar, value=False, position=(20, 20), size=30,
                                       color_on="red", color_off="grey")
    plotter.add_text("Cerrar simulación", position=(60, 25), font_size=12, color="white")

    print("Ventana abierta. Presiona el botón verde para iniciar el flujo.")
    plotter.show(interactive_update=True, auto_close=False)

    while not estado["cerrar"]:
        if estado["corriendo"]:
            paso()
            nube.points = pos.to_numpy()
            nube["Velocidad"] = rapidez.to_numpy()
        plotter.update()
        time.sleep(0.01)

    print("Cerrando simulación...")
    plotter.close()


def main() -> None:
    cfg = leer_argumentos()
    ti.init(arch=ti.cpu)
    rng = np.random.default_rng(cfg.semilla)

    print("Cargando modelo de elevación...")
    dem, cota_min = cargar_dem(cfg.ruta_dem, cfg.factor_reduccion)
    x_focos, y_focos = elegir_focos(dem, cota_min, cfg.max_focos, rng)
    print(f"{len(x_focos)} focos de inicio preparados.")

    pos_inicial = posiciones_iniciales(dem, cota_min, x_focos, y_focos, cfg.n_particulas, rng)
    pos, rapidez, paso = construir_simulador(dem, cota_min, pos_inicial)
    ejecutar_visualizacion(dem, pos, rapidez, paso)


if __name__ == "__main__":
    main()
