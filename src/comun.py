"""Funciones compartidas por todos los scripts del proyecto.

La idea clave: el dataset guarda la RAM de la consola (2048 bytes) y el
emulador nos da acceso a esa misma RAM. Por eso el estado del juego se
calcula con la MISMA función para las partidas humanas y para el agente.
Direcciones según https://datacrystal.tcrf.net/wiki/Super_Mario_Bros.:RAM_map
"""
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- rutas
RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos"
RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(exist_ok=True)
TABLA_FOTOGRAMAS = RESULTADOS / "fotogramas_1-1.parquet"

FPS = 60          # el juego corre a 60 fotogramas por segundo
NIVEL = "1-1"     # nivel con el que trabaja todo el proyecto
MITAD_NIVEL = 1600  # x aproximada de la mitad del nivel 1-1 (la meta está cerca de x = 3160)

# ------------------------------------------------------ direcciones RAM
PAGINA_X = 0x006D       # "página" (pantalla) en la que está Mario
X_PANTALLA = 0x0086     # x de Mario dentro de la página
Y_PANTALLA = 0x00CE     # y de Mario en pantalla (borde superior del sprite)
EN_AIRE = 0x001D        # 0 = en el suelo; 1, 2, 3 = en el aire / asta
VEL_X = 0x0057          # velocidad horizontal (con signo)
ENEMIGO_ACTIVO = 0x000F  # 5 bytes: 0x000F-0x0013
ENEMIGO_PAGINA = 0x006E  # 5 bytes: 0x006E-0x0072
ENEMIGO_X = 0x0087       # 5 bytes: 0x0087-0x008B
MAPA_BLOQUES = 0x0500    # 0x0500-0x069F: 2 páginas de 13 filas x 16 columnas
MUNDO = 0x075F           # mundo - 1
NIVEL_RAM = 0x075C       # nivel - 1


def posicion_x(ram):
    """Posición horizontal de Mario en el nivel, en píxeles."""
    return int(ram[PAGINA_X]) * 256 + int(ram[X_PANTALLA])


def bloque(ram, x, y):
    """Valor del bloque en la posición (x del nivel, y de pantalla). 0 = vacío."""
    pagina = (x // 256) % 2
    col = (x % 256) // 16
    fila = (y - 32) // 16
    if not 0 <= fila < 13:
        return 0
    return int(ram[MAPA_BLOQUES + pagina * 208 + fila * 16 + col])


# --------------------------------------------------- estado discreto
# 2 (aire) x 5 (enemigo) x 3 (terreno) x 3 (velocidad) = 90 estados
N_ESTADOS = 2 * 5 * 3 * 3
NOMBRES_ENEMIGO = ["lejos/ninguno", "48-96 px", "16-48 px", "0-16 px", "encima"]
NOMBRES_TERRENO = ["libre", "obstáculo", "hueco"]
NOMBRES_VELOCIDAD = ["quieto", "lento", "rápido"]


def rango_enemigo(ram, mx):
    """Distancia al enemigo activo más cercano por delante, en 5 rangos."""
    mejor = None
    for i in range(5):
        if ram[ENEMIGO_ACTIVO + i]:
            ex = int(ram[ENEMIGO_PAGINA + i]) * 256 + int(ram[ENEMIGO_X + i])
            dx = ex - mx
            if dx >= -16 and (mejor is None or dx < mejor):
                mejor = dx
    if mejor is None or mejor > 96:
        return 0
    if mejor > 48:
        return 1
    if mejor > 16:
        return 2
    if mejor >= 0:
        return 3
    return 4


def tipo_terreno(ram, mx):
    """0 = libre, 1 = obstáculo a la altura de Mario, 2 = hueco en el suelo."""
    fila_y = int(ram[Y_PANTALLA]) + 16           # fila del cuerpo de Mario
    for dx in (16, 32):
        if bloque(ram, mx + dx, fila_y) or bloque(ram, mx + dx, fila_y - 16):
            return 1
    suelo_y = 32 + 12 * 16                         # fila inferior de la pantalla
    if all(bloque(ram, mx + dx, suelo_y) == 0 for dx in (16, 32)):
        return 2
    return 0


def rango_velocidad(ram):
    v = int(np.int8(ram[VEL_X]))
    if v <= 4:
        return 0
    return 1 if v <= 24 else 2


def estado(ram):
    """Convierte la RAM (2048 bytes) en un número de estado entre 0 y 89."""
    mx = posicion_x(ram)
    aire = 1 if ram[EN_AIRE] != 0 else 0
    return ((aire * 5 + rango_enemigo(ram, mx)) * 3 + tipo_terreno(ram, mx)) * 3 + rango_velocidad(ram)


def describir_estado(s):
    vel = s % 3
    ter = (s // 3) % 3
    ene = (s // 9) % 5
    aire = s // 45
    return (f"{'aire' if aire else 'suelo'}, enemigo {NOMBRES_ENEMIGO[ene]}, "
            f"{NOMBRES_TERRENO[ter]}, {NOMBRES_VELOCIDAD[vel]}")


# ------------------------------------------------------------ acciones
# Mismo orden que gym_super_mario_bros.actions.RIGHT_ONLY
ACCIONES = ["nada", "derecha", "derecha+saltar", "derecha+correr", "derecha+correr+saltar"]
N_ACCIONES = len(ACCIONES)

CON_SALTO = {2: 1, 4: 3}   # acción con A -> la misma acción sin A


def acciones_por_fotograma(a, a_previa, ram, repetir):
    """Reparte una decisión del agente en `repetir` fotogramas.

    En la NES el salto se activa al PRESIONAR A, no al mantenerlo. Si A ya
    estaba pulsado en la decisión anterior y Mario está en el suelo, soltamos A
    durante el primer fotograma para que pueda volver a saltar. En el aire no se
    suelta, para no cortar la altura del salto.
    """
    if a in CON_SALTO and a_previa in CON_SALTO and ram[EN_AIRE] == 0:
        return [CON_SALTO[a]] + [a] * (repetir - 1)
    return [a] * repetir


# Bits del byte de acción del dataset (README del repositorio)
BIT_A, BIT_ARRIBA, BIT_IZQ, BIT_B = 128, 64, 32, 16
BIT_START, BIT_DER, BIT_ABAJO, BIT_SELECT = 8, 4, 2, 1


def accion_humana_a_indice(byte):
    """Traduce los botones del humano a una de las 5 acciones del agente.

    Devuelve -1 si la combinación no encaja (izquierda, START, SELECT,
    o saltar sin ir a la derecha).
    """
    if byte & (BIT_IZQ | BIT_START | BIT_SELECT):
        return -1
    der, a, b = bool(byte & BIT_DER), bool(byte & BIT_A), bool(byte & BIT_B)
    if not der:
        return 0 if not a else -1
    if a and b:
        return 4
    if b:
        return 3
    if a:
        return 2
    return 1


# ------------------------------------------------------------- entorno
def crear_entorno(modo_render=None):
    """Entorno del nivel 1-1 con las 5 acciones de RIGHT_ONLY.

    modo_render="human" abre una ventana para ver el juego (más lento).
    """
    import gymnasium as gym
    import gym_super_mario_bros  # noqa: F401  (registra los entornos)
    from gym_super_mario_bros.actions import RIGHT_ONLY
    from nes_py.wrappers import JoypadSpace

    env = gym.make(f"SuperMarioBros-{NIVEL}-v0", render_mode=modo_render)
    return JoypadSpace(env, RIGHT_ONLY)


def ram_de(env):
    """RAM actual del emulador (vista de numpy, 2048 bytes)."""
    return env.unwrapped.ram
