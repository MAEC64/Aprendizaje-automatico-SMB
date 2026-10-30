"""Paso 8: junta en un solo archivo todas las cifras que pide el reporte de avance.

Corre los pasos 01 a 06 primero. Luego:
    python src/08_resumen_reporte.py
Resultado:
    resultados/reporte_cifras.md  (ábrelo y copia cada cifra en la sección indicada)
"""
import json
import re
from datetime import date

import pandas as pd

import comun

R = comun.RESULTADOS
salida = []


def escribir(texto=""):
    salida.append(texto)


def tabla_md(df):
    """Tabla en formato Markdown sin dependencias extra."""
    def fmt(v):
        return f"{v:,.1f}" if isinstance(v, float) else str(v)
    cols = list(df.columns)
    lineas = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lineas += ["| " + " | ".join(fmt(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join(lineas)


def leer_json(nombre):
    ruta = R / nombre
    return json.loads(ruta.read_text()) if ruta.exists() else {}


df = pd.read_parquet(comun.TABLA_FOTOGRAMAS)
carga = leer_json("resumen_carga.json")
reg = leer_json("resumen_regresion.json")

escribir(f"# Cifras para el reporte de avance (generado el {date.today()})")
escribir("Copia cada cifra en la sección del reporte con el mismo número.")

# ------------------------------------------------------------------ 3.3
escribir("\n## 3.3 Dataset utilizado")
ep = df.groupby("episodio").agg(resultado=("resultado", "first"), duracion_s=("t", "max"),
                                x_max=("x", "max"))
gan = ep[ep.resultado == "win"]
escribir(f"- Fotogramas del nivel {comun.NIVEL} leídos: {len(df):,} "
         f"(archivos encontrados: {carga.get('archivos_encontrados', '?'):,}; "
         f"ilegibles: {carga.get('ilegibles', '?')})")
escribir(f"- Episodios: {len(ep)} ({(ep.resultado == 'win').sum()} completados, "
         f"{(ep.resultado == 'fail').sum()} fallidos)")
if len(gan):
    escribir(f"- Episodios completados: duración media {gan.duracion_s.mean():.1f} s "
             f"(mín. {gan.duracion_s.min():.1f}, máx. {gan.duracion_s.max():.1f}); "
             f"x final media {gan.x_max.mean():,.0f} px")
escribir(f"- Columnas de la tabla de trabajo ({len(df.columns)}): " + ", ".join(df.columns))
escribir("\nMuestra de la tabla (5 fotogramas de un episodio):\n")
cols = ["episodio", "fotograma", "t", "x", "en_aire", "velocidad", "estado", "accion",
        "accion_agente", "resultado"]
primero = df[df.episodio == df.episodio.iloc[0]]
escribir(tabla_md(primero[cols].iloc[::60].head(5).round(2)))
ruta_ej = R / "ejemplo_fotograma.png"
if ruta_ej.exists():
    escribir(f"\nFotograma de ejemplo: `{ruta_ej.name}` (insértalo en 3.3 como figura).")

# ------------------------------------------------------------------ 3.4
escribir("\n## 3.4 Preparación de los datos")
n_desc = carga.get("acciones_descartadas", int((df.accion_agente < 0).sum()))
visitados = df.estado.nunique()
filas_prep = [
    ["Bytes extra en la RAM por el modo texto de Windows",
     "Se deshizo la conversión \\r\\n → \\n y se verificó mundo/nivel contra el nombre del archivo",
     f"{carga.get('ram_corregidas', '?'):,} fotogramas corregidos; "
     f"{carga.get('incoherentes', '?')} incoherentes después de corregir"],
    ["Acciones que el agente no puede usar (izquierda, START, SELECT, saltar sin avanzar)",
     "Se marcaron con -1 y no se usan para la tabla Q",
     f"{n_desc:,} fotogramas ({n_desc / len(df):.1%})"],
    ["Episodios con saltos bruscos de x (tubería)", "Se excluyeron de la regresión",
     f"{len(reg.get('excluidos_por_salto_x', []))} episodio(s)"],
    ["Episodios fallidos (curvas incompletas)",
     "La regresión usa solo episodios completados" if reg.get("solo_ganados") else
     "Pocos completados: se usaron todos los episodios",
     f"{len(gan)} episodio(s) completado(s)"],
    ["Fotogramas consecutivos casi idénticos",
     f"Se tomó 1 de cada {reg.get('cada_n_fotogramas', 10)} (regresión) y 1 de cada 4 (tabla Q)",
     f"{reg.get('filas_antes_submuestreo', '?'):,} → {reg.get('filas_usadas', '?'):,} filas en la regresión"],
    ["Tiempo en escalas muy distintas para el polinomio",
     "Escalado MinMax dentro del pipeline (se ajusta solo con entrenamiento en cada pliegue)",
     "Sin fuga de información hacia validación"],
    ["RAM de 2 048 bytes, demasiado grande para una tabla Q",
     "Estado discreto: en el aire, distancia al enemigo, terreno delante y velocidad",
     f"90 estados posibles; el humano visitó {visitados}"],
]
escribir(tabla_md(pd.DataFrame(filas_prep, columns=["Situación detectada", "Tratamiento realizado",
                                                     "Resultado"])))

# ------------------------------------------------------------------ 3.5
escribir("\n## 3.5 Análisis exploratorio")
nombres = {-1: "descartada", **dict(enumerate(comun.ACCIONES))}
dist = df.accion_agente.map(nombres).value_counts(normalize=True).mul(100).round(1)
escribir("Distribución de acciones del humano (% de fotogramas): " +
         "; ".join(f"{k} {v}%" for k, v in dist.items()))
frec = df.estado.value_counts().head(3)
escribir("Estados más frecuentes: " +
         "; ".join(f"{comun.describir_estado(s)} ({n / len(df):.1%})" for s, n in frec.items()))
escribir(f"Velocidad (RAM 0x0057): media {df.velocidad.mean():.1f}, "
         f"mediana {df.velocidad.median():.0f}, máx. {df.velocidad.max()}")
escribir("Figuras para insertar: `eda_curvas_avance.png`, `eda_acciones.png`, `eda_velocidad.png`")

# ------------------------------------------------------------------ 3.6
escribir("\n## 3.6 Diseño experimental")
escribir(f"- Regresión: {reg.get('episodios_entrenamiento', '?')} episodios de entrenamiento "
         f"({reg.get('filas_entrenamiento', '?'):,} filas), {reg.get('episodios_prueba', '?')} reservados "
         f"para prueba; validación cruzada GroupKFold en {reg.get('pliegues_validacion', '?')} pliegues; "
         f"semilla {reg.get('semilla', 0)}")
configs = sorted(R.glob("agente_*_s*_config.json"))
if configs:
    c = json.loads(configs[0].read_text())
    escribir(f"- Q-learning: {c['episodios']} episodios por agente, alfa={c['alfa']}, gamma={c['gamma']}, "
             f"epsilon {c['eps_ini']}→{c['eps_fin']} en el {c['frac_eps']:.0%} de los episodios, "
             f"acción repetida {c['repetir']} fotogramas, corte tras {c['paciencia']} decisiones sin "
             f"avanzar, evaluación cada {c['evaluar_cada']} episodios con {c['partidas_eval']} partidas")
    escribir("- Corridas: " + ", ".join(p.name.replace("_config.json", "") for p in configs))

# ------------------------------------------------------------------ 3.7
escribir("\n## 3.7 Resultados preliminares")
ruta_val = R / "regresion_validacion.csv"
if ruta_val.exists():
    val = pd.read_csv(ruta_val)
    escribir(f"\nRegresión (validación cruzada; modelo elegido: {reg.get('modelo_elegido', '?')}):\n")
    escribir(tabla_md(val))
filas_rl = []
for ruta in sorted(R.glob("agente_*_s*_evaluacion.csv")):
    agente, semilla = re.match(r"agente_(\w)_s(\d+)_", ruta.name).groups()
    ev = pd.read_csv(ruta).sort_values("episodio")
    ini, fin = ev.iloc[0], ev.iloc[-1]
    etiqueta0 = ("Referencia: agente aleatorio (A sin entrenar)" if agente == "A"
                 else "B sin entrenar (solo tabla humana)")
    filas_rl.append([etiqueta0, semilla, 0, round(ini.x_max_media, 1), f"{ini.completados:.0%}"])
    filas_rl.append([f"Agente {agente} entrenado", semilla, int(fin.episodio),
                     round(fin.x_max_media, 1), f"{fin.completados:.0%}"])
if filas_rl:
    escribir("\nQ-learning (evaluación sin exploración; distancia máxima media en px):\n")
    escribir(tabla_md(pd.DataFrame(filas_rl, columns=["Modelo / configuración", "Semilla",
                                                       "Episodios de entrenamiento",
                                                       "Distancia media (px)", "Completados"])))
ruta_comp = R / "comparacion_agentes.csv"
if ruta_comp.exists():
    escribir("\nComparación con la curva humana (06_evaluar.py):\n")
    escribir(tabla_md(pd.read_csv(ruta_comp).round(1)))
escribir("\nFiguras para insertar: `regresion_curvas.png`, `curvas_aprendizaje.png`, "
         "`agentes_vs_humano.png`")

(R / "reporte_cifras.md").write_text("\n".join(salida), encoding="utf-8")
print("\n".join(salida))
print(f"\nGuardado en {R / 'reporte_cifras.md'}")
