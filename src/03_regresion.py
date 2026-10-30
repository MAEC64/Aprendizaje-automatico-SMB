"""Paso 3: regresión de la posición x de Mario contra el tiempo (partidas humanas).

Compara, de menos a más flexibles:
    referencia (media) -> predice siempre la posición media (punto de comparación)
    lineal             -> x = b0 + b1*t
    polinomial         -> grado elegido por validación cruzada
    por nudos (splines)-> número de nudos elegido por validación cruzada

Los episodios se separan 80 % entrenamiento / 20 % prueba. Los modelos se
comparan y eligen SOLO con validación cruzada por episodio (GroupKFold) dentro
del 80 % de entrenamiento. El 20 % de prueba queda reservado para el reporte
final: solo se usa si se ejecuta con --prueba.

Uso:
    python src/03_regresion.py            # reporte de avance (prueba reservada)
    python src/03_regresion.py --prueba   # reporte final (evalúa en prueba)
Resultados (en resultados/):
    regresion_validacion.csv, regresion_curvas.png, curva_humana.joblib,
    resumen_regresion.json, episodios_prueba.csv  (+ regresion_prueba.csv con --prueba)
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, SplineTransformer

import comun

CADA_N_FOTOGRAMAS = 10  # fotogramas seguidos son casi iguales: tomamos 1 de cada 10
SEMILLA = 0
METRICAS = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}


CAIDA_MAXIMA = 100  # px por debajo del máximo alcanzado que todavía cuentan como recorrido normal


def quitar_fuera_de_recorrido(df, resumen):
    """Quita los fotogramas en los que x cae de golpe por debajo de lo ya recorrido.

    Pasa al entrar al cuarto de monedas bajo tierra (la x vuelve a empezar en esa
    zona) o al cambiar de nivel tras la bandera (la x vuelve a 40). Se quitan solo
    esos fotogramas; el resto del episodio se conserva.
    """
    maximo = df.groupby("episodio").x.cummax()
    fuera = df.x < maximo - CAIDA_MAXIMA
    afectados = df[fuera].groupby("episodio")
    for ep, g in afectados:
        inicio = g.index[0]
        previo = df.loc[:inicio].iloc[-2] if len(df.loc[:inicio]) > 1 else g.iloc[0]
        print(f"  {ep}: x cae de {int(previo.x)} a {int(g.x.iloc[0])} px en t = {g.t.iloc[0]:.1f} s; "
              f"se quitan {len(g)} fotogramas")
    resumen["fotogramas_fuera_recorrido"] = int(fuera.sum())
    resumen["episodios_con_caida_x"] = int(afectados.ngroups)
    if fuera.any():
        print(f"Quité {int(fuera.sum())} fotogramas fuera del recorrido principal "
              f"en {afectados.ngroups} episodio(s).")
    return df[~fuera]


def preparar_datos(resumen):
    df = pd.read_parquet(comun.TABLA_FOTOGRAMAS).sort_values(["episodio", "fotograma"])
    df = df.reset_index(drop=True)
    ganados = df[df.resultado == "win"]
    resumen["solo_ganados"] = bool(ganados.episodio.nunique() >= 3)
    if resumen["solo_ganados"]:
        df = ganados
    else:
        print("AVISO: hay menos de 3 episodios completados; uso todos los episodios.")
    resumen["episodios_regresion"] = int(df.episodio.nunique())

    df = quitar_fuera_de_recorrido(df, resumen)
    if df.episodio.nunique() < 2:
        raise SystemExit("No quedan suficientes episodios para la regresión (se necesitan al menos 2). "
                         "Revisa resultados/fotogramas_1-1.parquet y la gráfica eda_curvas_avance.png.")
    resumen["filas_antes_submuestreo"] = len(df)
    df = df[df.fotograma % CADA_N_FOTOGRAMAS == 0]
    resumen["filas_usadas"] = len(df)
    return df


def candidatos():
    lineal = make_pipeline(LinearRegression())
    return {
        "referencia (media)": (make_pipeline(DummyRegressor(strategy="mean")),
                               {"dummyregressor__strategy": ["mean"]}),
        "lineal": (lineal, {"linearregression__fit_intercept": [True]}),
        # El escalado va DENTRO del pipeline: en cada partición se ajusta solo con entrenamiento
        "polinomial": (make_pipeline(MinMaxScaler(), PolynomialFeatures(include_bias=False),
                                     LinearRegression()),
                       {"polynomialfeatures__degree": list(range(2, 9))}),
        "nudos (splines)": (make_pipeline(MinMaxScaler(),
                                          SplineTransformer(degree=3, extrapolation="linear"),
                                          LinearRegression()),
                            {"splinetransformer__n_knots": list(range(3, 16))}),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prueba", action="store_true",
                   help="evaluar en el 20 %% reservado (solo para el reporte final)")
    args = p.parse_args()

    resumen = {"cada_n_fotogramas": CADA_N_FOTOGRAMAS, "semilla": SEMILLA}
    df = preparar_datos(resumen)
    X, y, grupos = df[["t"]].to_numpy(), df["x"].to_numpy(), df["episodio"].to_numpy()

    # División por EPISODIO (no por fotograma): 80 % entrenamiento, 20 % prueba reservada
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA)
    i_ent, i_pru = next(gss.split(X, y, grupos))
    ep_ent, ep_pru = sorted(set(grupos[i_ent])), sorted(set(grupos[i_pru]))
    pd.Series(ep_pru, name="episodio").to_csv(comun.RESULTADOS / "episodios_prueba.csv", index=False)
    resumen.update({"episodios_entrenamiento": len(ep_ent), "episodios_prueba": len(ep_pru),
                    "filas_entrenamiento": len(i_ent), "filas_prueba": len(i_pru)})
    pliegues = min(5, len(ep_ent))
    resumen["pliegues_validacion"] = pliegues
    print(f"Episodios: {len(ep_ent)} de entrenamiento (validación cruzada en {pliegues} pliegues), "
          f"{len(ep_pru)} reservados para prueba")

    filas, modelos = [], {}
    for nombre, (modelo, rejilla) in candidatos().items():
        busqueda = GridSearchCV(modelo, rejilla, cv=GroupKFold(n_splits=pliegues),
                                scoring=METRICAS, refit="rmse")
        busqueda.fit(X[i_ent], y[i_ent], groups=grupos[i_ent])
        i, cv = busqueda.best_index_, busqueda.cv_results_
        params = {k.split("__")[1]: v for k, v in busqueda.best_params_.items()
                  if k.split("__")[1] in ("degree", "n_knots")}
        filas.append({"modelo": nombre,
                      "hiperparametro": ", ".join(f"{k}={v}" for k, v in params.items()) or "-",
                      "RMSE_val_px": -cv["mean_test_rmse"][i],
                      "RMSE_val_desv": cv["std_test_rmse"][i],
                      "MAE_val_px": -cv["mean_test_mae"][i],
                      "R2_val": cv["mean_test_r2"][i]})
        modelos[nombre] = busqueda.best_estimator_  # reajustado con todo el entrenamiento

    tabla = pd.DataFrame(filas).set_index("modelo").round(3)
    tabla.to_csv(comun.RESULTADOS / "regresion_validacion.csv")
    print("\nMétricas de VALIDACIÓN CRUZADA (promedio de los pliegues):\n", tabla.to_string())

    # Elegimos por RMSE de validación (sin la referencia); si uno más simple está a menos de 5 %,
    # preferimos el simple
    reales = [n for n in modelos if n != "referencia (media)"]
    mejor = tabla.loc[reales, "RMSE_val_px"].min()
    elegido = next(n for n in reales if tabla.loc[n, "RMSE_val_px"] <= mejor * 1.05)
    resumen["modelo_elegido"] = elegido
    print(f"\nModelo elegido como curva humana de referencia: {elegido}")
    joblib.dump({"nombre": elegido, "modelo": modelos[elegido]},
                comun.RESULTADOS / "curva_humana.joblib")

    if args.prueba:
        filas_p = []
        for nombre, modelo in modelos.items():
            pred = modelo.predict(X[i_pru])
            filas_p.append({"modelo": nombre, "RMSE_prueba_px": root_mean_squared_error(y[i_pru], pred),
                            "MAE_prueba_px": mean_absolute_error(y[i_pru], pred),
                            "R2_prueba": r2_score(y[i_pru], pred)})
        prueba = pd.DataFrame(filas_p).set_index("modelo").round(3)
        prueba.to_csv(comun.RESULTADOS / "regresion_prueba.csv")
        print("\nMétricas en PRUEBA (reporte final):\n", prueba.to_string())
    else:
        print(f"\nPrueba reservada: {len(ep_pru)} episodios (lista en episodios_prueba.csv). "
              "Usa --prueba solo para el reporte final.")

    (comun.RESULTADOS / "resumen_regresion.json").write_text(json.dumps(resumen, indent=2))

    # Gráfico: datos de ENTRENAMIENTO + las curvas (la prueba no se muestra)
    t = np.linspace(0, df.t.max(), 400).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(X[i_ent, 0], y[i_ent], s=4, color="gray", alpha=0.3, label="datos de entrenamiento")
    for nombre, modelo in modelos.items():
        estilo = "--" if nombre.startswith("referencia") else "-"
        ax.plot(t, modelo.predict(t), estilo, lw=2,
                label=f"{nombre} (RMSE val. {tabla.loc[nombre, 'RMSE_val_px']:.0f} px)")
    ax.set(xlabel="Tiempo (s)", ylabel="Posición x (px)",
           title="Regresión del avance humano en el nivel 1-1 (validación cruzada por episodio)")
    ax.legend(fontsize=9)
    fig.savefig(comun.RESULTADOS / "regresion_curvas.png", dpi=120, bbox_inches="tight")
    print(f"Gráfico guardado en {comun.RESULTADOS / 'regresion_curvas.png'}")


if __name__ == "__main__":
    main()
