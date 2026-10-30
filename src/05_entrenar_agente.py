"""Paso 5: entrena un agente con Q-learning en el nivel 1-1.

    Agente A: tabla Q empieza en ceros (aprende desde cero).
    Agente B: tabla Q empieza con resultados/q_inicial_humana.npy (paso 4).

Regla de actualización (Q-learning):
    Q(s,a) <- Q(s,a) + alfa * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Uso (cada combinación en su propia terminal para aprovechar varios núcleos):
    python src/05_entrenar_agente.py --agente A --semilla 0
    python src/05_entrenar_agente.py --agente B --semilla 0
Prueba rápida:
    python src/05_entrenar_agente.py --agente A --semilla 0 --episodios 50 --evaluar-cada 25
Resultados (en resultados/, con prefijo agente_<A|B>_s<semilla>):
    _episodios.csv, _evaluacion.csv, _q.npy, _trayectoria.csv
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import comun


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--agente", choices=["A", "B"], required=True)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--episodios", type=int, default=2000)
    p.add_argument("--alfa", type=float, default=0.1, help="tasa de aprendizaje")
    p.add_argument("--gamma", type=float, default=0.9, help="factor de descuento")
    p.add_argument("--eps-ini", type=float, default=1.0)
    p.add_argument("--eps-fin", type=float, default=0.05)
    p.add_argument("--frac-eps", type=float, default=0.6,
                   help="fracción de episodios en la que epsilon baja de eps-ini a eps-fin")
    p.add_argument("--repetir", type=int, default=4, help="fotogramas que dura cada acción")
    p.add_argument("--paciencia", type=int, default=60,
                   help="decisiones sin avanzar antes de cortar el episodio (Mario atascado)")
    p.add_argument("--evaluar-cada", type=int, default=100)
    p.add_argument("--partidas-eval", type=int, default=10)
    return p.parse_args()


def mejor_accion(fila_q, rng):
    """argmax con desempate al azar (si no, con Q en ceros siempre elegiría 'nada')."""
    return int(rng.choice(np.flatnonzero(fila_q == fila_q.max())))


def jugar(env, q, args, rng, epsilon, aprender, guardar_trayectoria=False, noop_inicial=0):
    """Juega un episodio. Si aprender=True, actualiza la tabla Q."""
    env.reset()
    for _ in range(noop_inicial):          # arranques distintos para evaluar
        env.step(0)
    ram = comun.ram_de(env)
    s = comun.estado(ram)
    total, x_max, sin_avance, fotograma = 0.0, 0, 0, 0
    trayectoria, info, terminal, fin, a_previa = [], {}, False, False, 0
    while not fin:
        a = int(rng.integers(comun.N_ACCIONES)) if rng.random() < epsilon else mejor_accion(q[s], rng)
        r = 0.0
        for a_fotograma in comun.acciones_por_fotograma(a, a_previa, ram, args.repetir):
            _, ri, term, trunc, info = env.step(a_fotograma)
            r += ri
            fotograma += 1
            if guardar_trayectoria:
                trayectoria.append((fotograma / comun.FPS, info["x_pos"]))
            if term or trunc:
                terminal, fin = term, True
                break
        s2 = comun.estado(ram)
        if info["x_pos"] > x_max:
            x_max, sin_avance = info["x_pos"], 0
        else:
            sin_avance += 1
        if sin_avance >= args.paciencia:
            fin = True                      # atascado: cortamos sin tratarlo como muerte
        if aprender:
            objetivo = r if terminal else r + args.gamma * q[s2].max()
            q[s, a] += args.alfa * (objetivo - q[s, a])
        s, total, a_previa = s2, total + r, a
    return total, x_max, bool(info.get("flag_get", False)), fotograma, trayectoria


def main():
    args = argumentos()
    rng = np.random.default_rng(args.semilla)
    prefijo = comun.RESULTADOS / f"agente_{args.agente}_s{args.semilla}"

    if args.agente == "B":
        q = np.load(comun.RESULTADOS / "q_inicial_humana.npy")
    else:
        q = np.zeros((comun.N_ESTADOS, comun.N_ACCIONES))

    with open(f"{prefijo}_config.json", "w") as fh:  # configuración usada (para el reporte)
        json.dump(vars(args), fh, indent=2)

    env = comun.crear_entorno()
    env.reset(seed=args.semilla)
    registro, evaluaciones, inicio = [], [], time.time()
    pasos_eps = max(1, int(args.episodios * args.frac_eps))

    def evaluar(ep):
        res = [jugar(env, q, args, rng, 0.0, False, noop_inicial=int(rng.integers(0, 31)))
               for _ in range(args.partidas_eval)]
        evaluaciones.append({"episodio": ep,
                             "x_max_media": np.mean([r[1] for r in res]),
                             "x_max_mejor": max(r[1] for r in res),
                             "completados": np.mean([r[2] for r in res]),
                             "recompensa_media": np.mean([r[0] for r in res])})
        e = evaluaciones[-1]
        print(f"[{args.agente} s{args.semilla}] ep {ep:5d} | eval x media {e['x_max_media']:7.1f} "
              f"| mejor {e['x_max_mejor']:5d} | completados {e['completados']:.0%} "
              f"| {time.time() - inicio:6.0f} s", flush=True)

    evaluar(0)  # desempeño antes de entrenar (el agente B ya trae conocimiento humano)
    for ep in range(1, args.episodios + 1):
        epsilon = max(args.eps_fin, args.eps_ini - (args.eps_ini - args.eps_fin) * ep / pasos_eps)
        total, x_max, completo, fotogramas, _ = jugar(env, q, args, rng, epsilon, True)
        registro.append({"episodio": ep, "recompensa": total, "x_max": x_max,
                         "completado": completo, "epsilon": round(epsilon, 4),
                         "fotogramas": fotogramas})
        if ep % args.evaluar_cada == 0 or ep == args.episodios:
            evaluar(ep)
            pd.DataFrame(registro).to_csv(f"{prefijo}_episodios.csv", index=False)
            pd.DataFrame(evaluaciones).to_csv(f"{prefijo}_evaluacion.csv", index=False)
            np.save(f"{prefijo}_q.npy", q)

    # Trayectoria final sin exploración, para compararla con la curva humana
    *_, trayectoria = jugar(env, q, args, rng, 0.0, False, guardar_trayectoria=True)
    pd.DataFrame(trayectoria, columns=["t", "x"]).to_csv(f"{prefijo}_trayectoria.csv", index=False)
    env.close()
    print(f"Listo. Archivos guardados con el prefijo {prefijo}_*")


if __name__ == "__main__":
    main()
