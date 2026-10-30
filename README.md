# mario-ml

Proyecto: una IA aprende a jugar el nivel 1-1 de Super Mario Bros con
Q-learning, usando partidas humanas (dataset de Rafael C. Pinto) para una curva de
referencia (regresión lineal, polinomial y por nudos) y para inicializar la tabla Q.

Probado en Python 3.14 (Ubuntu 26.04 LTS).

Inicio rápido:
Descomprimir el dataset en /datos

    sudo apt install -y python3-venv python3-dev build-essential git git-lfs 7zip
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python src/00_probar_entorno.py

Orden: 01_cargar_dataset -> 02_exploracion -> 03_regresion -> 04_tabla_q_humana
-> 05_entrenar_agente (o ./entrenar_todos.sh) -> 06_evaluar -> 07_grabar_video
-> 08_resumen_reporte (cifras para el reporte en resultados/reporte_cifras.md)
