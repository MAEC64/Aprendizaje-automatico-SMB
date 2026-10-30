#!/usr/bin/env bash
# Entrena los agentes A y B con varias semillas, todos en paralelo (un proceso cada uno).
# Uso:  ./entrenar_todos.sh [episodios] [semillas...]
#       ./entrenar_todos.sh 2000 0 1 2      (por defecto: 2000 episodios, semillas 0 1 2)
set -e
cd "$(dirname "$0")"
EPISODIOS=${1:-2000}
shift || true
SEMILLAS=${@:-0 1 2}
mkdir -p resultados
for s in $SEMILLAS; do
  for ag in A B; do
    echo "Lanzando agente $ag, semilla $s ($EPISODIOS episodios)"
    nohup .venv/bin/python -u src/05_entrenar_agente.py --agente "$ag" --semilla "$s" \
      --episodios "$EPISODIOS" > "resultados/log_${ag}_s${s}.txt" 2>&1 &
  done
done
echo "Sigue el avance con:  tail -f resultados/log_A_s0.txt"
wait
echo "Terminado. Ahora corre:  .venv/bin/python src/06_evaluar.py"
