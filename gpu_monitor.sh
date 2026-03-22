#!/bin/bash

# Arquivo de saída
LOGFILE="gpu_temp.log"

# Cria/limpa o arquivo de log com cabeçalho
echo "timestamp,gpu_temp_C" > "$LOGFILE"

echo "Iniciando monitoramento da GPU... (salvando em $LOGFILE)"
echo "Pressione Ctrl+C para parar."

# Detecta tipo de GPU e usa o comando adequado
if command -v nvidia-smi &> /dev/null; then
    echo "Detectado: GPU NVIDIA"
    while true; do
        TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$TEMP" >> "$LOGFILE"
        sleep 1
    done

elif command -v sensors &> /dev/null; then
    echo "Detectado: GPU AMD ou outro compatível com lm-sensors"
    while true; do
        TEMP=$(sensors | grep -m 1 -Eo '\+?[0-9]+(\.[0-9]+)?°C' | head -n 1 | tr -d '+°C')
        echo "$(date '+%Y-%m-%d %H:%M:%S'),$TEMP" >> "$LOGFILE"
        sleep 1
    done

else
    echo "Nenhum comando compatível encontrado (nvidia-smi ou sensors)."
    exit 1
fi
