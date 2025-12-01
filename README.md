# Estrategia NFI_X7_Hopt

Este repositorio contiene la versión mejorada de la estrategia **NFI_X7_Hopt**. Sigue estos pasos para descargarla y usarla en Freqtrade.

## Descarga rápida del archivo
1. Sitúate en el directorio donde quieres guardarla.
2. Clona el repositorio o copia el archivo directamente:
   ```bash
   # Clonar el repositorio completo
   git clone https://github.com/<tu-usuario>/Romanjoseluis.git

   # O, si ya tienes el repo clonado, copia la estrategia al directorio de Freqtrade
   cp NFI_X7_Hopt.py /ruta/a/freqtrade/user_data/strategies/
   ```
3. (Opcional) Copia también la nota de auditoría si quieres conservar la documentación:
   ```bash
   cp AUDIT_NFI_X7_Hopt.md /ruta/a/freqtrade/user_data/strategies/
   ```

## Uso en Freqtrade
1. Abre tu archivo `config.json` y establece la estrategia:
   ```json
   "strategy": "NFI_X7_Hopt"
   ```
2. Si usas FreqAI, asegúrate de tener entrenados los modelos y de definir `freqai.enabled=true` en la configuración. La estrategia se comporta en modo *fail-closed* si la columna de predicción no está disponible.
3. Ajusta los parámetros opcionales en `strategy_parameters` o con tu pipeline de hyperopt según las necesidades de tu exchange.

## Referencias
- `NFI_X7_Hopt.py`: estrategia lista para Freqtrade con filtros de tendencia, control de rebuys y telemetría.
- `AUDIT_NFI_X7_Hopt.md`: resumen de la auditoría y mejoras implementadas.
