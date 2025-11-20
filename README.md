## Hi there 👋

### Script: `deep_freqai_hyperopt.py`

Este repositorio incluye un script pensado para automatizar búsquedas
profundas de hiperparámetros en Freqtrade + FreqAI utilizando el
algoritmo TPE de Hyperopt. A continuación encontrarás una guía paso a
paso para ponerlo en marcha.

#### 1. Requisitos previos

1. Tener instalado `freqtrade` con soporte para FreqAI y haber
   inicializado el entorno (`freqtrade create-userdir`, etc.).
2. Contar con un archivo de configuración base, por ejemplo
   `user_data/config.json`, y una estrategia FreqAI válida.
3. Preparar los datos históricos necesarios (por ejemplo con
   `freqtrade download-data`).

#### 2. Ejecutar una optimización básica

```bash
python scripts/deep_freqai_hyperopt.py \
  --config user_data/config.json \
  --strategy MyFreqAIStrategy \
  --max-evals 150 \
  --study-path user_data/hyperopt/freqai_trials.json
```

Qué hace cada parámetro principal:

| Parámetro | Descripción |
|-----------|-------------|
| `--config` | Ruta al archivo de configuración base de Freqtrade. |
| `--strategy` | Nombre de la estrategia FreqAI que vas a optimizar. |
| `--max-evals` | Número máximo de evaluaciones que realizará Hyperopt. |
| `--study-path` | Archivo JSON donde se guardan los resultados para poder reanudar más adelante. |

#### 3. Opciones adicionales útiles

- `--hyperopt-loss`: permite cambiar la clase de pérdida utilizada por
  Freqtrade (por defecto `SortinoHyperOptLossDaily`).
- `--minimum-avg-profit`: descarta pruebas cuyo beneficio medio quede
  por debajo del umbral indicado.
- `--startup-candles` y `--candle-limit`: controlan la preparación de
  velas y la cantidad máxima de datos procesados en cada backtest.
- `--dry-run`: ejecuta todo el flujo sin llamar realmente a `freqtrade`
  (útil para probar que la configuración general es correcta).

#### 4. Reanudar una sesión

El archivo indicado en `--study-path` guarda el estado de Hyperopt
después de cada iteración. Si detienes el proceso, basta con relanzar el
script con los mismos parámetros y reutilizará las evaluaciones previas
antes de continuar.

#### 5. Registros y depuración

El script escribe mensajes informativos en la consola. Si deseas más
detalle puedes exportar `LOG_LEVEL=DEBUG` antes de ejecutarlo para
obtener información extendida sobre cada intento de optimización.
