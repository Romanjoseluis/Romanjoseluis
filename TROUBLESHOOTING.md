# Guía rápida de solución de errores (FreqAI y `fillna`)

Este repositorio recoge notas breves para diagnosticar y corregir avisos como los que se muestran en los registros:

```
WARNING NFI_X7_Hopt - FreqAI KeyError para par FET/USDC: 'FET/USDC'
WARNING freqtrade.strategy.strategy_wrapper - Strategy caused the following exception: ValueError("Must specify a fill 'value' or 'method'.")
```

## 1. `FreqAI KeyError para par ...`
Estos avisos aparecen cuando el motor FreqAI no encuentra datos o un modelo entrenado para el par concreto. Suele ocurrir cuando el par no está incluido en la configuración de entrenamiento o no se ha descargado/entrenado correctamente.

**Pasos de comprobación y corrección**
1. Verifica que el par esté listado en `pairlists` y en la sección `freqai` de tu `config.json` (parámetros como `pair_whitelist`, `freqai.futures_leverage` o listas de entrenamiento deben incluirlo exactamente como lo devuelve el exchange).
2. Descarga datos históricos para el par y el timeframe que uses, por ejemplo:
   ```bash
   freqtrade download-data --config config.json --timeframe 5m --pairs FET/USDC
   ```
3. Entrena o vuelve a entrenar los modelos con los pares añadidos:
   ```bash
   freqtrade train --config config.json --freqaimodels
   ```
4. Si no quieres usar FreqAI para ciertos pares, sácalos del `pair_whitelist` o desactiva FreqAI para esa sesión (`freqai.enable: false`) para evitar los avisos.

## 2. `ValueError("Must specify a fill 'value' or 'method'.")`
Este error proviene de `pandas.DataFrame.fillna()` cuando se llama sin indicar cómo rellenar los valores faltantes. Puede dispararse si algún indicador devuelve `NaN`/`None` y luego se ejecuta `fillna(inplace=True)` sin `value` ni `method`.

**Pasos de comprobación y corrección**
1. Revisa `populate_indicators` de tu estrategia y localiza llamadas a `fillna()` o `ffill()` sin argumentos. Asegúrate de usarlas con parámetros, por ejemplo:
   ```python
   dataframe.fillna(method="ffill", inplace=True)
   dataframe.fillna(0, inplace=True)  # como última capa de seguridad
   ```
2. Identifica indicadores que generen `NaN` (p. ej., bandas de Bollinger al inicio) y rellena sus columnas justo después de crearlas (`indicator.ffill()` / `bfill()` / `fillna(0)`).
3. Si combinas features de FreqAI con tu dataframe principal, valida que las columnas añadidas no contengan `NaN` antes de llamar a `populate_buy_trend`/`populate_sell_trend`.
4. Vuelve a ejecutar la estrategia en seco (`freqtrade backtesting` o `--dry-run`) para comprobar que los avisos desaparecen tras aplicar los rellenos adecuados.

## Resumen rápido
- Añade cada par a la configuración y entrena modelos FreqAI antes de usarlo en vivo.
- Usa `fillna` siempre con `value` o `method` (`ffill`, `bfill`, `0`, etc.).
- Tras cada cambio, revalida con un backtest o `--dry-run` para detectar nuevas columnas con valores faltantes.
