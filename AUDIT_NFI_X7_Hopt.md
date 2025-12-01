# Auditoría de optimización para `NFI_X7_Hopt`

## Resumen ejecutivo
La estrategia `NFI_X7_Hopt` es un wrapper long-only para `NostalgiaForInfinityX7` con compatibilidad opcional para FreqAI. La lógica principal se centra en detectar retrocesos relativos a una EMA y filtrar con RSI; adicionalmente incorpora un hook para señales `ai_long_ok` cuando están presentes. Se identifican oportunidades de optimización ligadas a robustez operativa, eficiencia de cálculo y configurabilidad.

## Hallazgos y recomendaciones

### 1. Robustez de parámetros y validaciones de configuración
- **Observación:** Los overrides desde `strategy_parameters` usan asignaciones directas sin validar tipos u orden de llamada, pudiendo dejar valores fuera de rango o inconsistentes al reiniciarse el bot. 【F:AUDIT_NFI_X7_Hopt.md†L11-L13】
- **Recomendación:** Centralizar validaciones en un helper que reporte con claridad si un parámetro quedó recortado al rango permitido y registrar el valor efectivo tras el clamping. Esto mejora la trazabilidad durante hyperopt y reduce riesgos de configurar umbrales inválidos.

### 2. Evitar recálculo redundante de indicadores
- **Observación:** `populate_indicators` vuelve a calcular EMA, RSI y distancia a EMA si las columnas no existen, pero delega primero en `super()`, que ya podría calcularlos. No hay memoización ni caché de indicadores compartidos entre timeframe principal y secundarios. 【F:AUDIT_NFI_X7_Hopt.md†L15-L18】
- **Recomendación:** Explicitar qué indicadores calcula la base para evitar dobles cómputos; si se requieren en ambos frames (principal e informativo), usar `informative()` para compartir resultados o almacenar en `self.custom_info`. Esto reduce overhead en backtests largos.

### 3. Refinar la lógica de retrocesos y tags de entrada
- **Observación:** Los umbrales `deep_pullback` y `shallow_pullback` comparten la misma base `buy_ema_dist_min` y un único `buy_rsi_max`; la versión “shallow” se limita a multiplicar por 0.5/–5. Esto puede generar señales superpuestas y escasa diversidad de setups. 【F:AUDIT_NFI_X7_Hopt.md†L20-L23】
- **Recomendación:** Exponer parámetros independientes para retrocesos profundos y superficiales (distancia y RSI), y añadir un filtro de tendencia superior (p.ej. EMA200>EMA50 o ADX>20) para reducir entradas en rangos laterales. Permitir ajustar la ventana `rsi_len` y `ema_len` vía parámetros optimizables aumenta la capacidad de adaptar la estrategia al par/timeframe.

### 4. Gestión de rebuys y control de riesgo
- **Observación:** La etiqueta `long_rebuy` se dispara con RSI<30 o distancia>1.5×; no hay límite de rebuys por par ni relación con el drawdown de la operación. 【F:AUDIT_NFI_X7_Hopt.md†L25-L26】
- **Recomendación:** Incorporar contadores de rebuys por trade (via `custom_entry_price`/`custom_entry_position` o metadatos) y un umbral máximo por operación. Añadir un guardarraíl de drawdown máximo antes de permitir rebuys evitaría promediar en tendencias bajistas fuertes.

### 5. Integración con FreqAI y calidad de señales IA
- **Observación:** El hook IA filtra sólo entradas `long_normal`; si falta la columna de predicción, se acepta todo. No se registra la ausencia de `pred_col` salvo en debug. 【F:AUDIT_NFI_X7_Hopt.md†L28-L30】
- **Recomendación:** Elevar el nivel de logging a `info` cuando la columna de predicción no se encuentre, e incluir el par afectado. Considerar una opción de configuración (`fail_open` vs `fail_closed`) para decidir si se deshabilita la entrada cuando no haya predicción válida.

### 6. Normalización y preparación de features para FreqAI
- **Observación:** Las features básicas incluyen RSI, distancia a EMA, volumen y cierre sin escalado ni manejo de outliers; no se incluyen derivadas de volumen/volatilidad ni relaciones multi-timeframe. 【F:AUDIT_NFI_X7_Hopt.md†L32-L34】
- **Recomendación:** Añadir features robustas y escaladas (z-score o min-max) y métricas de volatilidad (ATR, HV) y tendencia (slope de EMA). Incluir features multi-timeframe con prefijos `%` siguiendo la API nueva, cuidando evitar look-ahead. Esto suele mejorar la estabilidad de los modelos FreqAI.

### 7. Estandarización de columnas `ai_*`
- **Observación:** `ai_long_ok` se inicializa con `int8`, `ai_short_*` con 0/0.0; no se valida que `close` exista al iniciar `cond_ok`. 【F:AUDIT_NFI_X7_Hopt.md†L36-L38】
- **Recomendación:** Definir explícitamente los `dtypes` deseados y comprobar la presencia de `close` antes de crear máscaras. Esto evita errores silenciosos en timeframes informativos o datasets incompletos.

### 8. Telemetría y trazabilidad
- **Observación:** Actualmente sólo se registran avisos de errores FreqAI. No hay métricas de cuántas señales fueron bloqueadas por IA, por retroceso profundo o por falta de predicción. 【F:AUDIT_NFI_X7_Hopt.md†L40-L41】
- **Recomendación:** Añadir contadores (p.ej. en `custom_info` o logs) de señales generadas/filtradas por categoría. Esto facilita comparar el impacto real de la IA vs. heurísticas tradicionales y orientar futuras optimizaciones.

## Próximos pasos sugeridos
1. Añadir parámetros separados para retrocesos profundos/superficiales y long/rehuy, con validación clara en `bot_start`.
2. Incorporar filtros de tendencia y volatilidad para reducir entradas en rangos y evitar rebuys agresivos en caídas prolongadas.
3. Mejorar la instrumentación de FreqAI: logging de predicciones ausentes, opción de `fail_closed`, y un set ampliado de features normalizadas.
4. Implementar telemetría ligera para medir impacto de cada filtro y alimentar futuras iteraciones de hyperopt.
