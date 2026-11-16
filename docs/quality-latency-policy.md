# Calidad y latencia: condición conjunta para cambios del traductor

Requisito del usuario, 2026-11-15: comparar un audio largo de 2–3 minutos traducido con contexto completo y con el sistema incremental, y medir este benchmark con cada cambio junto con el objetivo de 300 ms.

## Evidencia obligatoria

- Un tramo continuo de 120–180 segundos en español, congelado con origen, licencia, hash, límites de corte y procedencia de la transcripción. Una concatenación de clips no sustituye una conversación continua y debe etiquetarse como prueba sintética si se usa adicionalmente.
- La misma entrada para una traducción del documento completo y para el traductor incremental con reproducción causal. La traducción completa es un control con más contexto, no una referencia infalible.
- Un control adicional que traduzca con contexto completo el texto reconocido por el sistema permite separar errores de reconocimiento de errores de segmentación/traducción. No truncar silenciosamente el documento para acomodar los límites del adaptador incremental.
- Comparación de contenido contra el español: cobertura, omisiones, adiciones, repeticiones, negaciones, nombres, cifras y consistencia contextual. Las coincidencias de texto entre traducciones son diagnósticos, no una prueba suficiente de calidad. Referencias y revisiones automáticas/humanas deben identificarse como tales.
- Latencia por fragmento con correspondencia fuente→salida y origen temporal explícitos; primer PCM, primer sonido simulado y salida física separados. Registrar distribución, fallos, fragmentos sin correspondencia, incertidumbre, cola máxima y retraso acumulado al terminar. No sumar percentiles ni usar el primer audio de toda la sesión como latencia de cada fragmento.

## Cada cambio

Cada candidato que cambie código del traductor, modelos, prompts, segmentación, configuración de inferencia, voz, colas o reproducción debe ejecutar el benchmark de calidad larga y las mediciones de latencia sobre ese candidato antes de promoverse como mejora validada. Se conservará una comparación con la versión base bajo las mismas condiciones. Las ediciones durante el desarrollo pueden agruparse en un candidato; no se reutilizan resultados antiguos para aprobar una versión cuyo comportamiento cambió.

Los resultados deben identificar los hashes del candidato, modelos, prompts, políticas, corpus, evaluador y configuración de prueba. Ausencia de resultados, resultados de otra versión, errores o alineación insuficiente no equivalen a aprobación. Cambios puramente documentales no necesitan inferencia, pero deben declarar que no alteran el comportamiento evaluado.

La calidad y la latencia se deciden juntas: una reducción de tiempo con pérdida de contenido, cambio de sentido o repetición nueva no se acepta como mejora. Los avisos automáticos se revisan contra el contenido, sin convertirlos por defecto en errores reales ni en aprobaciones. No se inventan umbrales de calidad después de ver qué candidato gana.

El objetivo de 300 ms sigue pendiente. Una latencia medida desde que se reserva un fragmento excluye el reconocimiento y la espera de estabilización; no certifica el objetivo desde el fin del fragmento hablado. Sin correspondencia y precisión temporal suficientes, se informa «no verificable», aunque la inferencia aislada tarde menos de 300 ms. Una única grabación repetida es una prueba de regresión limitada, no una validación general de conversaciones.

## Estado de implementación

Política vigente; el arnés largo y su comprobación automática de resultados por versión están en desarrollo. Aún no existe una corrida larga aprobada. Hasta implementarlos, no promover cambios de comportamiento como mejoras validadas basándose solo en latencia o clips cortos.

Las mediciones actuales por componente y sus limitaciones están en [el informe provisional](../artifacts/analysis/coordinator-resume-20261115/component-runtime-1-provisional.md). El contrato incremental y las distinciones de alineación están en [incremental-benchmark.md](incremental-benchmark.md).
