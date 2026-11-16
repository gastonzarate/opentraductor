# Requisitos confirmados

OpenTraductor traduce voz en español argentino a voz en inglés estadounidense.
La prioridad es mantener una conversación fluida. La aplicación y los modelos
de inferencia deben funcionar completamente en esta Mac, sin enviar audio ni
transcripciones a servicios externos. Las descargas iniciales de dependencias,
modelos y corpus públicos están autorizadas.

## Dominio de uso

Confirmado por el usuario: el dominio de conversación es SOFTWARE Y TECNOLOGÍA
(reuniones de equipos de software: PR, merge, deploy, rollback, API, incidentes,
versiones, latencias y métricas). El vocabulario de prueba del benchmark sigue
ese dominio.

## Orden del trabajo

1. Construir y validar el benchmark.
2. Medir una primera solución local con audios humanos reales.
3. Comparar alternativas y optimizar usando el conjunto de desarrollo.
4. Integrar captura, reproducción, configuración e historial.
5. Ejecutar la evaluación final separada y documentar los objetivos incumplidos.

La infraestructura puede probarse con datos sintéticos claramente identificados.
Esas pruebas no cuentan como resultados de traducción de voz real.

## Latencia

El objetivo del usuario es menos de 300 ms entre el final acústico de una frase
o fragmento y el primer audio traducido. La espera del detector de fin de voz
forma parte de la latencia. El tiempo desde que el sintetizador recibe texto
es una medida por componente, no el objetivo de extremo a extremo.

Se publicarán p50, p95 y p99, número de ejemplos y hablantes independientes,
fallos y condiciones de ejecución. Para guiar la optimización se usará p95
como objetivo principal, sin ocultar los demás percentiles. Arranque en frío y
ejecución con modelos ya cargados se informarán por separado.

La latencia de PCM generado y la salida al dispositivo son medidas distintas.
No se afirmará una latencia acústica física sin medirla. Los límites de voz
estimados tendrán método e incertidumbre explícitos. También se evaluarán el
retraso acumulado, las omisiones y las revisiones durante habla continua;
emitir un sonido temprano no equivale a una traducción válida.

## Audio y configuración

- Selección de micrófono de entrada de macOS.
- Salida hacia un dispositivo virtual seleccionable como micrófono en Google
  Meet y Discord.
- Escucha local opcional para depuración, con dispositivo y volumen propios.
- Controles para iniciar y detener la traducción y ver el estado de los modelos.
- La captura empieza por una acción explícita en la aplicación.
- Los desbordamientos y cancelaciones deben ser visibles; no se ocultarán
  omisiones de contenido para aparentar menor latencia.

## Voz y calidad

Primero se usará una voz de ejemplo con licencia apropiada. El usuario podrá
grabar su propia muestra más adelante. La clonación y la conservación del tono
y la expresividad se evaluarán como alternativas: la velocidad tiene prioridad.
Una voz fija de buena calidad es una primera referencia válida, pero no debe
presentarse como clonación.

La evaluación distinguirá fidelidad semántica, inteligibilidad, naturalidad,
acento, expresividad y similitud del hablante. Las métricas automáticas se
etiquetarán como aproximaciones. Las valoraciones humanas requieren una prueba
de escucha a ciegas con evaluadores reales; no se inventarán resultados MOS.

## Datos locales y licencias

La aplicación guarda textos originales y traducciones en una base SQLite local
pequeña, con los metadatos mínimos de sesión/configuración. No guarda audio de
conversaciones. Los corpus públicos y los audios generados durante el benchmark
se almacenan por separado y conservan su procedencia y licencia.

El proyecto será abierto. Se documentarán las licencias de código, pesos,
datasets y voces por separado, evitando asumir que una licencia de código cubre
los pesos o que una descarga pública permite cualquier reutilización.

## Reproducibilidad del benchmark

- Versiones, revisiones de pesos, hashes de archivos, configuración y hardware.
- Audio entregado a ritmo real, sin exponer muestras futuras ni referencias al
  adaptador. Ensayos de texto y archivos completos separados del streaming.
- Conjuntos de desarrollo y evaluación separados por hablante y grabación
  cuando la procedencia lo permita; las limitaciones se declararán.
- Orden de candidatos y repeticiones controlado, calentamiento separado y
  una sola carga de rendimiento activa en la máquina.
- Registro explícito de errores, timeouts, datos no disponibles y exclusiones.
- Calidad del audio producido evaluada aparte del texto que se quería decir.
- Cobertura de español argentino declarada por evidencia, nunca inferida solo
  de la etiqueta genérica «español».
