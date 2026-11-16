# Candidato de prompt: inglés natural sin completar contenido

Solicitud del usuario, 2026-11-15: mejorar la naturalidad de la traducción de Qwen. Es un candidato para comparar; no cambia el prompt activo por sí mismo.

## Candidato seleccionado para validación larga

System:

```text
You translate Argentine Spanish speech into natural US English, including software discussions. Translate the intended meaning using idiomatic English grammar, not a word-for-word rendering. Preserve technical meaning, register, uncertainty, negations, numbers and names. Do not invent missing content or complete unfinished sentences. Output only the translation.
```

Entrada completa:

```text
Spanish:
{src}
```

Entrada parcial:

```text
Spanish (incomplete fragment, the speaker is still talking; translate only what is present and do not guess the rest):
{src}
```

El registro se refiere a la formalidad e intención presentes en el texto. Este cambio no incorpora prosodia ni emociones acústicas al sintetizador ni autoriza borrar dudas o correcciones del hablante. La instrucción parcial se conserva idéntica a la versión base.

## Comparación y decisión

- Mantener pesos, cuantización, temperatura, caché, límites, segmentación y reproducción iguales entre base y candidato. Registrar texto y hash de los tres prompts realmente usados; comprobar el formato con el chat template.
- Mantener el prefijo inglés ya confirmado exactamente en IDs de token. Un prompt no puede corregir palabras que ya se reprodujeron. No permitir reescritura, repetición ni contenido futuro para maquillar la naturalidad.
- Separar los tres clips públicos ya examinados de nuevos casos de regresión: modismos con contexto técnico y literal, negaciones, dudas, correcciones, cifras, nombres y formalidad. Los ejemplos de evaluación no se agregan al prompt.
- Evaluar fragmentos parciales causalmente y traducción completa, guardando fallos y salidas vacías. Un resultado vacío no demuestra una traducción correcta ni una mejora de latencia.
- Revisar significado contra el español, además de fluidez. Las coincidencias con una traducción esperada no bastan; conservar variantes válidas y diferencias de contexto.
- Medir tokens de entrada, tiempo hasta el primer token, traducción completa, primer PCM, cola y sonido simulado por separado. No convertir tokens adicionales en milisegundos estimados sin medición.
- Ejecutar la comparación larga y las mediciones de latencia requeridas en [la política de calidad y latencia](quality-latency-policy.md) antes de promover el candidato. Aún no hay resultados largos que permitan activarlo como mejora validada.

## Evidencia inicial

En el control corto previo, el mismo Qwen con el mismo ASR completo corrigió «Give it to the place…» a «Give the place…». Eso apunta al contexto parcial y al compromiso temprano del prefijo; no prueba que cambiar el prompt resuelva por sí solo la segmentación. Véase [el informe](../artifacts/analysis/quality-short-comparison-20261115/report.md).

## Resultado de exploración local

Se ensayaron tres variantes, 176 llamadas medidas en total (48 + 64 + 64), además de calentamientos. Cada corrida usa orden base/candidato/candidato/base, con dos repeticiones por perfil. Son casos de desarrollo adaptativos, no un conjunto independiente de aceptación. No se modificó la app.

- Variante 1 descartada: inventó «The server's still talking» ante «El servidor se», en ambas repeticiones. Copió contenido de la instrucción parcial.
- Variante 2, arriba: mejora algunas construcciones, por ejemplo «capaz que se recupera solo» pasa de «maybe it's recovering on its own» a «it might recover on its own», y la petición formal usa «when you have a moment». Mantiene los casos examinados de negación, cifras y corrección. Sigue fallando «en un toque»: «in a nutshell», mientras que la base produce «in one go»; ambas pierden el sentido temporal.
- Variante 3: casi idéntica a la base en los ejemplos completos, sin resolver «en un toque». En un fragmento ambiguo emitió «The server did», comprometiendo un tiempo verbal todavía no expresado.

La variante 2 queda solo como candidata para una comparación larga: no se considera mejora general validada. Mediana de tiempo hasta primer token: base 159,1 ms / candidata 160,6 ms; traducción completa: base 299,7 ms / candidata 293,8 ms (32 llamadas por perfil). No es latencia de audio ni demuestra una mejora estadística o el objetivo de 300 ms.

Artefactos: [perfil candidato exacto](../artifacts/analysis/natural-prompt-root-diagnostic-20261115-v2/candidate-profile.json), [resultados base/candidata](../artifacts/analysis/natural-prompt-root-diagnostic-20261115-v2/result.json), [comparación de las tres corridas](../artifacts/analysis/natural-prompt-root-diagnostic-20261115/comparison.json). Los resultados incluyen prompts, configuración, errores y hashes del código diagnóstico y del modelo.

## Aclaración del usuario: anticipación con confianza alta

El usuario propone completar anticipadamente solo cuando la confianza sea suficientemente alta. Esto añade una política experimental de anticipación, distinta de mejorar la redacción del prompt. El candidato de arriba todavía traduce únicamente contenido recibido; no implementa ese umbral.

Estado comprobado: `MLXQwenLM.generate` recibe `tok, _lp` pero descarta `_lp`; hoy no hay un estimador calibrado que autorice completar contenido futuro. Un valor autodeclarado por el modelo no se tratará como probabilidad verificada de acertar lo que dirá la persona. Las probabilidades de tokens, si se instrumentan, serán señales candidatas que deberán contrastarse con continuaciones reales, no el umbral final por sí solas.

Diseño a evaluar:

1. Separar preparación anticipada (texto o audio cancelable) de emisión audible irreversible. Registrar ambas, como exige el contrato incremental existente.
2. Estimar por separado la estabilidad de la traducción de lo ya oído y el acierto de una continuación aún no oída. Evaluar señales como estabilidad entre parciales, alternativas de traducción y evidencia lingüística; no asumir que alguna garantiza exactitud.
3. Calibrar la decisión con prefijos y continuaciones reales, incluyendo el mismo inicio con finales distintos, pausas largas, autocorrecciones y negaciones tardías. Separar datos de ajuste de datos de evaluación. Sin evidencia suficiente, retener la continuación y seguir traduciendo el contenido confirmado.
4. Como primera política conservadora, reservar cifras, nombres y negaciones hasta recibir soporte. No completar una acción como «reiniciar» solo porque sea una continuación plausible de «vamos a…».
5. Reportar aciertos y errores semánticos de las anticipaciones, cobertura (qué fracción se anticipa), error condicionado al umbral, correcciones audibles, trabajo descartado, cola y latencia. Una anticipación equivocada no cuenta como primer audio correcto ni ayuda a cumplir los 300 ms.
6. Elegir el umbral a partir de esa relación entre errores y latencia, sin fijar un porcentaje arbitrario. Medir el mismo audio largo base/candidato y los casos adversariales antes de activar anticipación audible.

Esta aclaración no activa predicción ni cambia el comportamiento actual. La preparación experimental y su validación forman parte del trabajo solicitado; no requieren volver a pedir autorización.

## Comparación ampliada posterior

El adaptador por instancia `natural-prompt-v3` pasó 51 pruebas CPU del coordinador, incluido arranque aislado y conservación de resultados interrumpidos. La corrida real `runs/coordinator-1` completó 812 llamadas sobre 26 casos sintéticos y 3 regresiones públicas, con dos pasadas por perfil y cuatro tipos de prueba. No hubo errores de ejecución ni diferencias entre pasadas, pero sí errores semánticos.

El candidato corrigió el puerto 8080 (la base produjo 880) y algunas construcciones. Ambos fallaron «cuatro menos diez» (4:10 en vez de3:50) y modismos como «estamos al horno con la entrega del viernes». En frases completas sintéticas, MTp50 base298,3ms/candidato306,5ms y p95 381,2/398,8ms, n52 por perfil; no es latencia de audio. La cadena simplificada que compromete toda la traducción parcial produjo además errores de gramática y omisiones; no representa la política exacta del motor activo.

Decisión posterior: **no promover el candidato como mejora validada**. Las mejoras puntuales no bastan; sigue pendiente el benchmark largo. [Informe ampliado del coordinador](../artifacts/analysis/coordinator-resume-20261115/natural-prompt-1-review.md).

`natural-prompt-v4` solo corrige documentación y una nota de estado: se verificó que código, corpus y perfiles son idénticos a v3. Los resultados se atribuyen a v3, que fue la versión realmente ejecutada.
