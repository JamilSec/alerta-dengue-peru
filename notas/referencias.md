# Referencias verificadas — Sistema inteligente de alerta temprana de brotes de dengue por distrito en el Perú

Fecha de verificación: 2026-11-09.
Método: cada fuente se abrió en la web (página del editor, PDF descargado, API de Crossref o Europe PMC, ficha de repositorio). Lo que no pudo comprobarse se marca `[NO VERIFICADO]`. Todo el contenido está parafraseado; las cifras se transcriben tal como figuran en la fuente.

Leyenda de verificación: **[V]** = metadatos y contenido comprobados en la fuente; **[V-meta]** = solo metadatos bibliográficos comprobados (Crossref/repositorio), contenido tomado del resumen o de fuentes secundarias.

---

## 1. Antecedentes (2021 o posteriores)

### 1.1 Internacionales

#### A1. Sebastianelli et al. (2024) — Brasil (con transferencia a Perú) **[V]**
- **Título:** A reproducible ensemble machine learning approach to forecast dengue outbreaks.
- **Objetivo general:** pronosticar con un mes de anticipación la tasa de incidencia de dengue (DIR) en los estados de Brasil, con énfasis en población menor de 19 años, y evaluar si el modelo se transfiere a departamentos de Perú.
- **Metodología:** ensamble de tres modelos (CatBoost, SVR y LSTM) cuyas salidas se combinan mediante un Random Forest que también entrega intervalos de predicción al 95 %; reducción de dimensionalidad con PLS; aumento de datos con ventanas móviles de 12 meses y ruido gaussiano; covariables climáticas (ERA5-Land), NDVI, elevación, pérdida forestal y 31 variables socioeconómicas.
- **Población:** 27 unidades federativas de Brasil (casos mensuales SINAN, enero 2001–diciembre 2019) y departamentos de Perú con datos 2010–2019 (Loreto, Madre de Dios, Piura, San Martín, Tumbes y Ucayali).
- **Muestra / partición:** entrenamiento 2001–2016 y validación 2017–2019 (Brasil); transferencia con ajuste fino para Perú.
- **Resultados (tal como figuran):** la única métrica reportada es el nRMSE. Para Brasil (población total) el mejor valor fue Distrito Federal 0,041 y el peor Acre 0,274. Para Perú (total / 0–19 años): Piura 0,117 / 0,097; Tumbes 0,156 / 0,162; Loreto 0,190 / 0,177; San Martín 0,160 / 0,138; Madre de Dios 0,157 / 0,162; Ucayali 0,294 / 0,327. No reporta AUC, F1 ni RMSE absoluto.
- **APA 7:** Sebastianelli, A., Spiller, D., Carmo, R., Wheeler, J., Nowakowski, A., Jacobson, L. V., Kim, D., Barlevi, H., Cordero, Z. E. R., Colón-González, F. J., Lowe, R., Ullo, S. L., & Schneider, R. (2024). A reproducible ensemble machine learning approach to forecast dengue outbreaks. *Scientific Reports, 14*, 3807. https://doi.org/10.1038/s41598-024-52796-9

#### A2. Ong et al. (2023) — Malasia **[V]**
- **Título:** Predicting dengue transmission rates by comparing different machine learning models with vector indices and meteorological data.
- **Objetivo general:** desarrollar modelos de clasificación de "brote / no brote" de dengue a partir de índices entomológicos y variables meteorológicas, comparando varios algoritmos de ML como base de un sistema de alerta temprana.
- **Metodología:** siete algoritmos (regresión logística, Naïve Bayes, árbol de decisión, SVM, Random Forest, XGBoost y AdaBoost); selección de variables con Boruta; búsqueda de hiperparámetros por grid search; comparación con ROC-AUC, exactitud y F1. La variable objetivo es binaria (ocurre o no un brote, definido como transmisión por encima de lo esperado según la definición operativa de la OMS para Malasia).
- **Población:** cinco distritos del Territorio Federal de Kuala Lumpur/Putrajaya (Titiwangsa, Kepong, Cheras, Lembah Pantai y Putrajaya), con índices vectoriales (HI, BI, CI, PI) y datos meteorológicos semanales de enero 2018 a diciembre 2020.
- **Muestra / partición:** 860 observaciones; 70 % entrenamiento (n = 602) y 30 % validación (n = 258).
- **Resultados (tal como figuran):** los métodos de ensamble (XGBoost, AdaBoost, Random Forest) superaron a regresión logística, Naïve Bayes, árbol de decisión y SVM; XGBoost obtuvo el AUC, la exactitud y el F1 más altos. El índice de recipientes (CI) fue la variable menos importante y al retirarla los modelos mejoraron al menos 6 % en AUC y F1 (mejora promedio de AUC de 6,71 % en el texto). Los valores numéricos exactos de AUC/F1 por modelo están en la Tabla 3 del artículo, que no pude leer en texto `[NO VERIFICADO: cifras exactas por modelo]`.
- **APA 7:** Ong, S. Q., Isawasan, P., Ngesom, A. M. M., Shahar, H., Lasim, A. M., & Nair, G. (2023). Predicting dengue transmission rates by comparing different machine learning models with vector indices and meteorological data. *Scientific Reports, 13*, 19129. https://doi.org/10.1038/s41598-023-46342-2

#### A3. Colón-González et al. (2021) — Vietnam **[V]**
- **Título:** Probabilistic seasonal dengue forecasting in Vietnam: A modelling study using superensembles.
- **Objetivo general:** construir un sistema operativo de pronóstico estacional de dengue que combine observaciones de la Tierra, pronósticos climáticos estacionales y casos rezagados para anticipar el riesgo hasta seis meses.
- **Metodología:** modelos bayesianos espacio-temporales ajustados a 19 años de datos provinciales; superensamble probabilístico de esos modelos; evaluación con CRPS y capacidad de detección de brotes frente a un modelo base que pronostica la misma incidencia cada mes.
- **Población:** provincias de Vietnam, casos mensuales 2002–2020.
- **Muestra:** 19 años de datos a nivel provincial (todas las provincias del país).
- **Resultados (tal como figuran):** CRPS del superensamble 66,8 (IC 95 % 60,6–148,0) frente a 79,4 (78,5–80,5) del modelo base, a 1–3 meses de anticipación; detección de brotes 69 % frente a 54,5 % del modelo base; en la predicción prospectiva mayo–octubre 2020, CRPS 110 (102–575) frente a 125 (120–168).
- **APA 7:** Colón-González, F. J., Soares Bastos, L., Hofmann, B., Hopkin, A., Harpham, Q., Crocker, T., Amato, R., Ferrario, I., Moschini, F., James, S., Malde, S., Ainscoe, E., Sinh Nam, V., Quang Tan, D., Duc Khoa, N., Harrison, M., Tsarouchi, G., Lumbroso, D., Brady, O. J., & Lowe, R. (2021). Probabilistic seasonal dengue forecasting in Vietnam: A modelling study using superensembles. *PLOS Medicine, 18*(3), e1003542. https://doi.org/10.1371/journal.pmed.1003542

#### A4 (complementario). Fang, Hu y Pan (2024) — revisión sistemática **[V]**
- **Título:** Meteorological factors cannot be ignored in machine learning-based methods for predicting dengue, a systematic review.
- **Objetivo general:** sintetizar los métodos de ML y los predictores usados para predecir la incidencia de dengue.
- **Metodología:** revisión sistemática (PRISMA) en PubMed, ScienceDirect y Web of Science hasta julio de 2023.
- **Población / muestra:** 23 artículos incluidos.
- **Resultados (tal como figuran):** predictores meteorológicos en 22 estudios (95,7 %), datos históricos de dengue en 14 (60,9 %); temperatura y lluvia en 20 (87,0 %) cada una; mejores modelos: SVM 6 (26,1 %), LSTM 5 (21,7 %), Random Forest 4 (17,4 %), LASSO 2 (8,7 %), ensambles 2 (8,7 %).
- **APA 7:** Fang, L., Hu, W., & Pan, G. (2024). Meteorological factors cannot be ignored in machine learning-based methods for predicting dengue, a systematic review. *International Journal of Biometeorology, 68*(3), 401–410. https://doi.org/10.1007/s00484-023-02605-1

### 1.2 Nacionales (Perú)

#### N1. Sánchez López, Candioti Nolberto, Taquía Gutiérrez y García López (2023) — Loreto **[V]**
- **Título:** Traditional Machine Learning based on Atmospheric Conditions for Prediction of Dengue Presence.
- **Objetivo general:** asociar los casos de dengue con condiciones climáticas mediante modelos de ML y determinar el algoritmo óptimo para esa asociación.
- **Metodología:** clasificación supervisada con SVM, árbol de decisión, Random Forest y AdaBoost; variable objetivo categórica "Class" (cuatro rangos de casos A–D); evaluación con AUC, exactitud (CA), precisión, recall y F1; partición 80/20 y validación cruzada de 5 y 10 pliegues.
- **Población:** casos de dengue por semana epidemiológica de los distritos de Iquitos, Belén, Punchana y San Juan Bautista (SE 1-2010 a SE 18-2022) y datos meteorológicos de SENAMHI de las estaciones San Roque, Amazonas, Puerto Almenara y Moralillo (SE 1-2016 a SE 4-2022).
- **Muestra:** más de 1900 registros antes de limpieza; tras la limpieza 243 (San Roque), 213 (Amazonas), 257 (Puerto Almenara) y 246 (Moralillo) registros.
- **Resultados (tal como figuran):** AUC óptimos entre 0,818 y 0,996 para SVM, Random Forest y AdaBoost; el árbol de decisión fue el peor en AUC (por ejemplo 0,417 en Amazonas). San Roque: SVM AUC 0,952 / CA 0,975 / F1 0,965; RF AUC 0,987; AB AUC 0,967. Puerto Almenara: SVM AUC 0,994, RF 0,996, AB 0,970. Con 10 pliegues en Moralillo: SVM AUC 0,994 / CA 0,955. Los autores proponen SVM como modelo óptimo.
- **APA 7:** Sánchez López, B. S., Candioti Nolberto, D., Taquía Gutiérrez, J. A., & García López, Y. (2023). Traditional machine learning based on atmospheric conditions for prediction of dengue presence. *Computación y Sistemas, 27*(3), 769–777. https://doi.org/10.13053/CyS-27-3-4383

#### N2. Rojas-Paucar, Carranza López, Francia-Candiotti, Pareja-Granda y Medina Soto (2025) — Perú por departamentos **[V]**
- **Título:** Aplicación del modelo Prophet para la predicción de brotes explosivos de dengue en regiones críticas de Perú: un estudio de series temporales 2000–2024.
- **Objetivo general:** predecir el avance de casos de dengue para 2025–2027 en los departamentos clasificados como "brote explosivo" usando el algoritmo Prophet.
- **Metodología:** series temporales univariadas con Prophet; entrenamiento 2000–2023 y prueba con los casos reales de 2024; métricas RMSE, MAE y MAPE.
- **Población:** casos de dengue del MINSA 2000–2024 por departamento.
- **Muestra:** grupo "brote explosivo": Áncash, Callao, Ica, La Libertad, Lambayeque, Lima, Madre de Dios, Piura, Tumbes y Ucayali.
- **Resultados (tal como figuran):** buen ajuste en Lambayeque (RMSE 860,13), Tumbes (1 655,11) y Callao (2 203,28); subestimación severa en La Libertad (real 2024 = 23 115 casos, predicho 5 652,87, RMSE = MAE = 17 462,13), Lima (RMSE 16 119,89), Ica (9 545,99) y Áncash (8 125,42); sobreestimación en Piura (RMSE 4 589,00; predicción 20 104) y Ucayali (4 651,68). Conclusión: el modelo univariado falla ante brotes repentinos.
- **APA 7:** Rojas-Paucar, W., Carranza López, A. O., Francia-Candiotti, E. J., Pareja-Granda, E. A., & Medina Soto, E. L. (2025). Aplicación del modelo Prophet para la predicción de brotes explosivos de dengue en regiones críticas de Perú: un estudio de series temporales 2000–2024. *Revista Cubana de Investigaciones Biomédicas, 44*(Supl.), e3991. https://revibiomedica.sld.cu/index.php/ibi/article/view/3991 `[NO VERIFICADO: la revista no muestra DOI en la ficha ni en el PDF]`

#### N3 (complementario). Padilla Pierola (2024) — tesis UNSM, San Martín **[V]**
- **Título:** Modelo de aprendizaje supervisado basado en el algoritmo XGBoost para predicción de la incidencia del dengue.
- **Objetivo general:** predecir la incidencia de dengue con un modelo supervisado de regresión no lineal basado en XGBoost, con miras a replicarlo en la región San Martín.
- **Metodología:** XGBoost (regresión) con datos históricos autorregresivos y datos de Google Trends; evaluación con RMSE, R-RMSE, R² y correlación de Pearson.
- **Población / muestra:** 272 periodos semanales de 20 ciudades de Brasil (el modelo no se entrenó con datos peruanos).
- **Resultados:** según el resumen, los modelos con Google Trends y la combinación con datos históricos tuvieron el mejor desempeño `[NO VERIFICADO: valores numéricos de RMSE/R² no extraídos del PDF]`.
- **APA 7:** Padilla Pierola, J. H. (2024). *Modelo de aprendizaje supervisado basado en el algoritmo XGBoost para predicción de la incidencia del dengue* [Tesis de título profesional, Universidad Nacional de San Martín]. Repositorio Institucional UNSM. https://gestion-repo.unsm.edu.pe/items/2fe2d775-d473-4cf8-b695-f5ecdee5025f

### 1.3 Local (Trujillo / La Libertad / norte peruano)

**Aviso claro:** no encontré ningún trabajo real (artículo o tesis) de aprendizaje automático para predicción de brotes de dengue en Trujillo, La Libertad o Cajamarca. Se revisaron los repositorios de la UNT (no accesible por API), UPN (bloqueado), UCV (accesible), UPAO y UNC (sin resultados de ML), además de ALICIA/CONCYTEC y búsquedas web. En la UCV-Trujillo solo hay tesis clínicas de factores predictores de gravedad (sin ML). El antecedente más cercano con ML en el norte del Perú es el siguiente.

#### L1. Mills et al. (2025) — Piura, Tumbes y Lambayeque (costa norte) **[V]**
- **Título:** Multi-model approach to understand and predict past and future dengue epidemic dynamics.
- **Objetivo general:** introducir un enfoque multimodelo que integra técnicas nuevas y existentes para entender y predecir epidemias de dengue pasadas y futuras, usando el norte del Perú como caso de estudio.
- **Metodología:** análisis de wavelets y GAM para sincronía epidémica; modelo bayesiano climático (Poisson inflado en ceros, efectos BYM, DLNM, INLA); pronósticos a un mes con paseo aleatorio (baseline), SARIMA, red convolucional temporal (TCN), TimeGPT (con y sin covariables) y ensambles (equiponderado, mediana y entrenado con 23 cuantiles); inferencia conformal.
- **Población:** 14 provincias de Piura, Tumbes y Lambayeque, casos mensuales del CDC Perú (MINSA) de mayo 2010 a diciembre 2021; clima WorldClim 2.1, SPI-6 e índices ENSO (ICEN, ONI).
- **Muestra / partición:** desarrollo 2010–2017, prueba 2018–2021.
- **Resultados (tal como figuran, Tabla 2, periodo de prueba):** ensamble "Median*": WIS 0,34, cobertura del intervalo 95 % = 95,2 %, R² 0,74, RMSE 0,81 (en log(casos+1)), sensibilidad 81,8 %, especificidad 94,3 %, AUC 0,88; TimeGPT: AUC 0,91, sensibilidad 90,9 %, especificidad 91,7 %; baseline: AUC 0,68. El ensamble detectó el 81,8 % de los primeros meses de brote (DIR ≥ 50/100 000) con 5,7 % de falsas alarmas.
- **APA 7:** Mills, C., Falconi-Agapito, F., Carrera, J.-P., Munayco, C., Kraemer, M. U. G., & Donnelly, C. A. (2025). Multi-model approach to understand and predict past and future dengue epidemic dynamics. *Royal Society Open Science, 12*(11), 241870. https://doi.org/10.1098/rsos.241870

#### L2 (contexto local, sin ML). Bravo Alarcón (2024) — Trujillo **[V-meta]**
- Tesis de Medicina (UCV, sede Trujillo): nivel de leucocitos como factor predictor de dengue grave en un hospital de Trujillo, 2024; diseño observacional transversal; leucocitos < 8 000 células/mm³ con sensibilidad 80,2 % y especificidad 77,3 %. Sirve solo para contextualizar la carga local; no es un antecedente de ML.
- **APA 7:** Bravo Alarcón, C. (2024). *Nivel de leucocitos como factor predictor de dengue grave en pacientes atendidos en un hospital de Trujillo 2024* [Tesis de pregrado, Universidad César Vallejo]. Repositorio UCV. https://hdl.handle.net/20.500.12692/178074

---

## 2. Contexto del problema (cifras con fuente y fecha)

### 2.1 Mundial / Américas
- **OMS (nota descriptiva, 21 de agosto de 2025):** los casos notificados a la OMS pasaron de 505 430 en 2000 a 14,6 millones en 2024, con más de 12 000 muertes relacionadas con dengue en 2024; la enfermedad es endémica en más de 100 países y cerca de la mitad de la población mundial está en riesgo; una estimación de modelización calcula 390 millones de infecciones anuales, de las cuales 96 millones son clínicas. `[V]` (la cifra 505 430 del año 2000 la tomo de la misma nota; el resto fue leído directamente).
- **OPS/OMS (Actualización Epidemiológica, 18 de febrero de 2026):** en 2024 la Región de las Américas registró un incremento histórico con 13 063 434 casos notificados; en 2025 (hasta SE 53) se notificaron 4 459 521 casos sospechosos y 1 682 588 confirmados, 8 966 (0,2 %) graves y 2 207 defunciones (letalidad 0,05 %), un descenso de 66 % frente a 2024. `[V]`

### 2.2 Perú
- **CDC Perú – MINSA, Alerta Epidemiológica AE-CDC N.º 006-2024:** en 2023 se notificaron 273 684 casos de dengue (84,8 % confirmados = 232 099; 15,2 % probables = 41 585), incidencia acumulada nacional de 808,9 por 100 mil habitantes; 88,5 % sin signos de alarma, 11,1 % con signos de alarma y 0,4 % graves; 442 defunciones confirmadas. Hasta la SE 7-2024: 24 981 casos, 28 defunciones (procedentes, entre otros, de La Libertad, Piura y Lambayeque), 476 distritos con casos y 261 distritos en transmisión activa; 11 departamentos superaban la IA nacional, entre ellos Piura, La Libertad y Lambayeque. `[V]`
- **Munayco et al. (2024), MMWR:** hasta la SE 30 de 2023 (29 de julio) Perú acumuló 222 620 casos, aproximadamente 10 veces el promedio de los 5 años previos para el mismo periodo (21 841) y 3,5 veces el brote de 2017 (64 431); 381 defunciones (letalidad 0,17 %); departamentos con más casos: Piura 67 697, Lambayeque 28 235 y La Libertad 20 289; Lima metropolitana 32 009. `[V]`
- **CDC Perú (nota de prensa, 22 de agosto de 2023):** hasta la SE 31 de 2023, 230 209 casos y 398 defunciones (313 confirmadas, 85 en investigación); 19 departamentos en situación de brote; La Libertad era el departamento con más distritos en transmisión activa (14), seguido de Piura (11), Tumbes y Cajamarca (8 cada uno) y Lambayeque (7). `[V]`
- **OPS (18 de febrero de 2026):** Perú notificó 37 041 casos en 2025 (SE 1–53), 34 834 confirmados, 56 defunciones, incidencia 108,9 por 100 000 y letalidad 0,15 %. `[V]`
- **Rufasto Goche et al. (2025), PLOS ONE:** 501 027 casos analizados entre 2000 y 2022 (datos MINSA); Loreto, Ucayali y Piura concentraron más del 60 % de los casos en años epidémicos; 78 % sin signos de alarma, 18 % con signos de alarma y 4 % graves. `[V-meta + resumen]`
- Cifra nacional 2024: tablas de la sala situacional CDC 2026 muestran 2024 con 258 384 casos y 247 defunciones (provisional); solo lo vi en fragmentos de búsqueda, no en el PDF `[NO VERIFICADO]`.

### 2.3 La Libertad / norte
- La Libertad 2023: 20 289 casos hasta la SE 30 (Munayco et al., 2024) `[V]`; 14 distritos en transmisión activa a la SE 31 (CDC Perú, 2023) `[V]`.
- La Libertad 2024: 23 115 casos reales en 2024 según la Tabla 1 de Rojas-Paucar et al. (2025), construida con datos MINSA `[V]`.
- Piura 2023: 67 697 casos y 130 defunciones (letalidad 0,19 %) hasta la SE 30 (Munayco et al., 2024) `[V]`.
- Lambayeque 2023: 28 235 casos y 115 defunciones (letalidad 0,41 %) hasta la SE 30 (Munayco et al., 2024) `[V]`.
- No encontré una cifra oficial de cierre 2023 (SE 52) para La Libertad en un documento del CDC/GERESA accesible; los PDF de sala situacional superan el tamaño que pude descargar `[NO VERIFICADO]`.

---

## 3. Bases teóricas

### 3.1 Canal endémico (corredor endémico)
- **Definición (Umaña et al., 2026, JMIR Public Health and Surveillance):** método de vigilancia que resume estadísticamente la dinámica histórica de una enfermedad; su curva define la tendencia central esperada de casos y su variación, delimitando tres niveles (seguridad, alarma y epidemia). Los parámetros son: la medida de tendencia central que actúa como umbral de alerta, el tamaño de la ventana retrospectiva y el tratamiento de años epidémicos previos y de valores cero. Los autores señalan que la falta de guías claras para fijar estos parámetros compromete la reproducibilidad. Hallazgos: acortar la ventana retrospectiva mejoró las métricas 6,34 % en promedio; desplazar los datos en 0,001 (en vez de sumar 1 para la transformación logarítmica) mejoró el desempeño 23,07 % en promedio. `[V]`
- **Origen operativo (Bortman, 1999):** describe la elaboración de corredores endémicos con hojas de cálculo. `[V-meta]` Metadatos confirmados en Crossref; el PDF de SciELO devolvió 403/404, así que el detalle del método (medias geométricas e IC 95 % según fuentes secundarias) queda `[NO VERIFICADO]` en el texto original.
- **Variabilidad de las definiciones de brote (Brady et al., 2015):** al aplicar múltiples definiciones candidatas de brote a datos de Brasil, las características detectadas (frecuencia, duración, carga) resultan muy heterogéneas y varían en el tiempo y el espacio, lo que cuestiona el uso de una única definición en sistemas de alerta temprana. `[V]`
- **Comparación de umbrales (Ouédraogo et al., 2023):** con datos mensuales 2016–2021 de Burkina Faso compararon tres métodos: media [media + 2 DE], mediana [tercer cuartil] y suma acumulada [C-sum + 1,96 DE]; los métodos basados en la media fueron sensibles a valores extremos y produjeron umbrales altos, mientras que la mediana captó mejor el incremento anormal. `[V]`
- **Uso en alerta temprana (Bowman et al., 2016):** aplicaron el método de Shewhart y el canal endémico a datos de cinco países (2007–2013) para identificar variables de alarma; el aumento de casos probables predijo brotes con sensibilidad/VPP de 93 %/83 % (México) y 97 %/86 % (República Dominicana). `[V]`
- **Experiencia regional (Badurdeen et al., 2013):** en diez países (incluido Perú) solo seis tenían una definición formal de brote distinta de la variación estacional y ninguno contaba con un enfoque sistemático para detectar y responder a las fases tempranas. `[V]`

**¿Es "media de 5 años + 2 DE" la fórmula oficial de MINSA/CDC Perú?**
- La NTS N.º 125-MINSA/2016/CDC-INS (aprobada por R.M. N.º 982-2016/MINSA) exige que en zonas endémicas la sala situacional incluya el canal endémico, pero en su texto no define la fórmula ni el número de años (revisé el PDF completo; no aparece "desviación estándar" asociada al canal, ni "promedio" ni "umbral"). `[V]`
- La Alerta AE-CDC-006-2024 ordena análisis epidemiológico ante "incremento de casos comparado con tendencias en las últimas seis semanas" y no menciona canal endémico. `[V]`
- Una presentación de la DGE (SE 04-2018) usa como criterio de riesgo "canal endémico: zona epidémica en la última SE", sin detallar el cálculo; el PDF descargado está truncado y solo pude leerlo por fragmentos de búsqueda `[NO VERIFICADO]`.
- Munayco et al. (2024) comparan los casos de 2023 con "el promedio de los 5 años previos para el mismo periodo", lo que respalda el uso de una línea base de 5 años en el CDC Perú, pero no la regla de 2 DE. `[V]`
- **Conclusión:** no encontré ningún documento oficial de MINSA/CDC Perú que fije "media + 2 DE" como fórmula del canal endémico. "Media + 2 DE" es una **variante ampliamente usada en la literatura** (Ouédraogo et al., 2023; Umaña et al., 2026) y en guías docentes, no una fórmula oficial comprobada. En el informe debe presentarse como "umbral tipo canal endémico (media de 5 años + 2 DE), variante de la metodología de corredores endémicos", citando a Bortman (1999), Umaña et al. (2026) y Ouédraogo et al. (2023). `[NO VERIFICADO: fórmula oficial]`

### 3.2 Desbalance de clases y PR-AUC
- **Saito y Rehmsmeier (2015):** demuestran que, con clases desbalanceadas, la curva precisión-recall es más informativa que la curva ROC, porque la ROC puede dar una impresión optimista al no reflejar la proporción de falsos positivos frente a los pocos positivos reales. `[V-meta + resumen]`
- **Chawla, Bowyer, Hall y Kegelmeyer (2002):** SMOTE genera ejemplos sintéticos de la clase minoritaria y, combinado con submuestreo de la mayoritaria, mejora el desempeño medido por AUC frente al submuestreo solo o al ajuste de costos. `[V]`

### 3.3 Random Forest
- **Breiman (2001):** un random forest es una combinación de árboles de decisión donde cada árbol depende de un vector aleatorio muestreado de forma independiente e idéntica, y el error de generalización converge a un límite a medida que crece el número de árboles; su desempeño depende de la fuerza de los árboles individuales y de la correlación entre ellos. `[V-meta]`

### 3.4 Gradient boosting
- **Friedman (2001):** formula el boosting como una optimización numérica en el espacio de funciones por descenso de gradiente, ajustando en cada paso un aprendiz débil (típicamente árboles) al gradiente negativo de la función de pérdida. `[V-meta]`
- **Chen y Guestrin (2016):** XGBoost es un sistema escalable de boosting de árboles que incorpora un algoritmo consciente de la dispersión de los datos, un sketch de cuantiles ponderado para el aprendizaje aproximado de árboles y optimizaciones de caché, compresión y fragmentación de datos. `[V]`

### 3.5 Validación temporal
- **Bergmeir y Benítez (2012):** estudian empíricamente el uso de validación cruzada para evaluar predictores de series temporales y proponen procedimientos por bloques que respetan la dependencia temporal frente a la partición aleatoria. `[V-meta]` (resumen no accesible; descripción basada en el título y en el conocimiento general del artículo) `[NO VERIFICADO: conclusión detallada]`
- Los antecedentes A1, A3 y L1 aplican partición temporal (entrenar en años anteriores, probar en años posteriores), que es la práctica a seguir en el proyecto.

---

## 4. Plataforma Nacional de Datos Abiertos y ficha del dataset

### 4.1 Ficha del dataset `[V]`
- **Título:** Vigilancia Epidemiológica de dengue.
- **URL de la ficha:** https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiológica-de-dengue
- **Entidad publicadora:** Ministerio de Salud – MINSA (etiqueta: CDC PERU).
- **Descripción (parafraseada):** notificación de casos de dengue al sistema de vigilancia en salud pública del Perú, gestionado por el Centro Nacional de Epidemiología, Prevención y Control de Enfermedades (CDC Perú); los datos provienen de la Red Nacional de Epidemiología (RENACE), conformada por 10 272 establecimientos de salud del MINSA, EsSalud y otros del sector.
- **Licencia:** Open Data Commons Attribution License (ODC-BY).
- **Nivel de acceso:** Público.
- **Fecha de lanzamiento:** 2023-11-22. **Última modificación:** 2026-10-21.
- **Identificador:** 2c3b2955-6252-4b9d-abd3-caeb61ccda39.
- **Recurso:** `datos_abiertos_vigilancia_dengue_2000_2025_0.csv` (CSV).
- **Administrador de la plataforma:** Secretaría de Gobierno y Transformación Digital, Presidencia del Consejo de Ministros.

### 4.2 Condiciones de uso de la plataforma
- La licencia ODC-BY permite copiar, distribuir, usar, modificar y producir trabajos a partir de los datos, con la obligación de atribuir la fuente (condición de la licencia ODC-BY; el texto de la licencia está en https://opendatacommons.org/licenses/by/).
- Marco normativo: Decreto Supremo N.º 016-2017-PCM, que aprueba la Estrategia Nacional de Datos Abiertos Gubernamentales 2017-2021 y el Modelo de Datos Abiertos Gubernamentales del Perú, crea el Portal Nacional de Datos Abiertos y establece que la publicación de datos abiertos por las entidades tiene carácter oficial y su contenido es responsabilidad de ellas. `[V-meta: contenido tomado de fragmentos de búsqueda del texto legal]`
- No localicé una página de "Términos y condiciones" propia del portal (las rutas /terminos, /terminos-de-uso, /terminos-y-condiciones devuelven 404) `[NO VERIFICADO]`.

---

## 5. Lista de referencias (APA 7, orden alfabético)

**Artículos científicos y trabajos de investigación (22):**

Badurdeen, S., Valladares, D. B., Farrar, J., Gozzer, E., Kroeger, A., Kuswara, N., Ranzinger, S. R., Tinh, H. T., Leite, P., Mahendradhata, Y., Skewes, R., & Verrall, A. (2013). Sharing experiences: Towards an evidence based model of dengue surveillance and outbreak response in Latin America and Asia. *BMC Public Health, 13*, 607. https://doi.org/10.1186/1471-2458-13-607

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences, 191*, 192–213. https://doi.org/10.1016/j.ins.2011.12.028

Bortman, M. (1999). Elaboración de corredores o canales endémicos mediante planillas de cálculo. *Revista Panamericana de Salud Pública, 5*(1), 1–8. https://doi.org/10.1590/s1020-49891999000100001

Bowman, L. R., Tejeda, G. S., Coelho, G. E., Sulaiman, L. H., Gill, B. S., McCall, P. J., Olliaro, P. L., Ranzinger, S. R., Quang, L. C., Ramm, R. S., Kroeger, A., & Petzold, M. G. (2016). Alarm variables for dengue outbreaks: A multi-centre study in Asia and Latin America. *PLOS ONE, 11*(6), e0157971. https://doi.org/10.1371/journal.pone.0157971

Brady, O. J., Smith, D. L., Scott, T. W., & Hay, S. I. (2015). Dengue disease outbreak definitions are implicitly variable. *Epidemics, 11*, 92–102. https://doi.org/10.1016/j.epidem.2015.03.002

Bravo Alarcón, C. (2024). *Nivel de leucocitos como factor predictor de dengue grave en pacientes atendidos en un hospital de Trujillo 2024* [Tesis de pregrado, Universidad César Vallejo]. Repositorio UCV. https://hdl.handle.net/20.500.12692/178074

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Colón-González, F. J., Soares Bastos, L., Hofmann, B., Hopkin, A., Harpham, Q., Crocker, T., Amato, R., Ferrario, I., Moschini, F., James, S., Malde, S., Ainscoe, E., Sinh Nam, V., Quang Tan, D., Duc Khoa, N., Harrison, M., Tsarouchi, G., Lumbroso, D., Brady, O. J., & Lowe, R. (2021). Probabilistic seasonal dengue forecasting in Vietnam: A modelling study using superensembles. *PLOS Medicine, 18*(3), e1003542. https://doi.org/10.1371/journal.pmed.1003542

Fang, L., Hu, W., & Pan, G. (2024). Meteorological factors cannot be ignored in machine learning-based methods for predicting dengue, a systematic review. *International Journal of Biometeorology, 68*(3), 401–410. https://doi.org/10.1007/s00484-023-02605-1

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451

Mills, C., Falconi-Agapito, F., Carrera, J.-P., Munayco, C., Kraemer, M. U. G., & Donnelly, C. A. (2025). Multi-model approach to understand and predict past and future dengue epidemic dynamics. *Royal Society Open Science, 12*(11), 241870. https://doi.org/10.1098/rsos.241870

Munayco, C. V., Valderrama Rosales, B. Y., Mateo Lizarbe, S. Y., Yon Fabian, C. R., Peña Sánchez, R., Vásquez Sánchez, C. H., García, M. P., Padilla-Rojas, C., Suárez, V., Sánchez-González, L., Jones, F. K., Kohatsu, L., Adams, L. E., Morgan, J., & Paz-Bailey, G. (2024). Notes from the field: Dengue outbreak — Peru, 2023. *MMWR. Morbidity and Mortality Weekly Report, 73*(4), 86–88. https://doi.org/10.15585/mmwr.mm7304a4

Ong, S. Q., Isawasan, P., Ngesom, A. M. M., Shahar, H., Lasim, A. M., & Nair, G. (2023). Predicting dengue transmission rates by comparing different machine learning models with vector indices and meteorological data. *Scientific Reports, 13*, 19129. https://doi.org/10.1038/s41598-023-46342-2

Ouédraogo, J. C. R. P., Ilboudo, S., Ouédraogo, E., Ouédraogo, W. T., Ouédraogo, S., Samadoulougou, B. C., Kaboré, M., & Savadogo, L. G. B. (2023). Setting dengue fever epidemic thresholds between 2016 and 2021 in the Central Health Region, Burkina Faso: An ecological study. *Journal of Epidemiology and Global Health, 13*(3), 557–565. https://doi.org/10.1007/s44197-023-00137-w

Padilla Pierola, J. H. (2024). *Modelo de aprendizaje supervisado basado en el algoritmo XGBoost para predicción de la incidencia del dengue* [Tesis de título profesional, Universidad Nacional de San Martín]. Repositorio Institucional UNSM. https://gestion-repo.unsm.edu.pe/items/2fe2d775-d473-4cf8-b695-f5ecdee5025f

Rojas-Paucar, W., Carranza López, A. O., Francia-Candiotti, E. J., Pareja-Granda, E. A., & Medina Soto, E. L. (2025). Aplicación del modelo Prophet para la predicción de brotes explosivos de dengue en regiones críticas de Perú: un estudio de series temporales 2000–2024. *Revista Cubana de Investigaciones Biomédicas, 44*(Supl.), e3991. https://revibiomedica.sld.cu/index.php/ibi/article/view/3991

Rufasto Goche, K. S., Lizarbe Castro, M. V., Lozano Zanelly, G. A., Lira Camargo, W. M., Ascayo Velasquez, E. Y., Murillo Carrasco, A. G., & Diaz-Obregón, D. (2025). Epidemiological dynamics of dengue in Peru: Temporal and spatial drivers between 2000 and 2022. *PLOS ONE, 20*(3), e0319708. https://doi.org/10.1371/journal.pone.0319708

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Sánchez López, B. S., Candioti Nolberto, D., Taquía Gutiérrez, J. A., & García López, Y. (2023). Traditional machine learning based on atmospheric conditions for prediction of dengue presence. *Computación y Sistemas, 27*(3), 769–777. https://doi.org/10.13053/CyS-27-3-4383

Sebastianelli, A., Spiller, D., Carmo, R., Wheeler, J., Nowakowski, A., Jacobson, L. V., Kim, D., Barlevi, H., Cordero, Z. E. R., Colón-González, F. J., Lowe, R., Ullo, S. L., & Schneider, R. (2024). A reproducible ensemble machine learning approach to forecast dengue outbreaks. *Scientific Reports, 14*, 3807. https://doi.org/10.1038/s41598-024-52796-9

Umaña, J. D., Montenegro-Torres, J., Otero, J., Tavera-Cifuentes, M. C., Niño-Machado, N., González-Uribe, C., Cordovez, J. M., & Santos-Vega, M. (2026). Endemic channel parametrization in dengue surveillance: Methodological assessment of retrospective windows, outbreak trends, and zero-case periods in Colombia. *JMIR Public Health and Surveillance, 12*, e79914. https://doi.org/10.2196/79914

**Fuentes institucionales y normativas (7):**

Centro Nacional de Epidemiología, Prevención y Control de Enfermedades. (2023, 22 de agosto). *CDC Perú emite Alerta Epidemiológica sobre epidemia de dengue: ante la extensión a nuevos distritos en el Perú*. Ministerio de Salud. https://www.dge.gob.pe/portalnuevo/?p=4861

Centro Nacional de Epidemiología, Prevención y Control de Enfermedades. (2024). *Alerta Epidemiológica AE-CDC N.º 006-2024: Epidemia de dengue en el Perú*. Ministerio de Salud. https://cdn.www.gob.pe/uploads/document/file/6024485/5140727-alerta-epidemiologica-n-006-2024-cdc-minsa-dengue-en-el-peru.pdf

Ministerio de Salud. (2016). *Resolución Ministerial N.º 982-2016/MINSA, que aprueba la NTS N.º 125-MINSA/2016/CDC-INS, Norma Técnica de Salud para la Vigilancia Epidemiológica y Diagnóstico de Laboratorio de Dengue, Chikungunya, Zika y otras Arbovirosis en el Perú*. https://cdn.www.gob.pe/uploads/document/file/192715/191438_RM-_N_982-2016-MINSA.pdf20180904-20266-jqjjk7.pdf

Ministerio de Salud. (2026). *Vigilancia Epidemiológica de dengue* [Conjunto de datos, licencia ODC-BY]. Plataforma Nacional de Datos Abiertos. https://www.datosabiertos.gob.pe/dataset/vigilancia-epidemiológica-de-dengue

Organización Mundial de la Salud. (2025, 21 de agosto). *Dengue and severe dengue* [Nota descriptiva]. https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue

Organización Panamericana de la Salud / Organización Mundial de la Salud. (2026, 18 de febrero). *Actualización Epidemiológica: Dengue en la Región de las Américas*. OPS/OMS. https://www.paho.org/sites/default/files/2026/02/2026-18feb-phe-actualizacion-dengue-es-final.pdf

Presidencia del Consejo de Ministros. (2017). *Decreto Supremo N.º 016-2017-PCM, que aprueba la "Estrategia Nacional de Datos Abiertos Gubernamentales del Perú 2017-2021" y el "Modelo de Datos Abiertos Gubernamentales del Perú"*. https://faolex.fao.org/docs/pdf/per171542.pdf `[V-meta: URL vista en resultados de búsqueda; el PDF no se abrió directamente]`

---

## 6. Pendientes / no verificados (resumen)
1. Fórmula oficial del canal endémico de MINSA/CDC Perú: no existe (o no se encontró) un documento oficial que fije "media + 2 DE"; usar la fórmula como variante citada en la literatura.
2. Valores exactos de AUC/F1 por modelo en Ong et al. (2023) (Tabla 3 no legible).
3. Métricas numéricas de la tesis UNSM (Padilla Pierola, 2024).
4. Contenido metodológico del texto completo de Bortman (1999) (acceso bloqueado); solo metadatos verificados.
5. Cifra oficial de cierre 2023 y 2024 para La Libertad desde CDC/GERESA; cifra nacional 2024 (258 384 casos, provisional) vista solo en fragmentos.
6. Página de términos y condiciones de datosabiertos.gob.pe (no localizada); licencia del dataset sí verificada (ODC-BY).
7. DOI de Rojas-Paucar et al. (2025): la revista no lo muestra.
8. Resumen de Bergmeir y Benítez (2012) no accesible; conclusión detallada no comprobada.
