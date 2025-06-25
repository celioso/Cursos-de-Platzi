# Curso de Fundamentos de Machine Learning

1. **¿Por qué es importante usar torch.no_grad() al evaluar el modelo en PyTorch?**
   
**R//=** torch.no_grad() impide cálculo y almacenamiento de gradientes, acelerando la evaluación y reduciendo memoria

2. **¿Por qué es fundamental distinguir modelos supervised de unsupervised al analizar los datos de Cebollitas?**
 
**R//=** Porque supervised usa datos etiquetados para predecir resultados, mientras unsupervised identifica patrones.

3. **Si un modelo de regresión lineal muestra R² negativo y alto RMSE, ¿qué estrategia es más adecuada?**
   
**R//=** Probar un árbol de decisión para capturar relaciones no lineales y reducir RMSE y MAE significativamente.

4. **¿Por qué no sería ideal usar posesión (%) como predictor principal si su correlación con goles_local es 0.17?**
   
**R//=** Una correlación de 0.17 es muy baja, por lo que posesión aporta poca información predictiva para goles_local.

5. **Dado que un mediocampista no marca ni asiste, ¿qué implica analizarlo con un modelo no supervisado?**
    
**R//=** Encontrar patrones de rendimiento sin depender de variables objetivo como goles o asistencias

6. **¿Cuál sería el impacto de no escalar ninguna variable al entrenar un modelo de ML con el dataset de Cebollitas FC?**
    
**R//=** El modelo asignaría pesos desproporcionados a variables con mayor magnitud produciendo predicciones inexactas

7. **¿Por qué Ridge regression evita overfitting al predecir la diferencia de goles en el pipeline supervisado?**
    
**R//=** Porque añade un término de penalización L2 que reduce coeficientes grandes y mejora la generalización

8. **Si tienes métricas de partidos como features y labels con resultados, ¿qué describe 'machine learning supervisado'?**
    
**R//=** Entrenar modelos usando features y labels históricos para predecir resultados futuros de partidos

9. **¿Cuál sería el impacto de cambiar a un Random Forest tras un modelo lineal con R² negativo?**
    
**R//=** Aumentaría R² y reduciría RMSE al capturar patrones complejos, mejorando el poder explicativo del modelo

10. **Si el cuerpo técnico quiere comparar delanteros y mediocampistas, ¿cómo usarías el widget interactive con PCA?**
    
**R//=** Seleccionar PC1 y PC2 en dropdowns para delanteros y mediocampistas y observar scatter plot interactivo

11. **¿Qué método de seaborn utilizas para visualizar la frecuencia de cada categoría en la columna 'sentiment'?**
    
**R//=** sns.countplot(x='sentiment', data=df) muestra la frecuencia de cada categoría de sentiment en un gráfico de barras.

12. **Dado un dropdown que filtra jugadores por cluster, ¿cómo mejora tu análisis en reuniones con cuerpo técnico?**
    
**R//=** Permitir mostrar en tiempo real los jugadores de un cluster facilita la toma de decisiones con el cuerpo técnico.

13. **Dado que queremos predecir resultados, ¿por qué es valiosa la variable diferencia de goles como target?**
    
**R//=** Porque resume el rendimiento en una métrica clara que el modelo aprende para predecir el resultado de partidos

14. **Dado un error bajo en X_train y alto en X_test, ¿por qué es esencial separar datos con train_test_split?**
    
**R//=** Porque ayuda a evaluar generalización y detectar overfitting al medir desempeño en datos no vistos.

15. **Dado un modelo con outliers frecuentes en predicciones de goles, ¿qué métrica usarías para un diagnóstico robusto del error?**
    
**R//=** Usaría MAE porque promedia errores absolutos, no pondera outliers y muestra el error medio en goles.