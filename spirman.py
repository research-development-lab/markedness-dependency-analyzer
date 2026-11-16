import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# 1. Загружаем данные (убедитесь, что файл называется data.csv и лежит в той же папке)
df = pd.read_excel('data.xlsx')

# 2. Фильтруем только цитируемые статьи (убираем нули)
df_cited = df[df['Group (Factor)'] == 'High_Status'].copy()

# Целевые переменные успеха и список лингвистических метрик
targets = ['Citation', 'FWCI']
metrics = [
    'PASI (%)',
    'MDD (tokens)',
    'MR (%)',
    'Syn_Equilibrium (%)',
    'Mark_Density (%)',
    'Mmat (%)',
    'Msem (%)',
    'Mimp (%)',
    'Msyn (%)',
    'Mprod (%)',
    'Mstyl (%)',
    'Mstat (%)',
]

results = []

# 3. Расчет корреляции Спирмена для каждой пары
for target in targets:
  temp_results = []
  for col in metrics:
    # Удаляем возможные пропуски в конкретной паре столбцов
    valid_data = df_cited[[target, col]].dropna()

    # Вычисляем коэффициент Спирмена (rho) и p-value
    rho, p_val = stats.spearmanr(valid_data[target], valid_data[col])

    temp_results.append({
        'Целевая метрика': target,
        'Лингвистический параметр': col,
        'Spearman rho': round(rho, 3),
        'p-value': p_val,
    })

  # 4. Поправка на множественные сравнения (метод Бенджамини-Хохберга / FDR) внутри каждой цели
  temp_df = pd.DataFrame(temp_results)
  temp_df['p-adjusted (FDR)'] = multipletests(
      temp_df['p-value'], method='fdr_bh'
  )[1]

  # Округляем p-значения для удобного чтения
  temp_df['p-value'] = temp_df['p-value'].round(4)
  temp_df['p-adjusted (FDR)'] = temp_df['p-adjusted (FDR)'].round(4)

  results.append(temp_df)

# Объединяем результаты в одну таблицу
final_df = pd.concat(results, ignore_index=True)

# Выводим результат в консоль и сохраняем в Excel
print(final_df.to_string(index=False))
final_df.to_excel('spearman_results.xlsx', index=False)
