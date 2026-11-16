import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# 1. Загружаем данные
df = pd.read_excel('data.xlsx')

# 2. Разделяем выборку на две группы: цитированные и нецитированные
group_cited = df[df['Group (Factor)'] == 'High_Status']
group_uncited = df[df['Group (Factor)'] == 'Low_Status']

# Список 12 лингвистических метрик (исключаем Citation и FWCI)
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

# 3. Проводим расчеты для каждой метрики
for col in metrics:
  x = group_cited[col].dropna()
  y = group_uncited[col].dropna()

  # Медиана и квартили (Q1 - Q3)
  med_x, q1_x, q3_x = np.median(x), np.percentile(x, 25), np.percentile(x, 75)
  med_y, q1_y, q3_y = np.median(y), np.percentile(y, 25), np.percentile(y, 75)

  # Двусторонний U-тест Манна-Уитни
  u_stat, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')

  # Размер эффекта: рангово-бисериальная корреляция (r_rb)
  # Положительное значение означает, что показатель выше у цитируемых (High_Status)
  n1, n2 = len(x), len(y)
  r_rb = (2 * u_stat) / (n1 * n2) - 1

  results.append({
      'Метрика': col,
      'Cited Me (Q1–Q3)': f'{med_x:.2f} ({q1_x:.2f}–{q3_x:.2f})',
      'Uncited Me (Q1–Q3)': f'{med_y:.2f} ({q1_y:.2f}–{q3_y:.2f})',
      'U-статистика': u_stat,
      'p-value': p_val,
      'Effect size (r_rb)': round(r_rb, 3),
  })

res_df = pd.DataFrame(results)

# 4. Поправка на множественные сравнения (метод Бенджамини-Хохберга / FDR)
res_df['p-adjusted (FDR)'] = multipletests(res_df['p-value'], method='fdr_bh')[1]

# Округление значений для красивой таблицы
res_df['p-value'] = res_df['p-value'].round(4)
res_df['p-adjusted (FDR)'] = res_df['p-adjusted (FDR)'].round(4)

# Вывод в окно Python и сохранение в Excel
print(res_df.to_string(index=False))
res_df.to_excel('mann_whitney_results.xlsx', index=False)
