import pandas as pd
from scipy import stats

df = pd.read_excel('data.xlsx')  # или pd.read_csv('data.csv')

group_cited = df[df['Group (Factor)'] == 'High_Status']
group_uncited = df[df['Group (Factor)'] == 'Low_Status']

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

levene_results = []

for col in metrics:
  x = group_cited[col].dropna()
  y = group_uncited[col].dropna()

  # Тест Левена относительно медианы (устойчивый к ненормальному распределению)
  stat, p_val = stats.levene(x, y, center='median')

  levene_results.append({
      'Метрика': col,
      'Статистика Левена (W)': round(stat, 3),
      'p-value (Левена)': round(p_val, 4),
      'Однородность дисперсий': 'Да (p > 0.05)' if p_val > 0.05 else 'Нет (p < 0.05)',
  })

res_levene_df = pd.DataFrame(levene_results)
print(res_levene_df.to_string(index=False))
res_levene_df.to_excel('levene_results.xlsx', index=False)
