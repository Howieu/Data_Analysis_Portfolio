"""
生成描述性统计表的代码
将这段代码添加到你的notebook中，在EDA部分运行
"""

import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('MavenRailChallenge.csv', parse_dates=['Departure', 'Scheduled Arrival', 'Actual Arrival'])

# 计算数值变量的描述性统计
numeric_stats = pd.DataFrame({
    'Variable': ['Price (£)', 'Scheduled Journey (min)', 'Actual Journey (min)', 'Delay (min, if delayed)'],
    'Mean': [
        df['Price'].mean(),
        (df['Scheduled Arrival'] - df['Departure']).dt.total_seconds().mean() / 60,
        (df['Actual Arrival'] - df['Departure']).dt.total_seconds().mean() / 60,
        df[df['Journey Status'] == 'Delayed']['DelayInMinutes'].mean() if 'DelayInMinutes' in df.columns else np.nan
    ],
    'Median': [
        df['Price'].median(),
        (df['Scheduled Arrival'] - df['Departure']).dt.total_seconds().median() / 60,
        (df['Actual Arrival'] - df['Departure']).dt.total_seconds().median() / 60,
        df[df['Journey Status'] == 'Delayed']['DelayInMinutes'].median() if 'DelayInMinutes' in df.columns else np.nan
    ],
    'SD': [
        df['Price'].std(),
        (df['Scheduled Arrival'] - df['Departure']).dt.total_seconds().std() / 60,
        (df['Actual Arrival'] - df['Departure']).dt.total_seconds().std() / 60,
        df[df['Journey Status'] == 'Delayed']['DelayInMinutes'].std() if 'DelayInMinutes' in df.columns else np.nan
    ],
    'Min': [
        df['Price'].min(),
        (df['Scheduled Arrival'] - df['Departure']).dt.total_seconds().min() / 60,
        (df['Actual Arrival'] - df['Departure']).dt.total_seconds().min() / 60,
        df[df['Journey Status'] == 'Delayed']['DelayInMinutes'].min() if 'DelayInMinutes' in df.columns else np.nan
    ],
    'Max': [
        df['Price'].max(),
        (df['Scheduled Arrival'] - df['Departure']).dt.total_seconds().max() / 60,
        (df['Actual Arrival'] - df['Departure']).dt.total_seconds().max() / 60,
        df[df['Journey Status'] == 'Delayed']['DelayInMinutes'].max() if 'DelayInMinutes' in df.columns else np.nan
    ]
})

# 格式化Range列
numeric_stats['Range'] = numeric_stats.apply(
    lambda row: f"{row['Min']:.2f}–{row['Max']:.2f}",
    axis=1
)

# 选择需要的列
numeric_stats = numeric_stats[['Variable', 'Mean', 'Median', 'SD', 'Range']]

# 格式化数值（保留2位小数）
for col in ['Mean', 'Median', 'SD']:
    numeric_stats[col] = numeric_stats[col].round(2)

print("="*80)
print("Numeric Variables - Descriptive Statistics")
print("="*80)
print(numeric_stats.to_string(index=False))
print()

# 计算分类变量的频率分布
print("="*80)
print("Categorical Variables - Frequency Distribution")
print("="*80)

# Journey Status
status_counts = df['Journey Status'].value_counts()
status_pct = df['Journey Status'].value_counts(normalize=True) * 100
print("\nJourney Status:")
for status, count in status_counts.items():
    pct = status_pct[status]
    print(f"  {status}: {count} ({pct:.1f}%)")

# Refund Request
refund_counts = df['Refund Request'].value_counts()
refund_pct = df['Refund Request'].value_counts(normalize=True) * 100
print("\nRefund Request:")
for refund, count in refund_counts.items():
    pct = refund_pct[refund]
    print(f"  {refund}: {count} ({pct:.1f}%)")

# Ticket Type
ticket_type_counts = df['Ticket Type'].value_counts()
ticket_type_pct = df['Ticket Type'].value_counts(normalize=True) * 100
print("\nTicket Type:")
for ttype, count in ticket_type_counts.items():
    pct = ticket_type_pct[ttype]
    print(f"  {ttype}: {count} ({pct:.1f}%)")

# Ticket Class
ticket_class_counts = df['Ticket Class'].value_counts()
ticket_class_pct = df['Ticket Class'].value_counts(normalize=True) * 100
print("\nTicket Class:")
for tclass, count in ticket_class_counts.items():
    pct = ticket_class_pct[tclass]
    print(f"  {tclass}: {count} ({pct:.1f}%)")

print()
print("="*80)

# 生成LaTeX表格代码（可选）
print("\nLaTeX Table Code:")
print("="*80)
latex_code = r"""
\begin{table}[H]
\centering
\caption{Descriptive statistics for key variables (n = """ + f"{len(df):,}" + r""")}
\label{tab:descriptive_stats}
\begin{tabular}{lrrrr}
\toprule
\textbf{Variable} & \textbf{Mean} & \textbf{Median} & \textbf{SD} & \textbf{Range} \\
\midrule
"""

for idx, row in numeric_stats.iterrows():
    latex_code += f"{row['Variable']} & {row['Mean']:.2f} & {row['Median']:.2f} & {row['SD']:.2f} & {row['Range']} \\\\\n"

latex_code += r"""\midrule
\multicolumn{5}{l}{\textbf{Categorical Variables}} \\
\midrule
"""

# Add categorical variable summary
status_line = "\\textit{Journey Status:} "
status_parts = []
for status, count in status_counts.items():
    pct = status_pct[status]
    status_parts.append(f"{status} ({pct:.1f}\\%)")
latex_code += f"\\multicolumn{{5}}{{l}}{{{status_line}{', '.join(status_parts)}}} \\\\\n"

refund_line = "\\textit{Refund Request:} "
refund_parts = []
for refund, count in refund_counts.items():
    pct = refund_pct[refund]
    refund_parts.append(f"{refund} ({pct:.1f}\\%)")
latex_code += f"\\multicolumn{{5}}{{l}}{{{refund_line}{', '.join(refund_parts)}}} \\\\\n"

ticket_type_line = "\\textit{Ticket Type:} "
ticket_type_parts = []
for ttype, count in ticket_type_counts.items():
    pct = ticket_type_pct[ttype]
    ticket_type_parts.append(f"{ttype} ({pct:.1f}\\%)")
latex_code += f"\\multicolumn{{5}}{{l}}{{{ticket_type_line}{', '.join(ticket_type_parts)}}} \\\\\n"

ticket_class_line = "\\textit{Ticket Class:} "
ticket_class_parts = []
for tclass, count in ticket_class_counts.items():
    pct = ticket_class_pct[tclass]
    ticket_class_parts.append(f"{tclass} ({pct:.1f}\\%)")
latex_code += f"\\multicolumn{{5}}{{l}}{{{ticket_class_line}{', '.join(ticket_class_parts)}}} \\\\\n"

latex_code += r"""\bottomrule
\end{tabular}
\end{table}
"""

print(latex_code)
