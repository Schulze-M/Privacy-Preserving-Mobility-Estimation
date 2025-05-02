import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import StrMethodFormatter


# Fitness
y1 = [
    0.9993091465804712,
    0.999760694930165 ,
    0.9999291199369503,
    0.9999584566693434,
    0.9999664068165492
]


y2 = [
    0.9402819065670308,
    0.9634003593835189,
    0.9820860067585642,
    0.9881280644621862,
    0.9901430361378598
]

y1_std = np.array([
    9.616195584990378e-06,
    8.454182256017438e-06,
    7.736382946930702e-06,
    7.736338494669477e-06,
    7.5121576291810755e-06
])
y2_std = np.array([
    0.0004576812026283038,
    0.0005899132521924977,
    0.0005673356847226876,
    0.000560290602395344,
    0.0006656272631773318
])

# Set the color palette
c = mpl.colormaps['tab10'].colors

# Set X-axis
x = [1.0, 0.8, 0.5, 0.2, 0.1]

# B x H
plt.figure(figsize=(11.7/(3.8*1.3), 8.27/1.3))

# plt.scatter(4.25, 1, s=112, c=c[0], label="Fitness", marker='|')
# plt.scatter(4.25, 1, s=112, c=c[0], label="F1-Score", marker='x')
# plt.scatter(4.25, 1, s=112, label="Open Problems", facecolors='none', edgecolors=c[4], marker='o')
# plt.scatter(4.25, 1, s=112, c=c[2], label="Sepsis Cases - Event Log", marker='|')
# plt.scatter(4.25, 1, s=112, c=c[1], label="Domestic Declarations", marker='|')
# plt.scatter(4.25, 1, s=112, c=c[1], label="International Declarations", marker='x')
# plt.scatter(4.25, 1, s=112, label="Prepaid Travel Cost", facecolors='none', edgecolors=c[1], marker='o')
# plt.scatter(4.25, 1, s=112, label="Request for Payment", facecolors='none', edgecolors=c[1], marker='s')
# plt.scatter(4.25, 1, s=112, c=c[3], label="BPIC 2017", marker='|')
# plt.scatter(4.25, 1, s=112, c=c[4], label="BPIC 2015_1", marker='|')
# plt.scatter(4.25, 1, s=112, c=c[4], label="BPIC 2015_2", marker='x')
# plt.scatter(4.25, 1, s=112, label="BPIC 2015_3", facecolors='none', edgecolors=c[4], marker='o')
# plt.scatter(4.25, 1, s=112, label="BPIC 2015_4", facecolors='none', edgecolors=c[4], marker='s')
# plt.scatter(4.25, 1, s=112, label="BPIC 2015_5", facecolors='none', edgecolors=c[4], marker='^')

# Plot all graphs
plt.plot(x, y1, c=c[0], linewidth=2, label="Fitness", marker='|', markersize=12)
plt.plot(x, y2, c=c[0], linewidth=2, label="F1-Score", marker='x', markersize=12)
# plt.plot(x, y3, c=c[0], linewidth=2, label="Open Problems", fillstyle='none', marker='o', markersize=12, markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue', markeredgecolor=c[0])
# plt.plot(x, y4, c=c[2], linewidth=2, label="Sepsis Cases - Event Log", marker='|', markersize=12)
# plt.plot(x, y5, c=c[1], linewidth=2, label="Domestic Declarations", marker='|', markersize=12)
# plt.plot(x, y6, c=c[1], linewidth=2, label="International Declarations", marker='x', markersize=12)
# plt.plot(x, y7, c=c[1], linewidth=2, label="Prepaid Travel Cost", fillstyle='none', marker='o', markersize=12, markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue', markeredgecolor=c[1])
# plt.plot(x, y8, c=c[1], linewidth=2, label="Request for Payment", fillstyle='none', marker='s', markersize=12, markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue', markeredgecolor=c[1])
# plt.plot(x, y9, c=c[3], linewidth=2, label="BPIC 2017", marker='|', markersize=12)
# plt.plot(x, y10, c=c[4], linewidth=2, label="BPIC 2015_1", marker='|', markersize=12)
# plt.plot(x, y11, c=c[4], linewidth=2, label="BPIC 2015_2", marker='x', markersize=12)
# plt.plot(x, y12, c=c[4], linewidth=2, label="BPIC 2015_3", fillstyle='none', marker='o', markersize=12, markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue', markeredgecolor=c[4])
# plt.plot(x, y13, c=c[4], linewidth=2, label="BPIC 2015_4", fillstyle='none', marker='s', markersize=12, markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue', markeredgecolor=c[4])
# plt.plot(x, y14, c=c[4], linewidth=2, label="BPIC 2015_5", fillstyle='none', marker='^', markersize=12, markerfacecolor='tab:blue', markerfacecoloralt='lightsteelblue', markeredgecolor=c[4])

plt.fill_between(x=x, y1=y1 - y1_std, y2=y1 + y1_std, alpha=0.1, facecolor=c[0])
plt.fill_between(x=x, y1=y2 - y2_std, y2=y2 + y2_std, alpha=0.125, facecolor=c[0])
# plt.fill_between(x=x, y1=y3 - y3_std, y2=y3 + y3_std, alpha=0.15, facecolor=c[0])
# plt.fill_between(x=x, y1=y4 - y4_std, y2=y4 + y4_std, alpha=0.1, facecolor=c[2])
# plt.fill_between(x=x, y1=y5 - y5_std, y2=y5 + y5_std, alpha=0.1, facecolor=c[1])
# plt.fill_between(x=x, y1=y6 - y6_std, y2=y6 + y6_std, alpha=0.1, facecolor=c[1])
# plt.fill_between(x=x, y1=y7 - y7_std, y2=y7 + y7_std, alpha=0.1, facecolor=c[1])
# plt.fill_between(x=x, y1=y8 - y8_std, y2=y8 + y8_std, alpha=0.1, facecolor=c[1])
# plt.fill_between(x=x, y1=y9 - y9_std, y2=y9 + y9_std, alpha=0.1, facecolor=c[3])
# plt.fill_between(x=x, y1=y10 - y10_std, y2=y10 + y10_std, alpha=0.1, facecolor=c[4])
# plt.fill_between(x=x, y1=y11 - y11_std, y2=y11 + y11_std, alpha=0.1, facecolor=c[4])
# plt.fill_between(x=x, y1=y12 - y12_std, y2=y12 + y12_std, alpha=0.1, facecolor=c[4])
# plt.fill_between(x=x, y1=y13 - y13_std, y2=y13 + y13_std, alpha=0.1, facecolor=c[4])
# plt.fill_between(x=x, y1=y14 - y14_std, y2=y14 + y14_std, alpha=0.1, facecolor=c[4])
# plt.fill_between(x=x, y1=y9 - y9_std, y2=y9 + y9_std, alpha=0.1, facecolor=c[8])  # IM

# Label the axes:
plt.xlabel("$\\varepsilon$", fontsize=14)
plt.ylabel("Eval Results", fontsize=14)

# Limits of the axes:
plt.xlim(0, 4.3)
plt.ylim(0.85, 1.03)
plt.xticks([3, 1, 0.1])

# Invert the X-axis
plt.gca().invert_xaxis()

plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))

# Show the legend on the plot:
# legend = plt.legend(loc='best', ncol=7, fontsize=10)

# Save the legend to a file:
# fig_legend = plt.figure(figsize=(16, .7))
# ax_legend = fig_legend.add_subplot(111)
# ax_legend.legend(handles=legend.legend_handles, labels=[t.get_text() for t in legend.get_texts()], ncol=7, fontsize=10)
# ax_legend.axis('off')
# fig_legend.savefig('legend.pdf', bbox_inches='tight', pad_inches=0)

# plt.savefig('Fitness-Final.pdf', bbox_inches='tight', pad_inches=0)

# Show the plot:
plt.show()