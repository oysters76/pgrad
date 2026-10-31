import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# MSE per-sample losses (every 20 samples, 3 epochs)
mse_losses = [
    # epoch 0
    349.6295, 61.3579, 104.5733, 109.1174, 115.3600, 68.1245, 124.2972, 41.7205,
    95.0647, 43.7504, 53.6533, 71.5093, 27.5131, 52.5936, 28.1685, 22.9428,
    49.3359, 45.2312, 14.8123, 53.6114, 51.6987, 41.2337, 78.6844, 28.2086,
    38.4844, 14.6014, 26.1078, 34.3641, 27.2570, 25.6309, 36.7189, 29.8042,
    39.4351, 11.2579, 32.5488, 18.9355, 49.5584, 36.3705, 15.3154, 15.2088,
    47.4356, 26.5552, 25.2489, 25.5160, 7.8738, 13.4820, 16.8080, 18.2862,
    41.4225, 20.4369,
    # epoch 1
    33.1925, 35.3635, 17.0646, 17.6524, 14.1752, 26.9461, 17.9563, 20.8068,
    39.1705, 20.5114, 17.8743, 28.2874, 16.1650, 30.9416, 15.5154, 10.0151,
    18.9964, 18.5707, 11.3580, 24.6173, 39.1474, 19.8451, 36.2793, 13.3188,
    21.8815, 9.6338, 9.6828, 18.2606, 11.3480, 14.4585, 27.1005, 13.0152,
    24.0345, 9.8802, 17.6902, 16.3511, 27.0972, 20.8851, 6.9166, 9.7241,
    22.7991, 18.9993, 12.2508, 14.1986, 5.8017, 9.7654, 11.5956, 8.9030,
    25.4312, 13.5094,
    # epoch 2
    20.5838, 19.8964, 10.6918, 9.6571, 9.8996, 18.8595, 12.4092, 15.4673,
    23.0057, 16.2617, 10.1196, 16.8234, 9.9062, 22.7992, 8.8000, 6.9871,
    13.0852, 12.0562, 10.1443, 14.0165, 29.1983, 13.3476, 22.9795, 8.4400,
    14.3859, 6.7154, 4.5619, 12.0902, 6.4042, 9.5984, 20.4405, 6.2874,
    17.2543, 8.8915, 12.6401, 11.7603, 16.2523, 14.1177, 3.8104, 7.3186,
    12.7000, 14.9813, 7.5926, 9.5346, 4.5695, 7.7742, 8.4394, 5.5942,
    16.6235, 8.4567,
]

# CE per-sample losses (every 20 samples, 3 epochs)
ce_losses = [
    # epoch 0
    26.9030, 13.4760, 0.4519, 12.6624, 9.6909, 2.4806, 5.3910, 0.5644,
    8.5745, 10.0974, 0.0417, 3.7360, 2.2769, 0.6704, 0.0279, 5.4159,
    3.2302, 4.3789, 5.4074, 1.9974, 0.0660, 6.4244, 4.7790, 3.7389,
    11.2129, 13.2758, 1.8816, 7.6411, 4.6481, 0.0560, 2.7473, 9.4431,
    0.0002, 0.3236, 1.8535, 0.0381, 3.5303, 5.8605, 1.3677, 0.7524,
    2.0242, 5.6322, 4.9226, 3.1092, 0.0014, 3.4463, 1.9330, 4.3357,
    0.4526, 0.5565,
    # epoch 1
    1.5725, 0.4163, 0.3552, 0.4293, 11.3345, 1.6432, 4.9231, 0.4884,
    2.0902, 6.6881, 0.1640, 0.0113, 1.8231, 0.6199, 0.0172, 3.4671,
    0.2075, 2.4956, 1.1840, 0.8301, 0.5267, 2.0117, 0.1887, 0.4056,
    8.2455, 6.5447, 1.4770, 3.8773, 1.5508, 0.0379, 1.1867, 4.9396,
    0.0002, 0.3410, 0.2993, 0.0558, 2.5543, 5.5205, 0.6031, 0.6694,
    1.0294, 3.5922, 3.2025, 0.2052, 0.0002, 1.9480, 1.7974, 2.1275,
    0.1087, 0.2546,
    # epoch 2
    1.0086, 0.1284, 0.1417, 0.0102, 9.0237, 1.5547, 2.7095, 0.7854,
    1.5552, 6.4854, 0.0712, 0.0058, 1.6800, 0.4800, 0.0133, 3.5116,
    0.0573, 1.2510, 0.3246, 0.2393, 0.6239, 1.4806, 0.0443, 0.2179,
    5.4216, 4.3366, 1.1162, 2.1500, 0.6034, 0.0198, 0.5360, 2.7200,
    0.0001, 0.2447, 0.8534, 0.0216, 2.3686, 5.4982, 0.3334, 0.4052,
    1.1750, 1.7341, 2.8003, 0.0072, 0.0001, 0.9418, 1.8822, 1.1193,
    0.0318, 0.0770,
]

mse_epoch_acc = [0.089, 0.100, 0.090]
mse_epoch_loss = [44.2693, 17.6926, 11.6255]

ce_epoch_acc = [0.509, 0.710, 0.781]
ce_epoch_loss = [2.6979, 0.9626, 0.6802]

epochs = [0, 1, 2]
steps = list(range(len(mse_losses)))

plt.style.use('seaborn-v0_8-whitegrid')

# --- Plot 1: Per-sample loss comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(steps, mse_losses, linewidth=0.6, color='#e74c3c', alpha=0.7)
ax1.set_title('MSE Loss (per sample)')
ax1.set_xlabel('Training step')
ax1.set_ylabel('Loss')
for e in [50, 100]:
    ax1.axvline(x=e, color='gray', linestyle='--', alpha=0.4)

ax2.plot(steps, ce_losses, linewidth=0.6, color='#2980b9', alpha=0.7)
ax2.set_title('Cross-Entropy Loss (per sample)')
ax2.set_xlabel('Training step')
ax2.set_ylabel('Loss')
for e in [50, 100]:
    ax2.axvline(x=e, color='gray', linestyle='--', alpha=0.4)

fig.suptitle('Per-Sample Loss: MSE vs Cross-Entropy', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig('docs/loss_comparison.png', dpi=150)
print('saved docs/loss_comparison.png')

# --- Plot 2: Accuracy comparison ---
fig2, ax = plt.subplots(figsize=(8, 5))

x = range(3)
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], [a * 100 for a in mse_epoch_acc], width, label='MSE', color='#e74c3c', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], [a * 100 for a in ce_epoch_acc], width, label='Cross-Entropy', color='#2980b9', alpha=0.8)

ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Training Accuracy: MSE vs Cross-Entropy', fontsize=14, fontweight='bold')
ax.set_xticks(list(x))
ax.set_xticklabels(['Epoch 0', 'Epoch 1', 'Epoch 2'])
ax.legend()
ax.set_ylim(0, 100)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5, f'{bar.get_height():.1f}%', ha='center', fontsize=9)

fig2.tight_layout()
fig2.savefig('docs/accuracy_comparison.png', dpi=150)
print('saved docs/accuracy_comparison.png')

# --- Plot 3: CE learning curve ---
fig3, ax1 = plt.subplots(figsize=(8, 5))

color_loss = '#2980b9'
color_acc = '#27ae60'

ax1.plot(epochs, ce_epoch_loss, 'o-', color=color_loss, linewidth=2, markersize=8, label='Avg Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Average Loss', color=color_loss)
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_xticks(epochs)

ax2 = ax1.twinx()
ax2.plot(epochs, [a * 100 for a in ce_epoch_acc], 's-', color=color_acc, linewidth=2, markersize=8, label='Accuracy')
ax2.set_ylabel('Accuracy (%)', color=color_acc)
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.set_ylim(0, 100)

fig3.suptitle('Cross-Entropy Classifier: Learning Curve', fontsize=14, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

fig3.tight_layout()
fig3.savefig('docs/epoch_summary.png', dpi=150)
print('saved docs/epoch_summary.png')
