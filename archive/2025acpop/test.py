import matplotlib.pyplot as plt
import numpy as np

# ===== FIGURE 1: PURE TONE =====
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
t_pt = np.linspace(0, 1, 1000)
pure_tone = 10 * np.sin(2 * np.pi * 20 * t_pt)
ax1.plot(t_pt, pure_tone, color=plt.cm.viridis(0.6), linewidth=6)
ax1.set_ylabel('Amplitude', fontsize=35, fontweight='bold')
ax1.set_xlabel('Time (seconds)', fontsize=35, fontweight='bold')
ax1.set_title('Pure Tone (20 Hz)', fontsize=45, fontweight='bold', pad=20)
ax1.set_xlim(0, 1)
ax1.set_ylim(-15, 15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(labelsize=30)
plt.tight_layout()
plt.savefig('pure_tone.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ===== FIGURE 2: AMPLITUDE MODULATED NOISE =====
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
t_am = np.linspace(0, 1, 10000)
noise = np.random.randn(len(t_am)) * 3
envelope = 5 * (1 + np.sin(2 * np.pi * 4 * t_am))
am_signal = noise * envelope
ax2.plot(t_am, am_signal, color=plt.cm.viridis(0.4), linewidth=0.5, alpha=0.8)
ax2.set_ylabel('Amplitude', fontsize=35, fontweight='bold')
ax2.set_xlabel('Time (seconds)', fontsize=35, fontweight='bold')
ax2.set_title('AM (4 Hz modulation)', fontsize=45, fontweight='bold', pad=20)
ax2.set_xlim(0, 1)
ax2.set_ylim(-20, 20)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(labelsize=30)
plt.tight_layout()
plt.savefig('amplitude_modulated_noise.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ===== FIGURE 3: NATURAL SOUNDS GRID =====
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 10))

# 20x20 grid
n_cells = 20

# Define colors for each sound type
colors_nat = {
    'Frogs': plt.cm.viridis(0.8),
    'Crickets': plt.cm.viridis(0.6),
    'Streamside': plt.cm.viridis(0.4),
    'Bees': plt.cm.viridis(0.2)
}

for i in range(n_cells):
    for j in range(n_cells):
        # Determine which sound type this cell belongs to
        if i < 10 and j < 10:
            color = colors_nat['Frogs']
        elif i < 10 and j >= 10:
            color = colors_nat['Crickets']
        elif i >= 10 and j < 10:
            color = colors_nat['Streamside']
        else:
            color = colors_nat['Bees']

        rect = plt.Rectangle((j, i), 0.95, 0.95,
                             facecolor=color, edgecolor='white',
                             alpha=0.7, linewidth=1.5)
        ax3.add_patch(rect)

# Add labels
ax3.text(5, 20.5, 'Frogs', ha='center', va='bottom', fontsize=35, fontweight='bold')
ax3.text(14.5, 20.5, 'Crickets', ha='center', va='bottom', fontsize=35, fontweight='bold')
ax3.text(6, -1, 'Streamside', ha='center', va='top', fontsize=35, fontweight='bold')
ax3.text(15, -1, 'Bees', ha='center', va='top', fontsize=35, fontweight='bold')

# Add "Within" arrows (vertical, within same category)
ax3.annotate('', xy=(2, 2), xytext=(2, 7),
             arrowprops=dict(arrowstyle='<->', color='black', lw=6))
ax3.text(-1.5, 4.5, 'Within', fontsize=35, fontweight='bold', rotation=90,
         va='center', ha='right')

# Add "Between" arrows (horizontal, between categories)
ax3.annotate('', xy=(7.5, 4.5), xytext=(12.5, 4.5),
             arrowprops=dict(arrowstyle='<->', color='black', lw=6))
ax3.text(10, 8, 'Between', fontsize=35, fontweight='bold',
         ha='center', va='bottom')

ax3.set_xlim(-3, 23)
ax3.set_ylim(-4, 23)
ax3.set_title('Natural Sounds', fontsize=45, fontweight='bold', pad=5)
ax3.set_aspect('equal')
ax3.axis('off')
plt.tight_layout()
plt.savefig('natural_sounds_grid.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ===== FIGURE 4: DATA ORGANIZATION =====
fig4, ax4 = plt.subplots(1, 1, figsize=(12, 11))

# Create grid representing data structure
n_sound_labels = 10
n_neurons = 10
n_trials = 8

# Draw main grid with colorful viridis gradient
for i in range(n_trials):
    for j in range(n_neurons):
        # Create a subtle gradient across the grid
        gradient_value = (i + j) / (n_trials + n_neurons)
        color = plt.cm.viridis(gradient_value * 0.3 + 0.1)
        alpha = 0.4
        edgecolor = '#888888'
        linewidth = 1.5

        # Highlight one example neuron response
        if i == 3 and j == 4:
            color = plt.cm.viridis(0.7)
            alpha = 1.0
            edgecolor = 'black'
            linewidth = 4

        rect = plt.Rectangle((j, i), 0.95, 0.95,
                             facecolor=color, edgecolor=edgecolor,
                             alpha=alpha, linewidth=linewidth)
        ax4.add_patch(rect)

# Add axis labels on the sides
ax4.text(-2.2, n_trials / 2, 'Trials', ha='center', va='center', fontsize=28,
         fontweight='bold', rotation=90)
ax4.text(n_neurons / 2, n_trials + 1.5, 'Neurons', ha='center', va='bottom', fontsize=28,
         fontweight='bold')

# Add sound label axis on left with viridis colors
for i in range(n_sound_labels):
    color_idx = i / n_sound_labels
    rect = plt.Rectangle((-1, -1.5 - i * 0.6), 0.9, 0.55,
                         facecolor=plt.cm.viridis(color_idx), edgecolor='black',
                         alpha=0.7, linewidth=1.5)
    ax4.add_patch(rect)

ax4.text(-3.5, -1.5 - (n_sound_labels / 2) * 0.6, 'Sound Label (Hz)', ha='center', va='center',
         fontsize=28, fontweight='bold', rotation=90)

# Add brain area labels at top with viridis colors
brain_areas = ['Primary', 'Posterior', 'Dorsal', 'Ventral']
brain_colors = [plt.cm.viridis(0.2), plt.cm.viridis(0.4),
                plt.cm.viridis(0.6), plt.cm.viridis(0.8)]

for idx in range(len(brain_areas)):
    rect = plt.Rectangle((idx * 2.5, n_trials + 2.2), 2.4, 0.7,
                         facecolor=brain_colors[idx], edgecolor='black',
                         alpha=0.8, linewidth=1.5)
    ax4.add_patch(rect)

ax4.text(n_neurons / 2, n_trials + 4, 'Brain Area Label', ha='center', va='bottom',
         fontsize=28, fontweight='bold', color=plt.cm.viridis(0.5))

# Legend
ax4.text(n_neurons / 2, -9, '■ = 1 neuron firing rate response',
         ha='center', va='top', fontsize=24, fontweight='bold',
         color=plt.cm.viridis(0.7))

ax4.set_xlim(-4.5, n_neurons + 0.5)
ax4.set_ylim(-10, n_trials + 5)
ax4.set_title('Data Organization', fontsize=32, fontweight='bold', pad=20)
ax4.set_aspect('equal')
ax4.axis('off')
plt.tight_layout()
plt.savefig('data_organization.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("All figures saved successfully!")
print("- pure_tone.png")
print("- amplitude_modulated_noise.png")
print("- natural_sounds_grid.png")
print("- data_organization.png")
