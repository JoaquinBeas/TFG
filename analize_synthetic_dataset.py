import os
from collections import Counter
import matplotlib.pyplot as plt
import os
from collections import Counter
import matplotlib.pyplot as plt
from src.config import OUTPUT_LABELED_DIR
from scripts.generate_synthetic import generate_synthetic_data
from scripts.label_synthetic_data import label_synthetic_data

def is_label_dir_empty(label_dir):
    return not os.path.exists(label_dir) or not any(fname.endswith(".txt") for fname in os.listdir(label_dir))

def analyze_label_distribution(label_dir=OUTPUT_LABELED_DIR, plot=True):
    if is_label_dir_empty(label_dir):
        print("⚠️ No se encontraron etiquetas sintéticas. Generando imágenes y etiquetándolas...")
        generate_synthetic_data()
        label_synthetic_data()

    label_counts = Counter()
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            path = os.path.join(label_dir, filename)
            with open(path, "r") as f:
                label = f.read().strip()
                label_counts[int(label)] += 1

    total = sum(label_counts.values())
    print("\n📊 Distribución de clases en el dataset sintético:")
    for cls in sorted(label_counts.keys()):
        count = label_counts[cls]
        percent = 100 * count / total
        print(f"Clase {cls}: {count} muestras ({percent:.2f}%)")

    if plot:
        classes = sorted(label_counts.keys())
        counts = [label_counts[c] for c in classes]
        plt.bar(classes, counts)
        plt.xticks(classes)
        plt.xlabel("Clase")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de clases en datos sintéticos")
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analyze_label_distribution()


# ¿Qué tan bien representa cada clase?
# Para esto, comparamos la distribución real de las etiquetas sintéticas con una distribución ideal uniforme (esperaríamos 10% por clase en un dataset balanceado).

# ¿Qué clases están sobre o sub-representadas?
# Calculamos el porcentaje real de cada clase.
# Definimos un umbral de tolerancia, por ejemplo ±20% de diferencia sobre el ideal (10%).