import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Experiment:
    """
    Un tracker di esperimenti stile WandB per il framework Vathos.
    Gestisce configurazioni, metriche, salvataggi e visualizzazioni avanzate.
    """

    def __init__(self, project_name: str, exp_name: str = None, config: dict = None, root_dir: str = "runs"):
        self.project_name = project_name

        # Genera un nome univoco se non fornito
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = exp_name if exp_name else f"run_{timestamp}"
        self.config = config or {}

        # Setup directory dell'esperimento
        self.run_dir = Path(root_dir) / self.project_name / self.exp_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Dizionari per le metriche.
        # Formato: {"loss": {"steps": [1,2,3], "values": [0.9, 0.8, 0.7]}}
        self.metrics = {}

        # Salva subito la configurazione
        self._save_config()
        print(f"🚀 Esperimento '{self.exp_name}' inizializzato in {self.run_dir}")

    def log(self, metrics_dict: dict, step: int):
        """
        Registra un dizionario di metriche per un dato step.
        Es: exp.log({"train_loss": 0.5, "val_accuracy": 0.8}, step=100)
        """
        for k, v in metrics_dict.items():
            # Converte tensori PyTorch o Numpy in float standard
            if hasattr(v, "item"):
                v = v.item()

            if k not in self.metrics:
                self.metrics[k] = {"steps": [], "values": []}

            self.metrics[k]["steps"].append(step)
            self.metrics[k]["values"].append(v)

    def _save_config(self):
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=4)

    def save(self):
        """Salva tutte le metriche correnti in un file JSON."""
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)
        # print(f"💾 Metriche salvate in {self.run_dir / 'metrics.json'}")

    @classmethod
    def load(cls, run_dir: str):
        """
        Inizializza un oggetto Experiment a partire da una cartella salvata.
        Permette di ricaricare e plottare run vecchie!
        """
        run_path = Path(run_dir)

        # Estrai project e exp name dal path (es. runs/Progetto/Run)
        exp_name = run_path.name
        project_name = run_path.parent.name

        with open(run_path / "config.json", "r") as f:
            config = json.load(f)

        # Ricostruisce l'oggetto bypassando la creazione della cartella root
        exp = cls(project_name=project_name, exp_name=exp_name, config=config, root_dir=run_path.parent.parent)

        with open(run_path / "metrics.json", "r") as f:
            exp.metrics = json.load(f)

        print(f"📂 Esperimento '{exp_name}' caricato con successo.")
        return exp

    def _smooth_curve(self, points, factor=0.85):
        """Applica un Exponential Moving Average (EMA) per rendere i plot bellissimi."""
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    def plot(self, smoothing: float = 0.85, save_fig: bool = False, figsize=(10, 6)):
        """
        Genera plot in stile accademico/WandB.
        Mostra le curve originali in trasparenza e le curve smoothate in evidenza.
        """
        if not self.metrics:
            print("Nessuna metrica da plottare!")
            return

        # Stile estetico "stupendo" (simile a Seaborn/WandB)
        plt.style.use('bmh')  # Usa un tema base pulito integrato in matplotlib

        num_metrics = len(self.metrics)
        cols = 2 if num_metrics > 1 else 1
        rows = (num_metrics + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Colori moderni
        colors = list(mcolors.TABLEAU_COLORS.values())

        for idx, (metric_name, data) in enumerate(self.metrics.items()):
            ax = axes[idx]
            steps = data["steps"]
            values = data["values"]
            color = colors[idx % len(colors)]

            # Plotta i dati raw (sbiaditi)
            ax.plot(steps, values, color=color, alpha=0.2, label='Raw')

            # Plotta i dati smoothati (in evidenza)
            if smoothing > 0 and len(values) > 5:
                smoothed = self._smooth_curve(values, factor=smoothing)
                ax.plot(steps, smoothed, color=color, linewidth=2.5, label=f'Smoothed (α={smoothing})')

            ax.set_title(metric_name.replace("_", " ").title(), fontsize=14, fontweight='bold')
            ax.set_xlabel("Steps", fontsize=11)
            ax.set_ylabel("Value", fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc="best", frameon=True, shadow=True)

        for idx in range(num_metrics, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f"Experiment: {self.project_name} - {self.exp_name}", fontsize=16, y=1.02)
        plt.tight_layout()

        if save_fig:
            fig_path = self.run_dir / "plots.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot salvato in {fig_path}")

        plt.show()


