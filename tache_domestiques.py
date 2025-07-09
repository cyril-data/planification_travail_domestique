import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import re

# Chargement des fichiers
tasks_df = pd.read_csv("taches.csv")
participants_df = pd.read_csv("participants.csv")

# Nettoyage des noms
participants_df["Participants"] = participants_df["Participants"].str.strip()
participants = participants_df.to_dict("records")


# Convertir le temps des tâches en minutes
def convert_time(time_str):
    if "h" in time_str:
        h, m = time_str.replace(" min", "").split("h")
        return float(h.strip()) * 60 + float(m.strip()) if m else float(h.strip()) * 60
    return float(time_str.replace(" min", ""))


tasks_df["Temps_minutes"] = tasks_df["Temps estimé"].apply(convert_time)

# Liste des jours
jours = ["Jeudi", "Vendredi", "Samedi", "Dimanche"]

# Séparer hommes et femmes
men = [p for p in participants if p["Genre"] == "M"]
women = [p for p in participants if p["Genre"] == "F"]
nb_homme = len(men)
nb_femme = len(women)

# Temps total à répartir
total_time = (tasks_df["Temps_minutes"] * tasks_df["Nbr de participants par tache"]).sum()

# Objectif : H = 2 x F
temps_moyen_femme = total_time / (2 * nb_homme + nb_femme)
temps_moyen_homme = 2 * temps_moyen_femme

# Temps visé par personne
temps_visé = {p["Participants"]: temps_moyen_homme if p["Genre"] == "M" else temps_moyen_femme for p in participants}

# Temps déjà assigné
temps_assigné = {p["Participants"]: 0 for p in participants}

# Affectations finales
assignments = []


# Fonction pour filtrer personnes disponibles un jour donné
def dispo(p, jour):
    return int(p[jour]) == 1


# Traitement de chaque tâche
for _, task in tasks_df.iterrows():
    jour = task["Jour"]
    needed = task["Nbr de participants par tache"]
    tps = task["Temps_minutes"]
    genre = task["Genre"]

    # Participants disponibles ce jour
    dispo_today = [p for p in participants if dispo(p, jour)]

    # Filtrage par genre
    if genre == "M":
        dispo_today = [p for p in dispo_today if p["Genre"] == "M"]
    elif genre == "F":
        dispo_today = [p for p in dispo_today if p["Genre"] == "F"]

    # Score = temps déjà assigné / temps visé => les plus "légers" en premier
    dispo_sorted = sorted(dispo_today, key=lambda p: temps_assigné[p["Participants"]] / temps_visé[p["Participants"]])

    # Sélection des N participants les moins chargés
    selected = dispo_sorted[:needed]

    # Si pas assez de monde dispo du bon genre => compléter par n’importe qui de dispo
    if len(selected) < needed:
        remaining = needed - len(selected)
        other_pool = [p for p in dispo_today if p not in selected]
        other_sorted = sorted(
            other_pool, key=lambda p: temps_assigné[p["Participants"]] / temps_visé[p["Participants"]]
        )
        selected += other_sorted[:remaining]

    # Enregistrer les assignations
    for person in selected:
        assignments.append(
            {
                "Jour": jour,
                "Horaire": task["Horaire"],
                "Tâche": task["Tâche"],
                "Participant": person["Participants"],
                "Genre": person["Genre"],
                "Temps": tps,
                "Équipe": task["Équipe"],
            }
        )
        temps_assigné[person["Participants"]] += tps

# Création du tableau final
assignments_df = pd.DataFrame(assignments)

# Analyse du résultat
total_h = sum(temps_assigné[p["Participants"]] for p in men)
total_f = sum(temps_assigné[p["Participants"]] for p in women)
ratio = (total_h / nb_homme) / (total_f / nb_femme)


def format_h_m(minutes):
    h = int(minutes // 60)
    m = int(round(minutes % 60))
    return f"{h}h{m:02d}"


print("\n")
print(f"Nombre de femmes : {nb_femme:.0f}")
print(f"Nombre d'hommes : {nb_homme:.0f}")
print(f"Ratio H/F : {(nb_homme / nb_femme):.2f}")
print(f"Temps total homme : {format_h_m(total_h)}")
print(f"Temps total femme : {format_h_m(total_f)}")
print(f"Temps moyen homme pour l'ensemble du séjour : {format_h_m(total_h / nb_homme)}")
print(f"Temps moyen femme pour l'ensemble du séjour: {format_h_m(total_f / nb_femme)}")

print(f"✅ Ratio H/F (objectif 2.0) : {ratio:.2f}")

# Liste des temps par personne
temps_par_homme = [temps_assigné[p["Participants"]] for p in men]
temps_par_femme = [temps_assigné[p["Participants"]] for p in women]

# Moyennes et écart-types
std_h = np.std(temps_par_homme)
std_f = np.std(temps_par_femme)
moy_h = total_h / nb_homme
moy_f = total_f / nb_femme

# Intervalle à 90% ≈ ±1.645σ
facteur_90 = 1.645
bas_h = moy_h - facteur_90 * std_h
haut_h = moy_h + facteur_90 * std_h
bas_f = moy_f - facteur_90 * std_f
haut_f = moy_f + facteur_90 * std_f

print(
    f"Écart-type temps hommes : {format_h_m(std_h)} → ~90% ont travaillé entre {format_h_m(bas_h)} et {format_h_m(haut_h)}"
)
print(
    f"Écart-type temps femmes : {format_h_m(std_f)} → ~90% ont travaillé entre {format_h_m(bas_f)} et {format_h_m(haut_f)}"
)

# Export
assignments_df.to_csv("repartition_taches.csv", index=False)
print("Répartition sauvegardée dans 'repartition_taches.csv'")
grouped_df = (
    assignments_df.groupby(["Jour", "Horaire", "Tâche", "Équipe"])
    .agg({"Participant": lambda x: ", ".join(sorted(x))})
    .reset_index()
)

# Ordre des jours
jour_order = ["jeudi", "vendredi", "samedi", "dimanche"]


# Convertir l'heure en valeur numérique pour tri
def heure_to_minutes(horaire):
    h = int(horaire.lower().replace("h", "").strip())
    return h * 60  # simplification (si format "18h")


grouped_df["Jour"] = pd.Categorical(grouped_df["Jour"], categories=jour_order, ordered=True)
grouped_df["Horaire_num"] = grouped_df["Horaire"].apply(heure_to_minutes)

# Tri chronologique
grouped_df = grouped_df.sort_values(by=["Jour", "Horaire_num"])

# Suppression colonne temporaire
grouped_df.drop(columns="Horaire_num", inplace=True)

# Export
grouped_df.to_csv("repartition_par_tache.csv", index=False)
print("✅ Fichier 'repartition_par_tache.csv' généré avec les participants regroupés par tâche.")


# Créer une colonne tâche-formatée pour chaque ligne
assignments_df["Tache_label"] = (
    assignments_df["Jour"] + "\n" + assignments_df["Horaire"] + "\n" + assignments_df["Tâche"]
)
# Remplacement propre sans warning
pivot_df = (
    assignments_df.pivot_table(index="Participant", columns="Tache_label", aggfunc="size", fill_value=0)
    .astype(str)
    .replace("0", "")
    .replace("1", "✔")  # Remplacement ici
    .reset_index()
)

# Tri par ordre alphabétique des participants
pivot_df = pivot_df.sort_values(by="Participant")

# Extraire les colonnes de tâches
tache_cols = [col for col in pivot_df.columns if col != "Participant"]


def extraire_ordre(col):
    try:
        lines = col.split("\n")
        jour = lines[0].lower()
        heure_str = lines[1].replace("h", "").strip()
        heure = int(heure_str)
        jour_index = jour_order.index(jour)
        return (jour_index, heure)
    except Exception as e:
        print(f"⚠️ Erreur dans extraire_ordre pour la colonne '{col}': {e}")
        return (99, 0)  # met à la fin


# Trier les colonnes selon (jour, heure)
tache_cols_sorted = sorted(tache_cols, key=extraire_ordre)

# Réordonner les colonnes du tableau final
pivot_df = pivot_df[["Participant"] + tache_cols_sorted]

# Export
pivot_df.to_csv("repartition_par_participant.csv", index=False)
print("✅ Fichier 'repartition_par_participant.csv' généré avec les participants en ligne et les tâches en colonnes.")

# ================= PLOT PDF =================

# Charger le CSV
# Rechargement des données depuis le CSV (sans index_col)
df_participant = pd.read_csv("repartition_par_participant.csv")

jours = ["jeudi", "vendredi", "samedi", "dimanche"]
day_colors = {"jeudi": "#E0F7FA", "vendredi": "#FFF3E0", "samedi": "#E8F5E9", "dimanche": "#F3E5F5"}


def forced_wrap_after_colon(text, width=30):
    # On découpe la chaîne en 2 parties : avant et après les ":"
    if ":" in text:
        before_colon, after_colon = text.split(":", 1)  # split en 2 parties max
        # on ajoute un retour à la ligne explicitement après le ":"
        before_colon += ":"
        # on wrappe la partie après le ":"
        after_colon_wrapped = textwrap.fill(after_colon.strip(), width=width)
        # on reconstitue avec un saut de ligne forcé
        return before_colon + "\n" + after_colon_wrapped
    else:
        # Pas de ":", on wrappe normalement
        return textwrap.fill(text, width=width)


with PdfPages("repartition_par_participant_par_jour.pdf") as pdf:

    for jour in jours:
        jour_cols = [col for col in df_participant.columns if col.startswith(jour)]
        if not jour_cols:
            continue

        df_jour = df_participant[["Participant"] + jour_cols].copy()

        # Nettoyage et wrap
        presence_flags = df_jour.iloc[:, 1:].apply(
            lambda col: col.map(lambda x: str(x).strip() != "" and str(x).lower() != "nan")
        )

        wrapped_columns = ["Participant"]

        for col in jour_cols:
            try:
                if "\n" in col:
                    parts = col.split("\n")
                    jour_nom = parts[0]
                    heure = parts[1]
                    description = parts[2] if len(parts) > 2 else ""
                else:
                    parts = col.split(" ")
                    jour_nom = parts[0]
                    heure = parts[1] if len(parts) > 1 else ""
                    description = " ".join(parts[2:]) if len(parts) > 2 else ""

                # Forcer retour à la ligne après ":" (ton code existant)
                wrapped_desc = forced_wrap_after_colon(description, width=30)

                # **Forcer retour à la ligne à chaque espace** en remplaçant les espaces par "\n"
                wrapped_desc = wrapped_desc.replace(" ", "\n")

                wrapped_label = f"{jour_nom}\n{heure}\n{wrapped_desc}"
                wrapped_columns.append(wrapped_label)
            except Exception:
                wrapped_columns.append(textwrap.fill(col, width=30))

        wrapped_values = []
        for idx, row in df_jour.iterrows():
            new_row = [row["Participant"]]
            for cell in row[1:]:
                # Nettoyer la cellule
                if pd.isna(cell):
                    new_cell = ""
                elif isinstance(cell, (list, np.ndarray)) and len(cell) == 0:
                    new_cell = ""
                else:
                    # Wrap long strings uniquement
                    if isinstance(cell, str) and len(cell) > 15:
                        new_cell = textwrap.fill(cell, width=20)
                    else:
                        new_cell = cell
                new_row.append(new_cell)
            wrapped_values.append(new_row)

        n_cols = len(wrapped_columns)
        n_rows = len(wrapped_values)
        col_width = 1.5
        row_height = 0.3
        fig_width = max(23.4, col_width * (n_cols + 1))
        fig_height = min(16.5, row_height * (n_rows + 2))

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table_data = [wrapped_columns] + wrapped_values
        table = ax.table(cellText=table_data, loc="center", cellLoc="center")

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.0)

        for (row, col), cell in table.get_celld().items():
            cell.set_text_props(ha="center", va="center", wrap=True)

            if row == 0:

                cell.set_fontsize(14)  # Taille plus grande
                cell.get_text().set_fontweight("bold")  # Gras
                cell.set_facecolor(day_colors.get(jour, "#FFFFFF"))
                cell.set_height(0.1)

            elif col == 0 and row > 0:
                # Première colonne (noms des participants)
                cell.set_fontsize(14)  # Plus grand que les autres lignes
                cell.set_facecolor("#FFFFFF")  # Couleur de fond blanche ou autre

            elif row > 0 and col > 0:
                val = table_data[row][col]
                # Colorier en gris uniquement si la cellule n'est pas vide et pas nan
                if isinstance(val, str) and val.strip() != "":
                    cell.set_facecolor("#D3D3D3")
                # Sinon blanc par défaut (tu peux aussi mettre explicitement)
                else:
                    cell.set_facecolor("#FFFFFF")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print("✅ PDF multi-pages par jour généré proprement avec cases grisées et blanches.")
