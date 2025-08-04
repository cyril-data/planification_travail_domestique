# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages
# from datetime import datetime
# import textwrap

# # Chargement des fichiers
# tasks_df = pd.read_csv("taches.csv")
# participants_df = pd.read_csv("participants.csv")

# # Nettoyage des noms
# participants_df["Participants"] = participants_df["Participants"].str.strip()
# participants = participants_df.to_dict("records")


# # Conversion des durées en minutes
# def convert_time(time_str):
#     if "h" in time_str:
#         h, m = time_str.replace(" min", "").split("h")
#         return float(h.strip()) * 60 + float(m.strip()) if m else float(h.strip()) * 60
#     return float(time_str.replace(" min", ""))


# tasks_df["Temps_minutes"] = tasks_df["Temps estimé"].apply(convert_time)

# # Trier les tâches par jour/horaire
# jour_order = {"jeudi": 0, "vendredi": 1, "samedi": 2, "dimanche": 3}
# tasks_df["Jour_order"] = tasks_df["Jour"].str.lower().map(jour_order)


# def parse_horaire(horaire_str):
#     try:
#         return datetime.strptime(horaire_str, "%Hh%M")
#     except ValueError:
#         try:
#             return datetime.strptime(horaire_str, "%Hh")
#         except ValueError:
#             return pd.NaT


# tasks_df["Horaire_order"] = tasks_df["Horaire"].apply(parse_horaire)
# tasks_df = tasks_df.sort_values(["Jour_order", "Horaire_order"])

# # Séparation hommes/femmes
# men = [p for p in participants if p["Genre"] == "H"]
# women = [p for p in participants if p["Genre"] == "F"]
# nb_homme = len(men)
# nb_femme = len(women)

# # Calcul du temps total à répartir
# total_time = (tasks_df["Temps_minutes"] * tasks_df["Nbr de participants par tache"]).sum()
# temps_moyen_femme = total_time / (2 * nb_homme + nb_femme)
# temps_moyen_homme = 2 * temps_moyen_femme

# # Initialisation des structures de suivi
# temps_vise = {p["Participants"]: temps_moyen_homme if p["Genre"] == "H" else temps_moyen_femme for p in participants}
# temps_impose = {p["Participants"]: 0 for p in participants}
# assignments = {p["Participants"]: [] for p in participants}


# # Fonction d’affectation
# def get_best_fit(genre, n):
#     pool = men if genre == "H" else women
#     pool_sorted = sorted(pool, key=lambda x: temps_impose[x["Participants"]])
#     return pool_sorted[:n]


# # Assignation des tâches
# for _, row in tasks_df.iterrows():
#     genre = row["Genre"]
#     nb_needed = row["Nbr de participants par tache"]

#     if genre == "H":
#         candidats = get_best_fit("H", nb_needed)
#     elif genre == "F":
#         candidats = get_best_fit("F", nb_needed)
#     else:
#         # Mixte : on priorise ceux avec le plus de "manque" par rapport à leur objectif
#         all_sorted = sorted(
#             participants, key=lambda x: temps_impose[x["Participants"]] / temps_vise[x["Participants"]]
#         )
#         candidats = all_sorted[:nb_needed]

#     for p in candidats:
#         nom = p["Participants"]
#         temps_impose[nom] += row["Temps_minutes"]
#         assignments[nom].append(
#             {
#                 "Jour": row["Jour"],
#                 "Horaire": row["Horaire"],
#                 "Tâche": row["Tâche"],
#                 "Équipe": row["Équipe"],
#                 "Durée": row["Temps_minutes"],
#             }
#         )


# # Génération des feuilles par personne dans des fichiers PDF
# def generate_pdf(assignments, genre, filename):
#     with PdfPages(filename) as pdf:
#         for nom, taches in assignments.items():
#             if participants_df.loc[participants_df["Participants"] == nom, "Genre"].values[0] != genre:
#                 continue
#             fig, ax = plt.subplots(figsize=(8.5, 11))
#             ax.axis("off")
#             lines = [f"Tâches attendues pour : {genre}", ""]

#             taches_sorted = sorted(taches, key=lambda x: (jour_order[x["Jour"].lower()], parse_horaire(x["Horaire"])))

#             for t in taches_sorted:
#                 ligne = f"{t['Jour'].capitalize()} à {t['Horaire']} - {t['Tâche']}, {int(t['Durée'])} min)"
#                 lines.extend(textwrap.wrap(ligne, 90))

#             ax.text(0, 1, "\n".join(lines), fontsize=10, va="top", family="monospace")
#             pdf.savefig(fig)
#             plt.close()


# # Export des piles
# generate_pdf(assignments, genre="H", filename="taches_hommes.pdf")
# generate_pdf(assignments, genre="F", filename="taches_femmes.pdf")


# # Analyse finale
# def format_h_m(minutes):
#     h = int(minutes) // 60
#     m = int(minutes) % 60
#     return f"{h}h{m:02d}"


# # Totaux
# total_h = sum([temps_impose[p["Participants"]] for p in men])
# total_f = sum([temps_impose[p["Participants"]] for p in women])

# # Moyennes et écart-type
# temps_hommes = [temps_impose[p["Participants"]] for p in men]
# temps_femmes = [temps_impose[p["Participants"]] for p in women]

# mean_h = np.mean(temps_hommes)
# mean_f = np.mean(temps_femmes)
# std_h = np.std(temps_hommes)
# std_f = np.std(temps_femmes)

# bas_h, haut_h = mean_h - 2 * std_h, mean_h + 2 * std_h
# bas_f, haut_f = mean_f - 2 * std_f, mean_f + 2 * std_f

# ratio = (total_h / nb_homme) / (total_f / nb_femme)

# # Console print
# print(f"Nombre de femmes : {nb_femme:.0f}")
# print(f"Nombre d'hommes : {nb_homme:.0f}")
# print(f"Ratio H/F : {(nb_homme / nb_femme):.2f}")
# print(f"Temps total homme : {format_h_m(total_h)}")
# print(f"Temps total femme : {format_h_m(total_f)}")
# print(f"Temps moyen homme pour l'ensemble du séjour : {format_h_m(total_h / nb_homme)}")
# print(f"Temps moyen femme pour l'ensemble du séjour: {format_h_m(total_f / nb_femme)}")
# print(f"✅ Ratio H/F (objectif 2.0) : {ratio:.2f}")
# print(
#     f"Écart-type temps hommes : {format_h_m(std_h)} → ~90% ont travaillé entre {format_h_m(bas_h)} et {format_h_m(haut_h)}"
# )
# print(
#     f"Écart-type temps femmes : {format_h_m(std_f)} → ~90% ont travaillé entre {format_h_m(bas_f)} et {format_h_m(haut_f)}"
# )

# # Export CSV
# rows = []
# for nom, taches in assignments.items():
#     for t in taches:
#         rows.append(
#             {
#                 "Participant": nom,
#                 "Jour": t["Jour"],
#                 "Horaire": t["Horaire"],
#                 "Tâche": t["Tâche"],
#                 "Équipe": t["Équipe"],
#                 "Durée": t["Durée"],
#             }
#         )

# assignments_df = pd.DataFrame(rows)
# assignments_df.to_csv("repartition_taches.csv", index=False)
# print("Répartition sauvegardée dans 'repartition_taches.csv'")

# # Exports supplémentaires
# # Répartition par tâche
# par_tache = (
#     assignments_df.groupby(["Jour", "Horaire", "Tâche", "Équipe"])
#     .agg({"Participant": lambda x: ", ".join(sorted(x)), "Durée": "mean"})
#     .reset_index()
# )
# par_tache.to_csv("repartition_par_tache.csv", index=False)

# # Ajout colonne genre
# assignments_df["Genre"] = assignments_df["Participant"].map(participants_df.set_index("Participants")["Genre"])

# # Créer la description complète des tâches : Jour + Horaire + Tâche
# assignments_df["Tâche_détaillée"] = assignments_df.apply(
#     lambda x: f"{x['Jour'].capitalize()} {x['Horaire']} – {x['Tâche']}", axis=1
# )

# # Regrouper par participant
# par_participant = (
#     assignments_df.groupby("Participant")
#     .agg({"Genre": "first", "Tâche_détaillée": lambda x: "\n• " + "\n• ".join(x)})
#     .reset_index()
# )

# # Remplacer genre code par label
# par_participant["Genre_label"] = par_participant["Genre"].map({"H": "Homme", "F": "Femme"})

# # Séparer hommes et femmes
# femmes_df = par_participant[par_participant["Genre"] == "F"][["Genre_label", "Tâche_détaillée"]].rename(
#     columns={"Genre_label": "Genre", "Tâche_détaillée": "Tâche"}
# )
# hommes_df = par_participant[par_participant["Genre"] == "H"][["Genre_label", "Tâche_détaillée"]].rename(
#     columns={"Genre_label": "Genre", "Tâche_détaillée": "Tâche"}
# )

# # Sauvegarde des deux fichiers
# femmes_df.to_csv("repartition_par_femme.csv", index=False)
# hommes_df.to_csv("repartition_par_homme.csv", index=False)

# print("✅ Fichier 'repartition_par_femme.csv' généré.")
# print("✅ Fichier 'repartition_par_homme.csv' généré.")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import textwrap

# Chargement des fichiers
tasks_df = pd.read_csv("taches.csv")
participants_df = pd.read_csv("participants.csv")

# Nettoyage des noms
participants_df["Participants"] = participants_df["Participants"].str.strip()

# Priorité des jours (plus tôt = plus bas)
jour_priorite = {
    "jeudi": 1,
    "vendredi": 2,
    "samedi": 3,
    "dimanche": 4,
}

# Calcul de l’ordre d’arrivée basé sur les jours cochés
participants_df["Ordre_arrivee"] = participants_df.apply(
    lambda row: sum([jour_priorite[jour] for jour in jour_priorite if row[jour] == 1]), axis=1
)

participants = participants_df.to_dict("records")


# Convertir le temps des tâches en minutes
def convert_time(time_str):
    if "h" in time_str:
        h, m = time_str.replace(" min", "").split("h")
        return float(h.strip()) * 60 + float(m.strip()) if m else float(h.strip()) * 60
    return float(time_str.replace(" min", ""))


tasks_df["Temps_minutes"] = tasks_df["Temps estimé"].apply(convert_time)

# Tri des tâches par jour et horaire
jour_order = {"jeudi": 0, "vendredi": 1, "samedi": 2, "dimanche": 3}
tasks_df["Jour_order"] = tasks_df["Jour"].str.lower().map(jour_order)

# Correction des horaires
import re


def parse_horaire(h):
    match = re.match(r"(\d{1,2})h(?:(\d{2}))?", h)
    if match:
        heure = int(match.group(1))
        minute = int(match.group(2)) if match.group(2) else 0
        return heure * 60 + minute
    return 0


tasks_df["Horaire_order"] = tasks_df["Horaire"].apply(parse_horaire)
tasks_df = tasks_df.sort_values(by=["Jour_order", "Horaire_order"]).reset_index(drop=True)

# Liste des participants
men = [p for p in participants if p["Genre"] == "H"]
women = [p for p in participants if p["Genre"] == "F"]
nb_homme = len(men)
nb_femme = len(women)

# Temps total
total_time = (tasks_df["Temps_minutes"] * tasks_df["Nbr de participants par tache"]).sum()
temps_moyen_femme = total_time / (2 * nb_homme + nb_femme)
temps_moyen_homme = 2 * temps_moyen_femme

temps_visé = {p["Participants"]: temps_moyen_homme if p["Genre"] == "H" else temps_moyen_femme for p in participants}
temps_assigné = {p["Participants"]: 0 for p in participants}

assignments = []


# Fonction pour trouver les meilleurs candidats
def get_best_fit(genre, n):
    pool = men if genre == "H" else women
    pool_sorted = sorted(
        pool, key=lambda x: (temps_assigné[x["Participants"]] / temps_visé[x["Participants"]], x["Ordre_arrivee"])
    )
    return pool_sorted[:n]


# Assignation
for _, row in tasks_df.iterrows():
    genre = row["Genre"]
    n = int(row["Nbr de participants par tache"])
    if genre == "H" or genre == "F":
        selected = get_best_fit(genre, n)
    else:
        all_sorted = sorted(
            participants,
            key=lambda x: (temps_assigné[x["Participants"]] / temps_visé[x["Participants"]], x["Ordre_arrivee"]),
        )
        selected = all_sorted[:n]

    for person in selected:
        assignments.append(
            {
                "Participant": person["Participants"],
                "Tâche": row["Tâche"],
                "Jour": row["Jour"],
                "Horaire": row["Horaire"],
            }
        )
        temps_assigné[person["Participants"]] += row["Temps_minutes"]

assignments_df = pd.DataFrame(assignments)

# Statistiques finales
from statistics import stdev


def format_h_m(mins):
    h = int(mins) // 60
    m = int(mins) % 60
    return f"{h}h{m:02d}"


total_f = sum(temps_assigné[p["Participants"]] for p in women)
total_h = sum(temps_assigné[p["Participants"]] for p in men)
ratio = (total_h / nb_homme) / (total_f / nb_femme)

std_f = stdev([temps_assigné[p["Participants"]] for p in women]) if nb_femme > 1 else 0
std_h = stdev([temps_assigné[p["Participants"]] for p in men]) if nb_homme > 1 else 0

moy_f = total_f / nb_femme
moy_h = total_h / nb_homme

bas_f, haut_f = moy_f - 2 * std_f, moy_f + 2 * std_f
bas_h, haut_h = moy_h - 2 * std_h, moy_h + 2 * std_h

print(f"Nombre de femmes : {nb_femme:.0f}")
print(f"Nombre d'hommes : {nb_homme:.0f}")
print(f"Ratio H/F : {(nb_homme / nb_femme):.2f}")
print(f"Temps total homme : {format_h_m(total_h)}")
print(f"Temps total femme : {format_h_m(total_f)}")
print(f"Temps moyen homme pour l'ensemble du séjour : {format_h_m(moy_h)}")
print(f"Temps moyen femme pour l'ensemble du séjour: {format_h_m(moy_f)}")
print(f"✅ Ratio H/F (objectif 2.0) : {ratio:.2f}")
print(
    f"Écart-type temps hommes : {format_h_m(std_h)} → ~90% ont travaillé entre {format_h_m(bas_h)} et {format_h_m(haut_h)}"
)
print(
    f"Écart-type temps femmes : {format_h_m(std_f)} → ~90% ont travaillé entre {format_h_m(bas_f)} et {format_h_m(haut_f)}"
)

# Sauvegarde CSV principal
assignments_df.to_csv("repartition_taches.csv", index=False)
print("Répartition sauvegardée dans 'repartition_taches.csv'")

# Ajout genre et tâche détaillée
assignments_df["Genre"] = assignments_df["Participant"].map(participants_df.set_index("Participants")["Genre"])
assignments_df["Tâche_détaillée"] = assignments_df.apply(
    lambda x: f"{x['Jour'].capitalize()} {x['Horaire']} – {x['Tâche']}", axis=1
)

# Par participant
par_participant = (
    assignments_df.groupby("Participant")
    .agg({"Genre": "first", "Tâche_détaillée": lambda x: "\n• " + "\n• ".join(x)})
    .reset_index()
)

par_participant["Genre_label"] = par_participant["Genre"].map({"H": "Homme", "F": "Femme"})

femmes_df = par_participant[par_participant["Genre"] == "F"][["Genre_label", "Tâche_détaillée"]].rename(
    columns={"Genre_label": "Genre", "Tâche_détaillée": "Tâche"}
)
hommes_df = par_participant[par_participant["Genre"] == "H"][["Genre_label", "Tâche_détaillée"]].rename(
    columns={"Genre_label": "Genre", "Tâche_détaillée": "Tâche"}
)

femmes_df.to_csv("repartition_par_femme.csv", index=False)
hommes_df.to_csv("repartition_par_homme.csv", index=False)
print("✅ Fichier 'repartition_par_femme.csv' généré.")
print("✅ Fichier 'repartition_par_homme.csv' généré.")
