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

# # Calcul de l’ordre d’arrivée basé sur les jours cochés
# participants_df["Ordre_arrivee"] = participants_df.apply(
#     lambda row: sum([jour_priorite[jour] for jour in jour_priorite if row[jour] == 1]), axis=1
# )

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


# Calcul du temps total à répartir
tasks_df["Charge_totale"] = tasks_df["Temps_minutes"] * tasks_df["Nbr de participants par tache"]
total_time = tasks_df["Charge_totale"].sum()

# Moyennes cibles
temps_moyen_femme = total_time / (2 * nb_homme + nb_femme)
temps_moyen_homme = 2 * temps_moyen_femme

# Temps total cible par genre
temps_total_femmes = nb_femme * temps_moyen_femme
temps_total_hommes = nb_homme * temps_moyen_homme
part_femme = temps_total_femmes / total_time
part_homme = temps_total_hommes / total_time

# Création des personnes
femmes = [{"id": f"Femme_{i+1}", "temps_total": 0, "taches": []} for i in range(nb_femme)]
hommes = [{"id": f"Homme_{i+1}", "temps_total": 0, "taches": []} for i in range(nb_homme)]

# Liste des repartitions
repartitions = []


def assigner(tache_row, groupe, nb_participants):
    for _ in range(nb_participants):
        personne = min(groupe, key=lambda x: x["temps_total"])
        personne["taches"].append(
            {
                "Jour": tache_row["Jour"],
                "Horaire": tache_row["Horaire"],
                "Tâche": tache_row["Tâche"],
                "Équipe": tache_row["Équipe"],
                "Temps": tache_row["Temps_minutes"],
            }
        )
        personne["temps_total"] += tache_row["Temps_minutes"]

        repartitions.append(
            {
                "Personne": personne["id"],
                "Genre": "F" if personne["id"].startswith("Femme") else "M",
                "Jour": tache_row["Jour"],
                "Horaire": tache_row["Horaire"],
                "Tâche": tache_row["Tâche"],
                "Équipe": tache_row["Équipe"],
                "Temps": tache_row["Temps_minutes"],
            }
        )


# Répartition des tâches
tasks_df = tasks_df.sort_values(by=["Jour_order", "Horaire_order"]).reset_index(drop=True)

for _, row in tasks_df.iterrows():
    genre_tache = row["Genre"]
    nb_participants = int(row["Nbr de participants par tache"])

    if genre_tache == "F":
        assigner(row, femmes, nb_participants)
    elif genre_tache == "M":
        assigner(row, hommes, nb_participants)
    else:
        # Tâche mixte : répartir proportionnellement aux temps cibles
        nb_f = round(part_femme * nb_participants)
        nb_h = nb_participants - nb_f
        assigner(row, femmes, nb_f)
        assigner(row, hommes, nb_h)


def format_h_min(minutes):
    heures = int(minutes) // 60
    mins = int(minutes) % 60
    return f"{heures}h{mins:02d}"


def afficher_stats(groupe, genre, cible_moyenne):
    total = sum(p["temps_total"] for p in groupe)
    moyenne = total / len(groupe)
    print(
        f"{genre}s : total = {format_h_min(total)}, moyenne = {format_h_min(moyenne)}, cible = {format_h_min(cible_moyenne)}"
    )

    temps = [p["temps_total"] for p in groupe]
    borne_basse = np.percentile(temps, 5)
    borne_haute = np.percentile(temps, 95)

    print(f"~90% des {genre}s ont travaillé entre {format_h_min(borne_basse)} et {format_h_min(borne_haute)}")


print("\n--- RÉPARTITION TOTALE ---")
afficher_stats(femmes, "Femme", temps_moyen_femme)
afficher_stats(hommes, "Homme", temps_moyen_homme)
print(f"Total général : {format_h_min(total_time)}")


# DataFrame des affectations individuelles (optionnel)
repartitions_df = pd.DataFrame(repartitions)


# 4. Afficher le résultat
def print_repartitions(personnes, genre):
    print(f"\n--- Assignation des {genre}s ---")
    for personne in personnes:
        print(f"\n{genre} {personne['id']} (total {personne['temps_total']} min):")
        for tache in personne["taches"]:
            print(
                f"  - {tache['Jour']} {tache['Horaire']} : {tache['Tâche']} ({tache['Équipe']}, {tache['Temps']} min)"
            )


# print_repartitions(femmes, "Femme")
# print_repartitions(hommes, "Homme")


import csv


def export_repartitions_csv(personnes, genre, nom_fichier):
    with open(nom_fichier, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Genre", "Tâche"])  # en-tête

        for personne in personnes:
            taches_liste = []
            for tache in personne["taches"]:
                # Format : "• Jour Horaire – Tâche"
                taches_liste.append(f"• {tache['Jour'].capitalize()} {tache['Horaire']} – {tache['Temps']} min")
            taches_concat = "\n".join(taches_liste)

            writer.writerow([genre, taches_concat])


# Exemple d’utilisation (à adapter selon tes variables femmes et hommes)
export_repartitions_csv(femmes, "Femme", "repartitions_femmes.csv")
export_repartitions_csv(hommes, "Homme", "repartitions_hommes.csv")


# # Assignation
# for _, row in tasks_df.iterrows():
#     genre = row["Genre"]
#     n = int(row["Nbr de participants par tache"])
#     if genre == "H" or genre == "F":
#         selected = get_best_fit(genre, n)
#     else:
#         all_sorted = sorted(
#             participants,
#             key=lambda x: (temps_assigné[x["Participants"]] / temps_visé[x["Participants"]], x["Ordre_arrivee"]),
#         )
#         selected = all_sorted[:n]

#     for person in selected:
#         assignments.append(
#             {
#                 "Participant": person["Participants"],
#                 "Tâche": row["Tâche"],
#                 "Jour": row["Jour"],
#                 "Horaire": row["Horaire"],
#             }
#         )
#         temps_assigné[person["Participants"]] += row["Temps_minutes"]

# assignments_df = pd.DataFrame(assignments)

# # Statistiques finales
# from statistics import stdev


# def format_h_m(mins):
#     h = int(mins) // 60
#     m = int(mins) % 60
#     return f"{h}h{m:02d}"


# total_f = sum(temps_assigné[p["Participants"]] for p in women)
# total_h = sum(temps_assigné[p["Participants"]] for p in men)
# ratio = (total_h / nb_homme) / (total_f / nb_femme)

# std_f = stdev([temps_assigné[p["Participants"]] for p in women]) if nb_femme > 1 else 0
# std_h = stdev([temps_assigné[p["Participants"]] for p in men]) if nb_homme > 1 else 0

# moy_f = total_f / nb_femme
# moy_h = total_h / nb_homme

# bas_f, haut_f = moy_f - 2 * std_f, moy_f + 2 * std_f
# bas_h, haut_h = moy_h - 2 * std_h, moy_h + 2 * std_h

# print(f"Nombre de femmes : {nb_femme:.0f}")
# print(f"Nombre d'hommes : {nb_homme:.0f}")
# print(f"Ratio H/F : {(nb_homme / nb_femme):.2f}")
# print(f"Temps total homme : {format_h_m(total_h)}")
# print(f"Temps total femme : {format_h_m(total_f)}")
# print(f"Temps moyen homme pour l'ensemble du séjour : {format_h_m(moy_h)}")
# print(f"Temps moyen femme pour l'ensemble du séjour: {format_h_m(moy_f)}")
# print(f"✅ Ratio H/F (objectif 2.0) : {ratio:.2f}")
# print(
#     f"Écart-type temps hommes : {format_h_m(std_h)} → ~90% ont travaillé entre {format_h_m(bas_h)} et {format_h_m(haut_h)}"
# )
# print(
#     f"Écart-type temps femmes : {format_h_m(std_f)} → ~90% ont travaillé entre {format_h_m(bas_f)} et {format_h_m(haut_f)}"
# )

# # Sauvegarde CSV principal
# assignments_df.to_csv("repartition_taches.csv", index=False)
# print("Répartition sauvegardée dans 'repartition_taches.csv'")

# # Ajout genre et tâche détaillée
# assignments_df["Genre"] = assignments_df["Participant"].map(participants_df.set_index("Participants")["Genre"])
# assignments_df["Tâche_détaillée"] = assignments_df.apply(
#     lambda x: f"{x['Jour'].capitalize()} {x['Horaire']} – {x['Tâche']}", axis=1
# )

# # Par participant
# par_participant = (
#     assignments_df.groupby("Participant")
#     .agg({"Genre": "first", "Tâche_détaillée": lambda x: "\n• " + "\n• ".join(x)})
#     .reset_index()
# )

# par_participant["Genre_label"] = par_participant["Genre"].map({"H": "Homme", "F": "Femme"})

# femmes_df = par_participant[par_participant["Genre"] == "F"][["Genre_label", "Tâche_détaillée"]].rename(
#     columns={"Genre_label": "Genre", "Tâche_détaillée": "Tâche"}
# )
# hommes_df = par_participant[par_participant["Genre"] == "H"][["Genre_label", "Tâche_détaillée"]].rename(
#     columns={"Genre_label": "Genre", "Tâche_détaillée": "Tâche"}
# )

# femmes_df.to_csv("repartition_par_femme.csv", index=False)
# hommes_df.to_csv("repartition_par_homme.csv", index=False)
# print("✅ Fichier 'repartition_par_femme.csv' généré.")
# print("✅ Fichier 'repartition_par_homme.csv' généré.")
