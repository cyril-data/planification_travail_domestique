
import numpy as np
import pandas as pd

# Chargement des fichiers
tasks_df = pd.read_csv('taches.csv')
participants_df = pd.read_csv('participants.csv')

# Nettoyage des noms
participants_df['Participants'] = participants_df['Participants'].str.strip()
participants = participants_df.to_dict('records')

# Convertir le temps des tâches en minutes
def convert_time(time_str):
    if 'h' in time_str:
        h, m = time_str.replace(' min', '').split('h')
        return float(h.strip()) * 60 + float(m.strip()) if m else float(h.strip()) * 60
    return float(time_str.replace(' min', ''))

tasks_df['Temps_minutes'] = tasks_df['Temps estimé'].apply(convert_time)

# Liste des jours
jours = ['Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

# Séparer hommes et femmes
men = [p for p in participants if p['Genre'] == 'M']
women = [p for p in participants if p['Genre'] == 'F']
nb_homme = len(men)
nb_femme = len(women)

# Temps total à répartir
total_time = (tasks_df['Temps_minutes'] * tasks_df['Nbr de participants par tache']).sum()

# Objectif : H = 2 x F
temps_moyen_femme = total_time / (2 * nb_homme + nb_femme)
temps_moyen_homme = 2 * temps_moyen_femme

# Temps visé par personne
temps_visé = {
    p['Participants']: temps_moyen_homme if p['Genre'] == 'M' else temps_moyen_femme
    for p in participants
}

# Temps déjà assigné
temps_assigné = {p['Participants']: 0 for p in participants}

# Affectations finales
assignments = []

# Fonction pour filtrer personnes disponibles un jour donné
def dispo(p, jour):
    return int(p[jour]) == 1

# Traitement de chaque tâche
for _, task in tasks_df.iterrows():
    jour = task['Jour']
    needed = task['Nbr de participants par tache']
    tps = task['Temps_minutes']
    genre = task['Genre']

    # Participants disponibles ce jour
    dispo_today = [p for p in participants if dispo(p, jour)]

    # Filtrage par genre
    if genre == 'M':
        dispo_today = [p for p in dispo_today if p['Genre'] == 'M']
    elif genre == 'F':
        dispo_today = [p for p in dispo_today if p['Genre'] == 'F']

    # Score = temps déjà assigné / temps visé => les plus "légers" en premier
    dispo_sorted = sorted(
        dispo_today,
        key=lambda p: temps_assigné[p['Participants']] / temps_visé[p['Participants']]
    )

    # Sélection des N participants les moins chargés
    selected = dispo_sorted[:needed]

    # Si pas assez de monde dispo du bon genre => compléter par n’importe qui de dispo
    if len(selected) < needed:
        remaining = needed - len(selected)
        other_pool = [p for p in dispo_today if p not in selected]
        other_sorted = sorted(
            other_pool,
            key=lambda p: temps_assigné[p['Participants']] / temps_visé[p['Participants']]
        )
        selected += other_sorted[:remaining]

    # Enregistrer les assignations
    for person in selected:
        assignments.append({
            'Jour': jour,
            'Horaire': task['Horaire'],
            'Tâche': task['Tâche'],
            'Participant': person['Participants'],
            'Genre': person['Genre'],
            'Temps': tps,
            'Équipe': task['Équipe']
        })
        temps_assigné[person['Participants']] += tps

# Création du tableau final
assignments_df = pd.DataFrame(assignments)

# Analyse du résultat
total_h = sum(temps_assigné[p['Participants']] for p in men)
total_f = sum(temps_assigné[p['Participants']] for p in women)
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
temps_par_homme = [temps_assigné[p['Participants']] for p in men]
temps_par_femme = [temps_assigné[p['Participants']] for p in women]

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

print(f"Écart-type temps hommes : {format_h_m(std_h)} → ~90% ont travaillé entre {format_h_m(bas_h)} et {format_h_m(haut_h)}")
print(f"Écart-type temps femmes : {format_h_m(std_f)} → ~90% ont travaillé entre {format_h_m(bas_f)} et {format_h_m(haut_f)}")

# Export
assignments_df.to_csv('repartition_taches.csv', index=False)
print("Répartition sauvegardée dans 'repartition_taches.csv'")
