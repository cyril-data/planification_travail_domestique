# Instalation

- Instalation python
- Instalation lib `pip install textwrap matplotlib pandas numpy`

# Utilisation
- remplir les fichiers : **taches.csv** et **participants.csv**
- lancer le code : `python tache_domestiques.py`


La répartion est aléatoire avec un critère politique : Le travail domestique des hommes est en moyenne 2 x plus important que celui des femmes. 
=> expérience pour renverser la norme patriarcale de notre société où le travail domestique est assurée au 2/3 par les femmes (=> 2 x plus que les hommes).  

Lignes : 
```
# Objectif : H = 2 x F
temps_moyen_femme = total_time / (2 * nb_homme + nb_femme)
temps_moyen_homme = 2 * temps_moyen_femme
```

# Résultats


```
Nombre de femmes : 13
Nombre d'hommes : 37
Ratio H/F : 2.85
Temps total homme : 81h00
Temps total femme : 17h00
Temps moyen homme pour l'ensemble du séjour : 2h11
Temps moyen femme pour l'ensemble du séjour: 1h18
✅ Ratio H/F (objectif 2.0) : 1.67
Écart-type temps hommes : 0h24 → ~90% ont travaillé entre 1h32 et 2h50
Écart-type temps femmes : 0h16 → ~90% ont travaillé entre 0h53 et 1h44
Répartition sauvegardée dans 'repartition_taches.csv'
✅ Fichier 'repartition_par_tache.csv' généré avec les participants regroupés par tâche.
✅ Fichier 'repartition_par_participant.csv' généré avec les participants en ligne et les tâches en colonnes.
✅ PDF multi-pages par jour généré proprement avec cases grisées et blanches.
```

Résultats pdf :  repartition_par_participant_par_jour.pdf 
Sour format numérique : *repartition_par_participant.csv* *repartition_par_tache.csv* *repartition_taches.csv*


