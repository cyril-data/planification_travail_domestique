# Instalation

- Instalation python
- Instalation lib `pip install textwrap matplotlib pandas numpy`

## Sous linux : 
Taper la commande dans un terminal 
`pip install textwrap matplotlib pandas numpy`


## Sous windows : 


Ouvrir l'Explorateur de fichiers et naviguer jusqu'au dossier souhaité.


Maintenir la touche Shift enfoncée et faites un clic droit sur le dossier (ou dans un espace vide à l'intérieur du dossier).

Dans le menu contextuel, sélectionnez :

```Ouvrir une fenêtre PowerShell ici (Windows 10/11 par défaut)```

ou 

```Ouvrir l'invite de commandes ici (si vous avez personnalisé votre menu).```

Il est aussi possible d'Appuyer sur `Win + R`, tapez `cmd` ou `powershell`, puis validez avec `Entrée`. Pour une meilleure expérience, utiliser PowerShell ou Windows Terminal (disponible via le Microsoft Store).

Dans le shell qui est ouvert (Invite de commandes ou PowerShell) Vérifier que Python est installé. Dans le shell, taper :

```bash
python --version
```
ou 

```bash
python3 --version
```

Vérifier que pip est disponible:

```bash
pip --version
```
ou 
```bash
python -m pip --version
```

Installer les bibliothèques avec pip

```bash
pip install textwrap matplotlib pandas numpy
```


## Installer Git pour Windows

Si Git n'est pas déjà installé :

Téléchargez Git for Windows depuis git-scm.com. Lancez l'installateur et suivez les étapes (laissez les options par défaut, sauf si vous avez des besoins spécifiques).

Cochez l'option pour ajouter Git au PATH (important pour utiliser Git depuis n'importe quel terminal). Terminez l'installation.

# Utilisation

Dans le shell récupérer les sources du repository avec git clone : 

```bash
git clone https://github.com/cyril-data/planification_travail_domestique
```

Un dossier `planification_travail_domestique` est créé. Entrer dedans avec l'Explorateur.  

Dans le shell/terminal rentrer dans le dossier : 

```bash
cd planification_travail_domestique
```


- remplir les fichiers : **taches.csv** et **participants.csv** avec un tableur et sauvegarder les .csv
- lancer le code python dans le terminal : 
```bash
python tache_domestiques.py
``` 


La répartion est aléatoire avec une pondération de nature politique : Le travail domestique des hommes est fixé en moyenne 2 x plus important que celui des femmes. 
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

# Mise en forme
Ensuite, modifier les fichier .odt et inserer un champs de publipostage avec les bases de données repartitions_femmes.csv et repartitions_hommes.csv qui ont été créé/modifier par le programme python. 