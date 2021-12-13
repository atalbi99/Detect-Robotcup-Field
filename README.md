# Detect-Robotcup-Field
Ce projet est développé dans le cadre de la Robocup, l'objectif est de créer un algorithme de traitement d'images< capable de détecter et reconstruire le terrain de football à partir d'une séquence d'images connues.
Nous avons réalisé le traitement sur la totalité des images des logs.
## Technologies
* Python 3
* OpenCV 4
# Utilisation 
Le fichier qu'il faut executé est le "final.py". Le code est configuré de tel sorte qu'on a en sortie le résultat de quelques images du log1 dont les numéros sont stockés dans la liste L1. On affiche aussi l'indice d'évaluation du résultat.
Pour ester le résultat sur les images du log2 ou log3 : 

```
Remplacez L1 (ligne 94) par L2 pour le log2 et L3 pour le log3.
Remplacez 'log1' des lignes 146 et 147 par 'log2' ou 'log3'.
```
Pour tester le résultat sur les images du log4: 

```
Remplacez L1 (ligne 94) par L4.
Remplacez 0:03d (ligne 96) par 0:02d.
Remplacez log1 des lignes 146 et 147 par log4.
```
