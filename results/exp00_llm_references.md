# Exp00 — Un LLM seul peut-il servir de source ?

Date : 2026-09-27
Modèle : gemini-3.8-flash (API Gemini, niveau gratuit), réflexion par défaut
Script : hallucination_test.py
Vérification : manuelle, chaque identifiant ouvert sur arxiv.org/abs/, puis recherche par titre

## Question 1 — sujet très connu (UAP très citées)

| N° | Article | Résultat | Détail |
|---|---|---|---|
| 1 | Universal adversarial perturbations | Correcte | |
| 2 | Generative Adversarial Perturbations | Correcte | |
| 3 | UAP Against Semantic Image Segmentation | Correcte | |
| 4 | Fast Feature Fool | Titre déformé | fin du titre réécrite |
| 5 | Art of Singular Vectors and UAP | Mauvais identifiant | 1711.02781 au lieu de 1709.03582 |

## Question 2 — sujet pointu (UAP dans l'espace latent du VAE de Stable Diffusion)

| N° | Article | Résultat | Détail |
|---|---|---|---|
| 1 | PhotoGuard | Correcte | |
| 2 | AdvDM | Mauvais identifiant | 2302.04478 au lieu de 2302.04578 |
| 3 | Anti-DreamBooth | Faux auteur | auteur emprunté à la référence n° 4 |
| 4 | SimAC | Mauvais identifiant + titre | 2312.12249 au lieu de 2312.07865 |
| 5 | MMA-Diffusion | Titre déformé | |

## Bilan

- 4 références exactes sur 10 (3/5 sujet connu, 1/5 sujet pointu).
- Aucun article entièrement inventé : les erreurs portent sur les détails.
- Pertinence : non vérifiée en détail (il faudrait lire les articles complets). D'après leurs
  résumés, aucun ne décrit explicitement une perturbation universelle construite dans l'espace
  latent du VAE. Deux descriptions de Gemini contredisent le résumé de l'article : SimAC
  (présenté comme universel, rien de tel dans le résumé) et MMA-Diffusion (présenté comme
  empêchant la génération d'un concept, alors que le résumé décrit une attaque pour contourner
  les filtres de sécurité).

## Limites

Une seule exécution par question : le modèle ne donne pas deux fois la même réponse.