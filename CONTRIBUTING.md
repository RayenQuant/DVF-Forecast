# Contribuer au projet

Merci de votre intérêt pour ce projet ! Voici quelques règles pour proposer des contributions.

## Workflow

1. **Fork** le dépôt et créez une branche depuis `develop` :
   ```bash
   git checkout -b feat/ma-nouvelle-feature
   ```

2. **Installez** les dépendances de développement :
   ```bash
   pip install -r requirements.txt
   ```

3. **Codez** votre modification en respectant les conventions ci-dessous.

4. **Testez** localement :
   ```bash
   make lint
   make test
   ```

5. **Ouvrez une Pull Request** vers la branche `develop` en décrivant clairement
   le changement et son intérêt.

## Conventions de code

- **Formatage** : `black` (longueur de ligne : 100)
- **Linting** : `ruff`
- **Tests** : `pytest`, au moins un test par nouvelle fonctionnalité
- **Docstrings** : style Google ou NumPy, en français
- **Commits** : convention [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat:` pour une nouvelle fonctionnalité
  - `fix:` pour une correction
  - `refactor:` pour une refonte sans changement de comportement
  - `docs:` pour la documentation
  - `test:` pour les tests
  - `chore:` pour la maintenance (CI, dépendances…)

## Structure des Pull Requests

- Une PR = un sujet. Évitez les PR fourre-tout.
- Liez une issue si possible (`Closes #123`).
- Vérifiez que la CI passe avant de demander la revue.

## Signaler un bug

Ouvrez une issue avec :
- la version de Python utilisée
- la commande qui échoue
- le message d'erreur complet
- les étapes pour reproduire

Merci 🙏
