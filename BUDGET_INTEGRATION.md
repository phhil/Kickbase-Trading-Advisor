# Kickbase Budget & Managerwerte Integration

Dieses Dokument erklärt, wie das Budget und weitere Managerwerte im Kickbase-Trading-Advisor berechnet werden. Damit kannst du die Logik einfach in eine andere App übernehmen.

## Übersicht der Berechnung

Die zentrale Funktion ist `calc_manager_budgets` in `features/budgets.py`. Sie liefert eine Tabelle mit folgenden Spalten für alle Manager einer Liga:

- **User**: Name des Managers
- **Budget**: Aktuelles verfügbares Budget
- **Team Value**: Marktwert des Teams
- **Max Negative**: Maximal erlaubtes Negativbudget
- **Available Budget**: Budget, das noch ausgegeben werden kann

## Schritte der Berechnung

1. **Aktivitäten & Boni abfragen**
   - Alle Aktivitäten (Transfers, Boni etc.) seit dem Startdatum werden geladen.
   - Login-Boni und Achievement-Boni werden separat summiert.

2. **Login-Bonus**
   - Für jeden Login gibt es einen Bonus, der aufsummiert wird:
     ```python
     total_login_bonus = sum(entry.get("data", {}).get("bn", 0) for entry in login_bonus)
     ```

3. **Achievement-Bonus**
   - Für jedes Achievement wird geprüft, wie oft es erreicht wurde und wie hoch die Belohnung ist:
     ```python
     for item in achievement_bonus:
         a_id = item.get("data", {}).get("t")
         amount, reward = get_achievement_reward(token, league_id, a_id)
         total_achievement_bonus += amount * reward
     ```
   - Falls kein Reward existiert, wird das Achievement übersprungen.

4. **Manager-Infos laden**
   - Alle Manager der Liga werden geladen, inkl. Teamwert und aktuellem Budget.

5. **Budget-Berechnung**
   - Das Startbudget wird um alle Transfers, Boni und Aktivitäten angepasst.
   - Negative Budgets und Teamwerte werden berücksichtigt.
   - Die genaue Formel kann je nach Liga-Setup variieren, im Code findest du die Details.

## Beispiel: Datenstruktur der Ergebnis-Tabelle

| User        | Budget      | Team Value   | Max Negative   | Available Budget |
|-------------|-------------|--------------|----------------|------------------|
| FullHD      | 19.825.790  | 161.297.795  | -59.770.783    | 79.596.573       |
| ...         | ...         | ...          | ...            | ...              |

## Hinweise zur Integration

- Die Funktion benötigt einen gültigen Kickbase-Token, die League-ID, das Startdatum und das Startbudget.
- Die API-Aufrufe erfolgen über Hilfsfunktionen wie `get_league_activities`, `get_managers` und `get_achievement_reward`.
- Fehlerhafte oder unvollständige Achievements werden automatisch übersprungen.
- Die Rückgabe ist ein Pandas DataFrame, das du beliebig weiterverarbeiten kannst.


## Vollständiges Beispielskript (Python)

```python
from kickbase_api.user import get_budget, get_username
from kickbase_api.league import (
   get_league_activities,
   get_league_ranking
)
from kickbase_api.manager import (
   get_managers,
   get_manager_performance,
   get_manager_info,
)
from kickbase_api.others import get_achievement_reward
import pandas as pd

def calc_manager_budgets(token, league_id, league_start_date, start_budget):
   """Calculate manager budgets based on activities, bonuses, and team performance."""

   try:
      activities, login_bonus, achievement_bonus = get_league_activities(token, league_id, league_start_date)
   except Exception as e:
      raise RuntimeError(f"Failed to fetch activities: {e}")

   activities_df = pd.DataFrame(activities)

   # Bonuses
   total_login_bonus = sum(entry.get("data", {}).get("bn", 0) for entry in login_bonus)

   total_achievement_bonus = 0
   skipped_achievements = 0
   for item in achievement_bonus:
      try:
         a_id = item.get("data", {}).get("t")
         if a_id is None:
            continue
         amount, reward = get_achievement_reward(token, league_id, a_id)
         if amount is None or reward is None:
            skipped_achievements += 1
            continue
         total_achievement_bonus += amount * reward
      except Exception:
         skipped_achievements += 1
   if skipped_achievements > 0:
      print(f"Skipped {skipped_achievements} achievement bonuses due to missing reward data.")

   # Manager performances
   try:
      managers = get_managers(token, league_id)
   except Exception as e:
      raise RuntimeError(f"Failed to fetch managers: {e}")

   performances = []
   for manager in managers:
      try:
         manager_name, manager_id = manager
         info = get_manager_info(token, league_id, manager_id)
         team_value = info.get("tv", 0)

         perf = get_manager_performance(token, league_id, manager_id, manager_name)
         perf["Team Value"] = team_value
         performances.append(perf)
      except Exception as e:
         print(f"Warning: Skipping manager {manager}: {e}")

   perf_df = pd.DataFrame(performances)
   if not perf_df.empty:
      perf_df["point_bonus"] = perf_df["tp"].fillna(0) * 1000
   else:
      perf_df["name"] = []
      perf_df["point_bonus"] = []
      perf_df["Team Value"] = []

   # Initial budgets from activities
   budgets = {user: start_budget for user in set(activities_df["byr"].dropna().unique())
                                .union(set(activities_df["slr"].dropna().unique()))}

   for _, row in activities_df.iterrows():
      byr, slr, trp = row.get("byr"), row.get("slr"), row.get("trp", 0)
      try:
         if pd.isna(byr) and pd.notna(slr):
            budgets[slr] += trp
         elif pd.isna(slr) and pd.notna(byr):
            budgets[byr] -= trp
         elif pd.notna(byr) and pd.notna(slr):
            budgets[byr] -= trp
            budgets[slr] += trp
      except KeyError as e:
         print(f"Warning: Skipping invalid activity row {row}: {e}")

   budget_df = pd.DataFrame(list(budgets.items()), columns=["User", "Budget"])

   # Merge performance bonuses
   budget_df = budget_df.merge(
      perf_df[["name", "point_bonus", "Team Value"]],
      left_on="User",
      right_on="name",
      how="left"
   ).drop(columns=["name"], errors="ignore")

   budget_df["Budget"] = budget_df["Budget"] + budget_df["point_bonus"].fillna(0)
   budget_df.drop(columns=["point_bonus"], inplace=True, errors="ignore")

   # add total login bonus equally to everyone (100% estimation, if the user logged in every day)
   budget_df["Budget"] += total_login_bonus

   # Ensure consistent float format
   budget_df["Budget"] = budget_df["Budget"].astype(float)

   # add total achievement bonus based on anchor value and current ranking (estimation approach)
   for user in budget_df["User"]:
      achievement_bonus = calc_achievement_bonus_by_points(token, league_id, user, total_achievement_bonus)
      budget_df.loc[budget_df["User"] == user, "Budget"] += achievement_bonus

   # Sync with own actual budget
   try:
      own_budget = get_budget(token, league_id)
      own_username = get_username(token)
      mask = budget_df["User"] == own_username
      if not budget_df.loc[mask, "Budget"].eq(own_budget).all():
         budget_df.loc[mask, "Budget"] = own_budget
   except Exception as e:
      print(f"Warning: Could not sync own budget: {e}")

   # TODO check if this also applies if the user has positiv budget, currently only tested with negative budget
   budget_df["Max Negative"] = (budget_df["Team Value"].fillna(0) + budget_df["Budget"]) * -0.33

   # Calculate available budget
   budget_df["Available Budget"] = (budget_df["Max Negative"].fillna(0) - budget_df["Budget"]) * -1

   # Sort by available budget ascending
   budget_df.sort_values("Available Budget", ascending=False, inplace=True, ignore_index=True)

   return budget_df

def calc_achievement_bonus_by_points(token, league_id, username, anchor_achievement_bonus):
   """Estimate achievement bonus for a user based on their total points compared to anchor user."""

   ranking = get_league_ranking(token, league_id)
   ranking_df = pd.DataFrame(ranking, columns=["Name", "Total Points"])

   # Total number of users
   num_users = len(ranking_df)
   if num_users == 0:
      return 0

   # Get anchor user's name and points
   anchor_user = get_username(token)
   anchor_row = ranking_df[ranking_df["Name"] == anchor_user]
   if anchor_row.empty:
      return 0
   anchor_points = anchor_row["Total Points"].values[0]

   # If the user is the anchor, return exactly the anchor achievement bonus
   if username == anchor_user:
      return anchor_achievement_bonus

   # Get target user's points
   user_row = ranking_df[ranking_df["Name"] == username]
   if user_row.empty:
      return 0
   user_points = user_row["Total Points"].values[0]

   # Calculate bonus scaling based on points ratio
   if anchor_points == 0:
      scale = 1.0
   else:
      scale = user_points / anchor_points

   estimated_bonus = anchor_achievement_bonus * scale
   return estimated_bonus
```

## Weiterverarbeitung

- Das Ergebnis kann als Tabelle, CSV, JSON oder direkt in eine andere App übernommen werden.
- Für die Anzeige im Terminal wird `tabulate` verwendet, für E-Mails HTML-Tabellen.

---

**Tipp:** Sieh dir die Funktion `calc_manager_budgets` in `features/budgets.py` für alle Details und Anpassungsmöglichkeiten an.
