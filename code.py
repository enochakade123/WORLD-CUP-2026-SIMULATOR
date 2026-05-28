"""
2026 FIFA World Cup Predictor
=============================
A Monte Carlo simulator that plays the full 48-team tournament (12 groups,
32-team knockout, final, and 3rd-place playoff) thousands of times to estimate
each nation's championship odds. It can also print a single tournament's full
set of scorelines from the group stage to the bronze-medal match.

Pure Python standard library. No third-party dependencies.
"""

import math
import random
from collections import defaultdict

# ==========================================================================
# 1. PLAYER AND TEAM BLUEPRINTS
# ==========================================================================
class Player:
    """A single player's stat card. Impact = consistency * level * weight."""
    def __init__(self, name, position, c_i, L_c, w_i):
        self.name = name
        self.position = position
        self.c_i = c_i      # raw consistency / base rating
        self.L_c = L_c      # level coefficient (quality of league/role)
        self.w_i = w_i      # importance weight

    def calculate_impact_score(self):
        return self.c_i * self.L_c * self.w_i


class Team:
    """
    A nation. Strength is split into:
      attack_strength() = S * E * Y   (star power x experience x form/youth)
      defense_strength() = D_f        (solidity; divides the opponent's xG)
    Momentum (M) tracks recent form and is updated after every match.
    Chaos variance (C_v) controls how unpredictable the team is.
    """
    def __init__(self, name, roster, C_v=0.15, E=1.0, D_f=1.0, Y=1.0):
        self.name = name
        self.roster = roster
        self.S = self.calculate_team_star_factor()
        self.M = 1.0
        self.C_v = C_v
        self.E = E
        self.D_f = D_f
        self.Y = Y

    def calculate_team_star_factor(self):
        if not self.roster:
            return 1.0
        return sum(p.calculate_impact_score() for p in self.roster) / len(self.roster)

    def attack_strength(self):
        return self.S * self.E * self.Y

    def defense_strength(self):
        return self.D_f

    def update_momentum(self, goal_differential, alpha=0.02):
        """Form nudges xG up or down based on the last result. Floored at 0.5."""
        self.M = max(0.5, 1.0 + alpha * goal_differential)


# ==========================================================================
# 2. MATCH ENGINE
# ==========================================================================
BASE_XG = 1.3  # neutral-venue baseline expected goals, identical for both sides


def apply_chaos_variance(adjusted_xg, C_v):
    """Random match-day swing of up to +/- C_v around the expected goals."""
    R = random.uniform(-1.0, 1.0)
    return max(0.0, adjusted_xg * (1 + R * C_v))


def poisson_probability(lmbda, k):
    """P(exactly k goals) given an expected-goals rate of lambda."""
    return ((lmbda ** k) * math.exp(-lmbda)) / math.factorial(k)


def sample_goals(xg, max_goals=9):
    """Draw a goal count from a Poisson distribution truncated at max_goals."""
    weights = [poisson_probability(xg, k) for k in range(max_goals + 1)]
    return random.choices(range(max_goals + 1), weights=weights)[0]


def simulate_match(team_a, team_b, is_knockout=False):
    """
    Each side's expected goals = its attack divided by the opponent's defense,
    scaled by form and a neutral baseline, then jittered by chaos variance.
    Knockout ties are broken by a coin flip (extra time / penalties).
    """
    xg_a = BASE_XG * team_a.M * team_a.attack_strength() / team_b.defense_strength()
    xg_b = BASE_XG * team_b.M * team_b.attack_strength() / team_a.defense_strength()

    goals_a = sample_goals(apply_chaos_variance(xg_a, team_a.C_v))
    goals_b = sample_goals(apply_chaos_variance(xg_b, team_b.C_v))

    if is_knockout and goals_a == goals_b:
        if random.random() < 0.5:
            goals_a += 1
        else:
            goals_b += 1

    return goals_a, goals_b


# ==========================================================================
# 3. GROUP STAGE
# ==========================================================================
def simulate_group_stage(all_teams):
    """Shuffle 48 teams into 12 groups, play round-robins, return ranked groups."""
    teams = list(all_teams)          # copy so we never mutate the master list
    random.shuffle(teams)
    groups = {f"Group {chr(65 + i)}": teams[i * 4:(i + 1) * 4] for i in range(12)}

    standings = {t.name: {"Points": 0, "GD": 0, "GF": 0, "TeamObject": t} for t in teams}

    for group in groups.values():
        matchups = [
            (group[0], group[1]), (group[0], group[2]), (group[0], group[3]),
            (group[1], group[2]), (group[1], group[3]), (group[2], group[3]),
        ]
        for a, b in matchups:
            ga, gb = simulate_match(a, b, is_knockout=False)
            standings[a.name]["GF"] += ga
            standings[b.name]["GF"] += gb
            standings[a.name]["GD"] += ga - gb
            standings[b.name]["GD"] += gb - ga
            if ga > gb:
                standings[a.name]["Points"] += 3
            elif gb > ga:
                standings[b.name]["Points"] += 3
            else:
                standings[a.name]["Points"] += 1
                standings[b.name]["Points"] += 1
            a.update_momentum(ga - gb)
            b.update_momentum(gb - ga)

    final_standings = {}
    for name, group in groups.items():
        final_standings[name] = sorted(
            group,
            key=lambda t: (standings[t.name]["Points"], standings[t.name]["GD"], standings[t.name]["GF"]),
            reverse=True,
        )
    return final_standings, standings


def get_advancing_teams(final_group_standings, standings):
    """Top 2 of each group (24) + the 8 best third-placed teams = 32 (FIFA 2026)."""
    advancing, thirds = [], []
    for ranked in final_group_standings.values():
        advancing.extend([ranked[0], ranked[1]])
        thirds.append(ranked[2])
    thirds.sort(
        key=lambda t: (standings[t.name]["Points"], standings[t.name]["GD"], standings[t.name]["GF"]),
        reverse=True,
    )
    advancing.extend(thirds[:8])
    return advancing


# ==========================================================================
# 4. KNOCKOUT BRACKET (with 3rd-place playoff)
# ==========================================================================
def simulate_knockout_bracket(teams_in_bracket, verbose=False):
    """
    Single-elimination reduction. Captures the two semifinal losers and the
    final's runner-up, then plays the bronze-medal match.
    Returns (champion, runner_up, third, fourth).
    """
    round_names = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
                   4: "Semi-finals", 2: "Final"}
    current = list(teams_in_bracket)
    sf_losers, runner_up = [], None

    while len(current) > 1:
        nxt, size = [], len(current)
        if verbose:
            print(f"\n--- {round_names.get(size, f'{size} teams')} ---")
        for i in range(0, size, 2):
            a, b = current[i], current[i + 1]
            ga, gb = simulate_match(a, b, is_knockout=True)
            winner, loser = (a, b) if ga >= gb else (b, a)
            winner.update_momentum(abs(ga - gb))
            nxt.append(winner)
            if verbose:
                print(f"{a.name:>12} {ga} - {gb} {b.name:<12} -> {winner.name}")
            if size == 4:
                sf_losers.append(loser)
            elif size == 2:
                runner_up = loser
        current = nxt

    champion = current[0]
    third = fourth = None
    if len(sf_losers) == 2:
        ga, gb = simulate_match(sf_losers[0], sf_losers[1], is_knockout=True)
        third, fourth = (sf_losers[0], sf_losers[1]) if ga >= gb else (sf_losers[1], sf_losers[0])
        if verbose:
            print(f"\n--- 3rd-Place Playoff ---")
            print(f"{sf_losers[0].name:>12} {ga} - {gb} {sf_losers[1].name:<12} -> {third.name}")
            print(f"\nCHAMPION: {champion.name} | Runner-up: {runner_up.name} "
                  f"| 3rd: {third.name} | 4th: {fourth.name}")

    return champion, runner_up, third, fourth


# ==========================================================================
# 5. SIMULATION DRIVERS
# ==========================================================================
def simulate_one_tournament(teams, verbose=True):
    """Play a single full tournament and (optionally) print every scoreline."""
    for t in teams:
        t.M = 1.0
    group_standings, raw = simulate_group_stage(teams)

    if verbose:
        print("=== GROUP STAGE FINAL STANDINGS ===")
        for name, ranked in group_standings.items():
            line = ", ".join(
                f"{t.name}({raw[t.name]['Points']}pts)" for t in ranked
            )
            print(f"{name}: {line}")

    knockout = get_advancing_teams(group_standings, raw)
    return simulate_knockout_bracket(knockout, verbose=verbose)


def run_pre_tournament_oracle(teams, num_simulations=2000):
    """Monte Carlo: replay the tournament many times and report championship odds."""
    champions, finals = defaultdict(int), defaultdict(int)

    for _ in range(num_simulations):
        for t in teams:
            t.M = 1.0  # reset form each tournament so runs stay independent
        group_standings, raw = simulate_group_stage(teams)
        knockout = get_advancing_teams(group_standings, raw)
        champ, runner, _third, _fourth = simulate_knockout_bracket(knockout)
        champions[champ.name] += 1
        finals[champ.name] += 1
        finals[runner.name] += 1

    print(f"\n=== 2026 WORLD CUP ORACLE ({num_simulations} simulations) ===")
    print(f"{'Team':<14}{'Win %':>8}{'Reach Final %':>16}")
    for team, wins in sorted(champions.items(), key=lambda x: x[1], reverse=True):
        print(f"{team:<14}{wins / num_simulations * 100:7.2f}%{finals[team] / num_simulations * 100:15.2f}%")


# ==========================================================================
# 6. THE 48 TEAMS
# ==========================================================================
def generate_team(name, tier):
    cfg = {
        "Elite":    dict(c_i=1.00, L_c=1.00, w_i=1.05, C_v=0.08, E=1.05, D_f=1.10, Y=1.05),
        "Strong":   dict(c_i=1.00, L_c=1.00, w_i=1.00, C_v=0.15, E=1.00, D_f=1.00, Y=1.00),
        "Average":  dict(c_i=1.00, L_c=0.85, w_i=1.00, C_v=0.20, E=0.95, D_f=0.90, Y=1.00),
        "Wildcard": dict(c_i=1.00, L_c=0.70, w_i=0.95, C_v=0.30, E=0.85, D_f=0.80, Y=0.90),
    }[tier]
    roster = [Player(f"{name} Star", pos, cfg["c_i"], cfg["L_c"], cfg["w_i"])
              for pos in ["Attacker", "Midfielder", "Defender", "Goalkeeper", "Any"]]
    return Team(name, roster, C_v=cfg["C_v"], E=cfg["E"], D_f=cfg["D_f"], Y=cfg["Y"])


def build_all_teams():
    custom = [
        Team("France",    [Player("Star", "Any", 1.05, 1.0, 1.10) for _ in range(5)], C_v=0.05, E=1.10, D_f=1.15, Y=1.05),
        Team("Argentina", [Player("Star", "Any", 1.00, 1.0, 1.10) for _ in range(5)], C_v=0.10, E=1.05, D_f=1.05, Y=1.05),
        Team("Spain",     [Player("Star", "Any", 1.00, 1.0, 1.05) for _ in range(5)], C_v=0.05, E=1.08, D_f=1.05, Y=1.15),
        Team("USA",       [Player("Star", "Any", 1.00, 1.0, 1.00) for _ in range(5)], C_v=0.20, E=1.15, D_f=0.95, Y=1.00),
        Team("Mexico",    [Player("Star", "Any", 1.00, 1.0, 1.00) for _ in range(5)], C_v=0.20, E=1.10, D_f=0.95, Y=0.95),
        Team("Canada",    [Player("Star", "Any", 1.00, 1.0, 1.00) for _ in range(5)], C_v=0.25, E=1.10, D_f=0.85, Y=1.00),
    ]
    elite = [generate_team(n, "Elite") for n in
             ["Brazil", "England", "Germany", "Portugal", "Netherlands"]]
    strong = [generate_team(n, "Strong") for n in
              ["Uruguay", "Croatia", "Belgium", "Switzerland", "Colombia",
               "Sweden", "Turkiye", "Japan", "Morocco", "Senegal"]]
    average = [generate_team(n, "Average") for n in
               ["Ecuador", "Paraguay", "Austria", "Czechia", "Norway",
                "Scotland", "Australia", "Iran", "South Korea", "Algeria",
                "Cote d'Ivoire", "Egypt", "Ghana", "Tunisia"]]
    wildcard = [generate_team(n, "Wildcard") for n in
                ["Saudi Arabia", "Qatar", "Panama", "New Zealand", "DR Congo",
                 "Cape Verde", "Uzbekistan", "Iraq", "Jordan", "Curacao",
                 "Haiti", "South Africa", "Bosnia and Herzegovina"]]
    return custom + elite + strong + average + wildcard


# ==========================================================================
# 7. ENTRY POINT
# ==========================================================================
if __name__ == "__main__":
    teams = build_all_teams()
    assert len(teams) == 48, f"Expected 48 teams, got {len(teams)}"

    # A) One full tournament with every scoreline printed:
    simulate_one_tournament(teams, verbose=True)

    # B) The probability engine:
    run_pre_tournament_oracle(teams, num_simulations=2000)
