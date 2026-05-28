# WORLD-CUP-2026-SIMULATOR
# 2026 FIFA World Cup Predictor

A Monte Carlo simulator that plays the entire 48-team FIFA 2026 tournament,
group stage through the final and 3rd-place playoff, thousands of times to
estimate each nation's odds of winning. It can also replay a single tournament
and print every scoreline along the way.

Pure Python standard library. No third-party dependencies.

## Features

- Full FIFA 2026 format: 48 teams, 12 groups of 4, top 2 plus the 8 best
  third-placed teams advance to a 32-team knockout.
- Opponent-aware scoring: each side's expected goals are its attack divided by
  the opponent's defense, so strong defenses actually suppress opponents.
- Poisson goal model with a tunable chaos-variance factor per team.
- Momentum that updates after every match to reflect recent form.
- A 3rd-place playoff between the two semifinal losers.
- Two run modes: a verbose single tournament (every scoreline) and a Monte
  Carlo oracle (championship and final-appearance probabilities).

## Requirements

- Python 3.8 or newer. Nothing else to install.

## Usage

```bash
python world_cup_2026_simulator.py
```

This prints one full tournament with all scorelines, then the probability table
from 2000 simulations.

### Use it in your own code

```python
from world_cup_2026_simulator import build_all_teams, run_pre_tournament_oracle, simulate_one_tournament

teams = build_all_teams()

# One tournament, every scoreline printed:
simulate_one_tournament(teams, verbose=True)

# Probabilities from many simulations:
run_pre_tournament_oracle(teams, num_simulations=5000)
```

## How the model works

Each team carries a small set of multipliers:

| Symbol | Meaning |
|--------|---------|
| `S`   | Star factor, the average impact score of the roster |
| `E`   | Experience multiplier |
| `Y`   | Form / youth multiplier |
| `D_f` | Defensive solidity (divides the opponent's expected goals) |
| `C_v` | Chaos variance, how unpredictable the team is |
| `M`   | Momentum, updated after every match |

A team's expected goals in a match are:

```
xG = BASE_XG * momentum * (attack of this team) / (defense of opponent)
attack = S * E * Y
```

That number is jittered by chaos variance and then fed into a Poisson draw to
produce an actual scoreline. Knockout ties are resolved by a coin flip standing
in for extra time and penalties.

## Tuning

- `num_simulations` in `run_pre_tournament_oracle` trades runtime for precision.
  Monte Carlo error shrinks on the order of 1 / sqrt(N).
- `BASE_XG` sets the global scoring level.
- Edit `build_all_teams` to change rosters, tiers, or the per-team multipliers.

## Notes and possible extensions

- The knockout bracket pairs advancing teams in list order rather than using
  realistic seeding. Adding a seeded draw would improve realism.
- For large simulation counts, swapping the manual Poisson draw for
  `numpy.random.poisson` would speed things up.
- The simulations are independent and share no mutable state once momentum is
  reset, so the oracle parallelizes cleanly across processes.

## License

MIT. See [LICENSE](LICENSE).
