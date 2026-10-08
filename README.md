# NFL unit board (automated)

* `nfl_spread_pipeline.py` builds the weekly unit-rating board and team ratings (saved in `boards/`).
* `grade_boards.py` grades saved pregame boards against results (`forward_test_*.csv`).
* `.github/workflows/weekly.yml` runs both on a schedule (Wed + Fri boards, Tue grading).

Setup: put `game_lines.csv` (from the Odds API pull) in `data/`, add an optional `ODDS_API_KEY` repo secret,
then run the workflow once by hand from the Actions tab.

## Rating experiments (opponent adjustment, recency decay, extra EPA stats)

Run the workflow by hand with **experiments** ticked (Actions > nfl-weekly-board > Run workflow). After the usual
out-of-sample check it prints a table of variants on the same 2023-25 games, each with its change in average miss
versus the current model (+ = better) and a 95% interval. Adopt a variant only if it says "clear gain": add repository
variables (Settings > Secrets and variables > Actions > Variables) `RATING_DECAY` (e.g. 0.9), `OPP_ADJ` (1) and/or
`EXTRA_FEATS` (1). With no variables set the model is unchanged.
