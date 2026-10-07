# NFL unit board (automated)

* `nfl_spread_pipeline.py` builds the weekly unit-rating board and team ratings (saved in `boards/`).
* `grade_boards.py` grades saved pregame boards against results (`forward_test_*.csv`).
* `.github/workflows/weekly.yml` runs both on a schedule (Wed + Fri boards, Tue grading).

Setup: put `game_lines.csv` (from the Odds API pull) in `data/`, add an optional `ODDS_API_KEY` repo secret,
then run the workflow once by hand from the Actions tab.
