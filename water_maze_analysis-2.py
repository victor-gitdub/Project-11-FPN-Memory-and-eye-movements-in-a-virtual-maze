"""
============================================================
 Virtual Morris Water Maze - Good vs. Bad Learner Classifier
============================================================

WHAT THIS SCRIPT DOES
----------------------
You export one CSV file per participant per session from the eye
tracking / maze software (e.g. "PID16_SESSION1.csv", "PID16_SESSION2.csv").
Each row in those files is one trial, and includes a column called
"distance_to_target" - this is how far away the participant's pin
(their guess for the hidden platform) was from the platform's real
location.

This script:

  1. Reads in as many of those CSV files as you give it.
  2. Figures out, from the file name, which participant and which
     session each file belongs to.
  3. For each participant, looks at SESSION 1 and compares the mean
     time_taken of the first 2 trials vs. the last 2 trials within
     each maze. This gives a percentage improvement (speed) score,
     which is then averaged across all 3 mazes.
  4. Labels each participant based purely on their own improvement:
        - got faster (> 0%)  ->  "Good Learner"
        - no improvement     ->  "Bad Learner"
  5. Saves two clean CSV files so that later parts of your code
     (e.g. the eye-tracking correlation analysis) can just load the
     results, instead of re-reading and reprocessing every raw file
     again.

HOW TO USE THIS SCRIPT (read this part!)
------------------------------------------
Scroll all the way down to the bottom, to the section titled
"RUN THE ANALYSIS". That is the only part you need to edit:

  1. Put the path to every CSV file you have into the `csv_files`
     list (one participant can appear more than once, e.g. once per
     session - that is expected and handled automatically).
  2. Run the script (in a terminal: `python water_maze_analysis.py`,
     or press the "Run" button in your code editor).
  3. Check the "processed_data" folder that appears - your results
     will be in there.

You should NOT need to edit anything above the "RUN THE ANALYSIS"
section. That part is just the toolbox the script uses internally.

FILE NAMING RULE
-----------------
Each file name must contain "_SESSION" followed by a number, e.g.:
    PID16_SESSION1.csv      ParticipantID = PID16,  Session = 1
    PID16_SESSION2.csv      ParticipantID = PID16,  Session = 2
    PID23_SESSION1.csv      ParticipantID = PID23,  Session = 1
Whatever comes before "_SESSION" is used as the participant's ID, so
your existing file names already work without any changes.
"""

import re
from pathlib import Path

import pandas as pd


# These columns must exist in every raw CSV file for the analysis to
# work. If a file is missing one of these, the script will stop and
# tell you exactly which one, instead of quietly producing wrong
# numbers later on.
REQUIRED_COLUMNS = [
    "Maze_ID",
    "endpoint_X",
    "endpoint_Y",
    "target_X",
    "target_Y",
    "distance_to_target",
]


class Participant:
    """
    Holds all the trial data for ONE participant, across however many
    session files we have loaded for them.

    You will not normally create one of these by hand - the
    WaterMazeStudy class below does that automatically as it loads
    your files.
    """

    def __init__(self, participant_id):
        self.participant_id = participant_id
        # Dictionary: {session_number (int): trial-level dataframe for that session}
        self.sessions = {}

    def add_session(self, session_number, trial_data):
        """Stores one session's trial-level dataframe for this participant."""
        self.sessions[session_number] = trial_data

    def final_session_number(self):
        """The highest session number we have on file for this participant."""
        return max(self.sessions.keys())

    def final_session_data(self):
        """The trial-level dataframe for this participant's final session."""
        return self.sessions[self.final_session_number()]

    def percentage_improvement_session_1(self):
        """
        Calculates how much faster (in %) a participant got across
        SESSION 1, by comparing their first 2 vs. last 2 trials within
        each maze separately, then averaging across all mazes.

        For each maze:
          - first_mean: average time_taken of the 1st and 2nd trial
            in that maze (in the order they happened).
          - last_mean:  average time_taken of the 2nd-to-last and last
            trial in that maze.
          - improvement = (first_mean - last_mean) / first_mean * 100
            Positive  -> they got FASTER (improved).
            Negative  -> they got SLOWER (got worse).
            Zero      -> no change at all.

        The final score returned is the average of this percentage
        across all 3 mazes. This is the number used to classify each
        participant as a Good or Bad Learner.
        """
        if 1 not in self.sessions:
            raise ValueError(
                f"Participant {self.participant_id} has no Session 1 data loaded. "
                f"Sessions available: {sorted(self.sessions.keys())}"
            )
        session_data = self.sessions[1]
        maze_improvements = []

        for maze_id, maze_trials in session_data.groupby("Maze_ID"):
            # Preserve the original (chronological) row order
            maze_trials = maze_trials.reset_index(drop=True)

            if len(maze_trials) < 4:
                raise ValueError(
                    f"Participant {self.participant_id}, Maze {maze_id} in Session 1 "
                    f"only has {len(maze_trials)} trial(s). At least 4 are needed "
                    f"to compare first-2 vs. last-2 trials."
                )

            first_two_mean = maze_trials["time_taken"].iloc[:2].mean()
            last_two_mean  = maze_trials["time_taken"].iloc[-2:].mean()

            # Positive value = time shrank = participant got faster = improved
            improvement_pct = (first_two_mean - last_two_mean) / first_two_mean * 100
            maze_improvements.append(improvement_pct)

        # Average the per-maze improvement into one score per participant
        return sum(maze_improvements) / len(maze_improvements)

    def all_trials_combined(self):
        """
        Returns ONE dataframe with every trial from EVERY session this
        participant has, tagged with which participant/session each row
        belongs to. Useful later (e.g. for eye-tracking analyses) when
        you need all trials, not just the final session.
        """
        tagged_sessions = []
        final_session = self.final_session_number()

        for session_number, trial_data in self.sessions.items():
            trial_data = trial_data.copy()
            trial_data.insert(0, "Participant_ID", self.participant_id)
            trial_data.insert(1, "Session", session_number)
            trial_data.insert(2, "Used_For_Classification", session_number == 1)
            tagged_sessions.append(trial_data)

        return pd.concat(tagged_sessions, ignore_index=True)


class WaterMazeStudy:
    """
    Manages every participant in your study. This is the class you
    actually interact with: hand it your CSV files, ask it to classify
    learners, and ask it to save the results.
    """

    # Matches file names like "PID16_SESSION1.csv".
    # Group 1 = everything before "_SESSION" (the participant ID).
    # Group 2 = the digits after "SESSION" (the session number).
    FILENAME_PATTERN = re.compile(r"(.+)_SESSION(\d+)", re.IGNORECASE)

    def __init__(self):
        # Dictionary: {participant_id (str): Participant object}
        self.participants = {}
        self.summary = None  # filled in by classify_learners()

    # ---------------------------------------------------------- loading

    def load_csv_files(self, filepaths):
        """Loads a list of CSV file paths, one call handles any number of files."""
        for filepath in filepaths:
            self._load_single_file(filepath)
        return self

    def load_folder(self, folder_path, pattern="*.csv"):
        """
        Convenience option: instead of listing every file by hand,
        point this at a folder and it loads every matching CSV inside.
        Example: study.load_folder("data/")
        """
        folder = Path(folder_path)
        filepaths = sorted(folder.glob(pattern))
        if not filepaths:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {folder_path}")
        return self.load_csv_files(filepaths)

    def _load_single_file(self, filepath):
        filepath = Path(filepath)
        participant_id, session_number = self._parse_filename(filepath.name)

        trial_data = pd.read_csv(filepath)
        self._validate_columns(trial_data, filepath.name)

        if participant_id not in self.participants:
            self.participants[participant_id] = Participant(participant_id)
        self.participants[participant_id].add_session(session_number, trial_data)

        print(f"Loaded {filepath.name}  ->  participant={participant_id}, "
              f"session={session_number}, trials={len(trial_data)}")

    def _parse_filename(self, filename):
        match = self.FILENAME_PATTERN.search(filename)
        if not match:
            raise ValueError(
                f"Could not work out the participant/session from the file name "
                f"'{filename}'. Expected something like 'PID16_SESSION1.csv'."
            )
        participant_id = match.group(1)
        session_number = int(match.group(2))
        return participant_id, session_number

    def _validate_columns(self, trial_data, filename):
        missing = [col for col in REQUIRED_COLUMNS if col not in trial_data.columns]
        if missing:
            raise ValueError(
                f"'{filename}' is missing required column(s): {missing}.\n"
                f"Columns found in file: {list(trial_data.columns)}"
            )

    # ---------------------------------------------------------- analysis

    def classify_learners(self):
        """
        Performs the good/bad learner split:

          1. For each participant, look at SESSION 1 only.
          2. Within that session, for each maze:
               - average time_taken of the first 2 trials
               - average time_taken of the last 2 trials
               - percentage improvement = (first_mean - last_mean)
                                          / first_mean * 100
          3. Average that improvement percentage across all 3 mazes
             to get one score per participant.
          4. Label each participant:
                improvement >  0%  ->  "Good Learner" (got faster)
                improvement <= 0%  ->  "Bad Learner"  (no improvement
                                                        or got slower)

        Classification is based entirely on each participant's OWN
        improvement, not a comparison against the rest of the group.
        Even with only one participant loaded, the label is meaningful.

        Returns the summary table (also stored as self.summary).
        """
        rows = []
        for participant in self.participants.values():
            improvement_pct = participant.percentage_improvement_session_1()
            rows.append({
                "Participant_ID": participant.participant_id,
                "Session_Used":   1,
                "Pct_Improvement": round(improvement_pct, 2),
            })

        summary = pd.DataFrame(rows)

        # Good Learner = any positive improvement (got closer to the platform).
        # Bad Learner  = zero or negative (did not improve, or got worse).
        summary["Learner_Group"] = summary["Pct_Improvement"].apply(
            lambda pct: "Good Learner" if pct > 50 else "Bad Learner"
        )

        # Sort best improvers first
        self.summary = summary.sort_values(
            "Pct_Improvement", ascending=False
        ).reset_index(drop=True)
        return self.summary

    # ---------------------------------------------------------- saving

    def save_results(self, output_folder="processed_data"):
        """
        Saves two CSV files into `output_folder`:

          participant_summary.csv
              One row per participant: their average distance, the
              group mean it was compared against, and their Good/Bad
              Learner label. This is your main result.

          all_trials_combined.csv
              Every trial from every file you loaded, tagged with
              Participant_ID / Session / Used_For_Classification.
              Keep this around for later analyses (e.g. eye-tracking
              correlations) so you don't need to re-read the raw files.
        """
        if self.summary is None:
            raise RuntimeError("Call classify_learners() before save_results().")

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        summary_path = output_folder / "participant_summary.csv"
        self.summary.to_csv(summary_path, index=False)

        all_trials = pd.concat(
            [p.all_trials_combined() for p in self.participants.values()],
            ignore_index=True,
        )
        trials_path = output_folder / "all_trials_combined.csv"
        all_trials.to_csv(trials_path, index=False)

        print(f"Saved: {summary_path}")
        print(f"Saved: {trials_path}")
        return summary_path, trials_path


# ======================================================================
# RUN THE ANALYSIS
# ======================================================================
# This is the only part of the file you need to touch.
if __name__ == "__main__":

    # STEP 1 - list every CSV file you want included. Add as many
    # participants and sessions as you like, the script handles any
    # number of files as long as they follow the PIDxx_SESSIONx.csv
    # naming pattern.
    csv_files = [
        "PID16_SESSION1.csv",
        "PID16_SESSION2.csv",
        # "PID17_SESSION1.csv",
        # "PID17_SESSION2.csv",
    ]

    # STEP 2 - load the files, classify learners, print + save results.
    study = WaterMazeStudy()
    study.load_csv_files(csv_files)

    summary = study.classify_learners()
    print("\n--- Good / Bad Learner Summary ---")
    print(summary.round(3).to_string(index=False))

    study.save_results("processed_data")
