---
license: mit
---

# ARC-AGI-2 Human testing data

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/cookics/human-data-analysis)

This repository contains data from human testing sessions on ARC-AGI tasks. 

Each row represents a single test attempt by a human participant on a specific task-test pair in the "Public Train" or "Public Eval" ARC-AGI-2 datasets. Not all tasks in the released "Public Train"
sets were tested, so these results are not comprehensive. This data does not include tasks from "Semi Private Evaluation" or "Private Evaluation"

### Column Descriptions

- **task_ID**: A unique identifier for the ARC task that was attempted.
- **task_set**: Indicates whether the task belongs to the "Public Train" or "Public Eval" dataset.
- **test_index**: The index of the test example within the task (typically 0 or 1).
- **session_ID**: A unique identifier for the testing session in which this attempt was made.
- **start_time_seconds**: The time (in seconds) from the start of the session when this task attempt was begun.
- **duration_seconds**: The total time (in seconds) spent on this task attempt.
- **submissions**: The number of solution submissions made during this attempt.
- **correct_submissions**: The number of correct solution submissions made during this attempt (0 if the participant did not solve the task correctly, 1 or more if they did).

For more information about ARC Prize and ARC-AGI-2, see the [technical paper](https://arcprize.org/blog/arc-agi-2-technical-report).