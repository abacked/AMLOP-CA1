# Initial project plan

These are starter Product Backlog Items derived from the CA1 brief. The team should confirm priorities, ownership, and effort estimates during its requirements and sprint meetings. Each item should become a GitHub Issue and be tracked on the Project board.

| ID | Product Backlog Item | Suggested effort | Target |
| --- | --- | ---: | --- |
| PBI-01 | Explore the base data and document data quality, distributions, anomalies, and class balance | 5 | Part 1 |
| PBI-02 | Build reusable cleaning, validation, joins, and feature transformations for all three schemas | 8 | Part 2 |
| PBI-03 | Design MySQL tables and configure an AWS RDS instance | 5 | Part 2 |
| PBI-04 | Load base and new batch data into RDS with idempotent, observable ETL | 8 | Part 2 |
| PBI-05 | Deploy a team MLflow service on AWS Elastic Beanstalk | 8 | Part 2 |
| PBI-06 | Train and log at least two model versions per student | 8 per student | Part 2 |
| PBI-07 | Compare team models, register the best two, and promote a champion | 5 | Part 2 |
| PBI-08 | Orchestrate each new batch's ETL and retrain every team model with named MLflow runs | 8 | Part 2 |
| PBI-09 | Evaluate and document a more robust MLflow architecture using the required web GenAI conversation | 5 | Part 2 |
| PBI-10 | Build three meaningful RDS business insight queries and visuals (individual work) | 5 per student | Part 2 |
| PBI-11 | Run the individual analytics with Prefect and expose current results in flow logs or reports | 5 per student | Part 2 |
| PBI-12 | Prepare submission evidence, deployment runbook, and a rehearsed three-minute demo | 5 | Part 2 |

Effort values are **suggested story points**, not final team estimates. The brief specifies that each student completes their individual component independently. The optional AWS Prefect deployment can be added after the core work is reliable.

## Board setup

Use status columns `Backlog`, `Ready`, `In progress`, `In review`, and `Done`. Add an `Estimate` number field and a `Sprint` iteration field if available. Create a board view grouped by status. Do not assign teammates until the team agrees on ownership.

## Definition of done

- The issue's acceptance criteria are met and its evidence is linked.
- Code changes are reviewed through a pull request.
- Configuration contains no credentials or hard-coded service URLs.
- The relevant pipeline has been run on permitted data and its outcome recorded.
