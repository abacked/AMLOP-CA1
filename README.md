# AMLOP CA1 — Fraud Detection

Team workspace for ST1519 Agile MLOps CA1 (AY2026/27 Semester 2). The project studies fraudulent orders on the fictional SP-buy platform and develops reproducible data, model, and analytics pipelines.

## Assignment milestones

| Milestone | Deadline (Singapore time) |
| --- | --- |
| Part 1 group product backlog | 13 Nov 2026, 4:00 pm |
| Part 2 group and individual submissions | 27 Nov 2026, 4:00 pm |
| Interview and live demo | Week 7; tutor to confirm slot |

Submission is through Brightspace. A GitHub repository or Project board is a work area, not the final submission by itself.

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Group data exploration and cleaning notebook |
| `src/amlop_ca1/` | Reusable data, feature, training, and ETL code |
| `tests/` | Checks for pipeline behavior and schema assumptions |
| `infra/` | Deployment configuration for MLflow and database setup |
| `docs/` | Decisions, runbooks, and demo notes |
| `.github/` | GitHub issue and pull request templates |

The three supplied base CSVs are stored in `data/` with Git LFS. Install Git LFS before cloning or pulling so you receive the CSV contents rather than pointer files. The assessment PDF is not committed. Future batch files remain ignored by default; record their source URL, batch suffix, and processing outcome in ETL logs.

## Team workflow

1. Refine the Product Backlog Items in GitHub Issues during the Week 3 requirements meeting. Set estimates with the team.
2. Add each issue to the GitHub Project. During Week 4 sprint planning, break PBIs into assigned tasks.
3. Create a branch for each task (`feature/<issue-number>-short-name`), open a pull request, and request a teammate review.
4. Record model experiment runs, metrics, and registry decisions in MLflow. Keep infrastructure endpoints and secrets in environment variables, never in Git.
5. Before the interview, rehearse the new-batch ETL and individual analytics flows against permitted test data. Only process tutor-provided interview URLs when the tutor supplies them at the interview.

See [docs/project-plan.md](docs/project-plan.md) for the initial backlog and acceptance criteria.

GitHub Project: [AMLOP CA1 - Fraud Detection](https://github.com/users/abacked/projects/4). Its `Product backlog` table holds the PBIs and estimates; `Sprint board` tracks their progress.
