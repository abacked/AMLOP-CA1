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

The starter repository contains the collaboration structure only. The supplied CSVs and assessment PDF stay local; `.gitignore` excludes them from commits. Copy future batch files into `data/` locally, and record the source URL, batch suffix, and processing outcome in the ETL logs.

## Team workflow

1. Refine the Product Backlog Items in GitHub Issues during the Week 3 requirements meeting. Set estimates with the team.
2. Add each issue to the GitHub Project. During Week 4 sprint planning, break PBIs into assigned tasks.
3. Create a branch for each task (`feature/<issue-number>-short-name`), open a pull request, and request a teammate review.
4. Record model experiment runs, metrics, and registry decisions in MLflow. Keep infrastructure endpoints and secrets in environment variables, never in Git.
5. Before the interview, rehearse the new-batch ETL and individual analytics flows against permitted test data. Only process tutor-provided interview URLs when the tutor supplies them at the interview.

See [docs/project-plan.md](docs/project-plan.md) for the initial backlog and acceptance criteria.
