# Repository history

The `research-main` branch starts from the September 25, 2026 release tree and groups subsequent work by responsibility.

The complete 486-commit development lineage remains at `archive-development-20261109` (`6801a3ceb6bdfac5dc06e78fa182a21c1f30227d`). Published branches and remote refs were not rewritten.

The tree at the end of the grouped commits is byte-for-byte identical to the submitted development snapshot. The later maintenance commits contain the storage and layout changes.

| Commit | Scope |
| --- | --- |
| `9552187f4a` | 2026-09-25 release baseline |
| `98d7cbf73c` | feat(data): add catalog expansion and reproducible window protocols |
| `d3f532156d` | feat(models): integrate event-time Mamba and regional point processes |
| `b4035bb794` | feat(experiments): add revision runners, diagnostics and analysis |
| `169cd7dd00` | chore(data): record source snapshots and exclude generated artifacts |

Use `git log --oneline research-main` for the concise history and `git log archive-development-20261109` for the original lineage.

Generated caches and training weights are local artifacts. Source snapshots are tracked through Git LFS; do not put new cache variants into Git. No remote push has been performed.
