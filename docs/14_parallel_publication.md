# Parallel publication of generated artifacts

`scripts/publish_generated_package_parallel.py` is a separate uploader for the
same audited multipart packages. It leaves the original uploader and any active
serial upload unchanged. No upload was performed while implementing or testing it.

The original package validator and remote digest validator remain unchanged.
The new script captures its own and both imported publisher-file hashes before
loading them. It checks every existing remote asset before launching uploads.
An identical uploaded asset is retained; a conflicting or incomplete asset stops
the run. It never passes `--clobber`, deletes assets, or silently replaces a release.

At most three independent `gh release upload` processes run at once, each for a
different missing asset. API, tag-resolution, and release-creation commands have
a 45-second cap and share the remaining overall budget. Failed or interrupted
runs stop their own process groups, including descendants whose leader exited.
They do not stop another publisher. Successfully uploaded assets can be verified
and reused on a later run with the same request and report path. Interrupted
partial assets require review; they are never removed automatically.

Before success, the script rechecks the actual tag commit, public release
identity, title and notes, the exact asset set, every remote size and SHA256, all
local package bytes and source bindings, and unchanged code/notes hashes. The
completed report retains the original publisher's schema, with concurrency and
release-header checks added. A separate report name keeps an existing publisher's
state immutable; attempting to resume with different inputs fails before writing
that state.

Use the same arguments as the original publisher, with a **new report path**:

```bash
.venv/bin/python scripts/publish_generated_package_parallel.py \
  --package-report results/generated_publication_confirmation_complete.json \
  --tag TAG --target EXACT_PUBLISHED_COMMIT --title TITLE \
  --notes-file NOTES_FILE --report NEW_PUBLICATION_REPORT --workers 3
```

This example contains placeholders and is not an executed publication. Run only
after the package and independent byte review pass. Keep one publisher per release.
The default overall budget is six hours; local validation also checks this budget
before the next network command. Cleanup has a short bounded grace period.

Focused tests use local fake `gh` subprocesses, with no network calls. They check
the three-process cap, distinct asset names, matching-asset resume, conflicts
before mutation, failed uploads, timeouts, orphan descendants, hanging metadata
and create commands, annotated tag resolution, draft rejection, and preservation
of a prior request's state.
