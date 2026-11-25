# Security policy

## Reporting a vulnerability

Please report security problems **in private**, not in a public issue.

Use GitHub's private reporting: open the [Security tab](https://github.com/gastonzarate/opentraductor/security) of this repository and choose **Report a vulnerability**. The report is visible only to the maintainer.

If that option is not available to you, open a public issue that says only "I need to report a security problem, please contact me", with no details, and the maintainer will arrange a private channel.

Please include: what you found, the steps to reproduce it, the commit or version, and your Mac model and macOS version.

## What to expect

opentraductor is maintained by one person. You will get an answer within about 7 days, and we will agree on a date to publish the fix. There is no bug bounty.

## Scope

In scope: the local control server (it listens on `127.0.0.1` only, with a per-run token and `Host`/`Origin` checks), the local web UI, reading and writing of the local text history, model download and file verification (`runtime.setup`, `scripts/setup_moonshine.py`), and the packaging scripts in `packaging/`.

Out of scope: vulnerabilities in third-party models, libraries or in BlackHole (report them to their own projects), and attacks that already need full control of your Mac account.

## Supported versions

Only the latest commit on `main`. There are no tagged releases yet.

## Design facts that matter for security

- After the one-time model download, the runtime forces offline mode. There is no telemetry and no account.
- No audio is stored. The history keeps only text, in a local SQLite file.
