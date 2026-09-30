# Changelog

Each release has a version: the `VERSION` file, which `/health` and the admin
sidebar show for the build that is running. Its entry here says what changed
for the people asking questions and for admins, and what an upgrade to it asks
of an admin. `docs/UPGRADE_RUNBOOK.md` is how to deploy a release and how to
roll it back.

A release that changes what a knowledge-base build writes raises the
knowledge-base format (`KB_FORMAT` in `core/release.py`), and its entry says so.
Every workspace a different format built is then told to rebuild: in the
startup log, in the dashboard's inbox and on its setup page.

## 2.1.0

The first release with a version. Builds before it report 2.0.0, whatever
they are.

### Upgrade

- **Rebuild the knowledge base.** This release builds knowledge-base format 1.
  A knowledge base built before it is format 0, and its workspace says so
  until it is rebuilt (runbook section 8).
- **Keep the key file.** At startup the server checks that it can read every
  saved credential with its key, and names each one it cannot, saying whether
  the key file is missing, cannot be opened by the service's user, or is a
  different key. Only saving a credential writes a new key file; reading one
  never does.
- **Read `/health` after the deploy.** It gives the version and release now,
  and counts the workspaces to rebuild and the credentials that cannot be read.
- The store upgrades itself at startup. There is nothing to run by hand.

### Answers

- A French question about orders ("commandes") is answered with the quantity
  ordered, not the back-ordered quantity.
- A count by code keeps a member whose code was never filled in.
- A French "lowest" headline names the lowest, whatever the word's agreement.
- A quantity kept in several units of measure is given as one total per unit.
  The totals are never ranked, compared or added across units.
- A share of a total ("holds 62% of the total") is only given where the answer
  saw the whole total. A top five cut by a limit, rows in more than one unit
  and a measure that does not add up get no share.
- A measure's verb in any tense ("sold", "vendu") is never read as a member to
  filter by.
- A diagnostic reply on Teams, Slack and Zoom is in the reader's language.

### Admin

- Test connection names what a warehouse login can do beyond reading, in
  amber. The connection still works.
- The setup wizard and the join tester say what a suggested join does: nothing,
  until it is confirmed.
- The dashboard's inbox and a workspace's setup page say when its knowledge
  base was built by another release. The dashboard names saved credentials the
  key cannot read, and the dashboard and Databases page no longer fail on one.
- The sidebar shows the version running.
