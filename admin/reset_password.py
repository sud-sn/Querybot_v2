"""
Set a new admin console password from the server.

    python -m admin.reset_password            # asks twice, without echo
    python -m admin.reset_password --stdin    # reads one line, for automation

For when the admin password is lost, or when the stored one can no longer be
read because the encryption key changed. The web setup page stays closed once
a password exists, so this runs where only someone with a shell on the server
can run it. Run it as the service's user, from the application directory, with
the same QUERYBOT_DB_PATH, DATABASE_URL and QUERYBOT_KEY_FILE as the service.
"""

from __future__ import annotations

import argparse
import getpass
import sys

import store
from admin import credentials


def main(argv: list[str] | None = None, *, prompt=getpass.getpass, stdin=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m admin.reset_password",
                                     description="Set a new admin console password.")
    parser.add_argument("--stdin", action="store_true",
                        help="read the new password from one line of standard input")
    args = parser.parse_args(argv)

    if args.stdin:
        password = (stdin or sys.stdin).readline().rstrip("\r\n")
    else:
        password = prompt("New admin password: ")
        if password != prompt("Type it again: "):
            print("The two passwords differ. Nothing was changed.", file=sys.stderr)
            return 1
    if len(password) < credentials.MIN_LENGTH:
        print(f"The password must be at least {credentials.MIN_LENGTH} characters. "
              "Nothing was changed.", file=sys.stderr)
        return 1

    store.init_db()
    credentials.set_password(password)
    print("The admin password is set and every admin session is signed out. "
          "Sign in at /admin/login.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
