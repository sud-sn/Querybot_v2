"""
Make a named admin account from the server.

    python -m admin.create_admin you@company.com "Your Name"                 # a Super admin; asks for a password twice
    python -m admin.create_admin you@company.com "Your Name" --stdin         # reads the password from one line
    python -m admin.create_admin a@company.com "A" --role data_steward --workspace acct-1 --workspace acct-2

For a server where nobody can sign in as a Super admin and the emergency
sign-in is off, or for setting up admins by script. Run it as the service's
user, from the application directory, with the same QUERYBOT_DB_PATH,
DATABASE_URL and QUERYBOT_KEY_FILE as the service. An email that is an admin
already gets the new password (and role, and workspaces) instead.
"""

from __future__ import annotations

import argparse
import getpass
import sys

import store
from store.admin_store import ADMIN_ROLES, MIN_PASSWORD, SUPER_ADMIN


def main(argv: list[str] | None = None, *, prompt=getpass.getpass, stdin=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m admin.create_admin", description="Make a named admin account.")
    parser.add_argument("email")
    parser.add_argument("name", nargs="?", default="")
    parser.add_argument("--role", choices=ADMIN_ROLES, default=SUPER_ADMIN)
    parser.add_argument("--workspace", action="append", default=[],
                        help="a workspace (account id) a workspace admin or data steward looks after; repeat for more")
    parser.add_argument("--stdin", action="store_true", help="read the password from one line of standard input")
    args = parser.parse_args(argv)

    email = store.admin_store.normalise_email(args.email)
    if "@" not in email:
        print("Give an email address. Nothing was changed.", file=sys.stderr)
        return 1
    if args.role != SUPER_ADMIN and not args.workspace:
        print("A workspace admin or data steward needs at least one --workspace. Nothing was changed.", file=sys.stderr)
        return 1
    if args.stdin:
        password = (stdin or sys.stdin).readline().rstrip("\r\n")
    else:
        password = prompt("Password: ")
        if password != prompt("Type it again: "):
            print("The two passwords differ. Nothing was changed.", file=sys.stderr)
            return 1
    if len(password) < MIN_PASSWORD:
        print(f"The password must be at least {MIN_PASSWORD} characters. Nothing was changed.", file=sys.stderr)
        return 1

    store.init_db()
    known = {c["account_id"] for c in store.list_clients()}
    unknown = [w for w in args.workspace if w not in known]
    if unknown:
        print(f"No such workspace: {', '.join(unknown)}. Nothing was changed.", file=sys.stderr)
        return 1
    admin = store.get_admin_by_email(email)
    if admin is None:
        admin_id = store.create_admin(email, args.name, args.role, password=password, workspaces=args.workspace,
                                      created_by="python -m admin.create_admin")
        verb = "made"
    else:
        admin_id = int(admin["id"])
        store.update_admin(admin_id, name=args.name or None, role=args.role if args.role != admin["role"] else None,
                           is_active=True)
        store.set_admin_password(admin_id, password)
        store.set_admin_workspaces(admin_id, [] if args.role == SUPER_ADMIN else args.workspace)
        verb = "updated"
    store.record_admin_event(admin_id=admin_id, admin_email=email, action="admin_created" if verb == "made" else
                             "admin_changed", method="CLI", path="python -m admin.create_admin",
                             detail=f"{email} as {store.ADMIN_ROLE_LABELS[args.role]}, from the server")
    print(f"The {store.ADMIN_ROLE_LABELS[args.role]} account for {email} is {verb}. Sign in at /admin/login.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
