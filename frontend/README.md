# The portal's Vue pieces

The dashboard's filter bar (the date picker over every tile, each field's filter, the owner's Add filter) and the
in-place redraw under it. Built with Vite into `static/dist`, which is kept in the repository: the server never
needs Node, and the page loads `static/dist/dashboard.js` through `asset()`, versioned by its content.

After changing anything here:

    cd frontend
    npm ci          # once, or after package-lock.json changes
    npm run build   # writes static/dist and static/dist/SOURCES

and commit `static/dist` with the change. `tests/test_the_built_frontend_matches_its_sources.py` fails when a
source changed and the build was not run (it compares the hash of these sources with `static/dist/SOURCES`).
