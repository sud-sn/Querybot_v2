// The sources static/dist was built from, as one hash (static/dist/SOURCES): a test computes it again and fails
// when a source changed and the build was not run. Line endings are read as \n, so a Windows checkout agrees.
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = fileURLToPath(new URL('..', import.meta.url))
const files = []
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) walk(path)
    else files.push(path)
  }
}
walk(join(here, 'src'))
files.push(join(here, 'package.json'), join(here, 'package-lock.json'), join(here, 'vite.config.js'))
const hash = createHash('sha256')
for (const path of files.map((p) => relative(here, p).split(sep).join('/')).sort()) {
  hash.update(path + '\0')
  hash.update(readFileSync(join(here, path), 'utf8').replace(/\r\n/g, '\n') + '\0')
}
writeFileSync(join(here, '..', 'static', 'dist', 'SOURCES'), hash.digest('hex') + '\n')
