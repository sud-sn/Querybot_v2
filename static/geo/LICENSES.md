# Map files and where they come from

Built by tools/geo/build_geo.py from:

- us-states.json, ca-provinces.json, countries.json, names.json: Natural Earth (naturalearthdata.com),
  1:50m admin-1 states and provinces and 1:110m admin-0 countries. Public domain: "No permission is
  needed to use Natural Earth." Outlines rounded to 0.01 degree; Hawaii's islands north-west of Kauai are
  left off, as on most maps of the states.

- us-zip.csv: US ZIP codes with a latitude and longitude, from github.com/millbj92/US-Zip-Codes-JSON,
  under the MIT licence below. Only the ZIP code and its point are kept, at 0.01 degree.

```
MIT License

Copyright (c) 2022 Brandon Miller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
