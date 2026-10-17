# Windows prerequisites by hand

`deploy\windows\deploy.ps1` installs all of these itself. Use this page only when the server cannot download them
(no winget and no internet route to python.org or Microsoft's downloads): fetch the files on another machine, copy
them to the server's Downloads folder, install them as below in an Administrator PowerShell, then run `deploy.ps1`.

## 1. The files

1. Git for Windows: `https://git-scm.com/download/win`
2. Python 3.12, Windows x64: `https://www.python.org/downloads/windows/`
3. Microsoft Visual C++ Redistributable x64: `https://aka.ms/vs/17/release/vc_redist.x64.exe`
4. Microsoft ODBC Driver 18 for SQL Server x64:
   `https://learn.microsoft.com/sql/connect/odbc/download-odbc-driver-for-sql-server`

## 2. Install them

```powershell
Start-Process -FilePath "$env:USERPROFILE\Downloads\Git-64-bit.exe" -ArgumentList "/VERYSILENT /NORESTART" -Wait

Start-Process -FilePath "$env:USERPROFILE\Downloads\python-3.12.10-amd64.exe" `
  -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1 Include_test=0" -Wait

Start-Process -FilePath "$env:USERPROFILE\Downloads\vc_redist.x64.exe" -ArgumentList "/install /quiet /norestart" -Wait

Start-Process msiexec.exe -Wait `
  -ArgumentList "/i `"$env:USERPROFILE\Downloads\msodbcsql.msi`" /quiet /norestart IACCEPTMSODBCSQLLICENSETERMS=YES"
```

Use the file names you downloaded where they differ. Close PowerShell and open a new Administrator one, so the new
`PATH` is read, then check:

```powershell
git --version
py -3.12 --version
Get-OdbcDriver -Name "ODBC Driver 18 for SQL Server" -Platform 64-bit
```

## 3. Install Solviz

```powershell
git clone https://github.com/sud-sn/Querybot_v2.git C:\Solviz
cd C:\Solviz
powershell -ExecutionPolicy Bypass -File deploy\windows\deploy.ps1 -AdminEmail you@company.com -AdminName "Your Name"
```

The Python libraries still come from PyPI (`pypi.org` and `files.pythonhosted.org`); a server with no route there
needs a PyPI mirror or proxy, given to pip in `%APPDATA%\pip\pip.ini`.

## Optional: PostgreSQL for Solviz's own store

SQLite (`data\querybot.db`) is right for one server. To use PostgreSQL instead, create a database and a login that
owns it, then add to `.env` (percent-encode special characters in the password) and restart:

```text
DATABASE_URL=postgresql://querybot:<password>@127.0.0.1:5432/querybot
```

Solviz creates and upgrades its tables on start. Keep PostgreSQL's port closed to the internet.
