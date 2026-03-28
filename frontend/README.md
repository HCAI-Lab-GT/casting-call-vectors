# Frontend CSV Viewer

Quick static frontend to inspect CSV outputs as a table.

## Run

From the `persona-vectors` folder:

```bash
cd frontend
python -m http.server 8080
```

Then open:

- http://localhost:8080

## Usage

1. Click **Select CSV** and choose your results CSV.
2. Use **Search** to filter rows.
3. Click column headers to sort ascending/descending.
4. Use pagination controls to browse large files.

## Notes

- CSV parser supports quoted fields, escaped quotes, and multiline values.
- This app is static and has no backend dependencies.
