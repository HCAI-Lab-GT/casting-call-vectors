const state = {
  rows: [],
  columns: [],
  filteredRows: [],
  sortKey: null,
  sortDirection: "asc",
  currentPage: 1,
  rowsPerPage: 25,
  expandedRowIds: new Set(),
};

const RESPONSE_COLUMNS = new Set(["assistant_axis", "baseline", "steered"]);

const fileInput = document.getElementById("csv-file-input");
const searchInput = document.getElementById("search-input");
const rowsPerPageSelect = document.getElementById("rows-per-page");
const statusMessage = document.getElementById("status-message");
const tableContainer = document.getElementById("table-container");
const pageInfo = document.getElementById("page-info");
const prevPageButton = document.getElementById("prev-page");
const nextPageButton = document.getElementById("next-page");

function getVisibleColumns() {
  return state.columns.filter((column) => !RESPONSE_COLUMNS.has(column));
}

function parseCsv(csvText) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;

  for (let index = 0; index < csvText.length; index += 1) {
    const char = csvText[index];

    if (inQuotes) {
      if (char === '"') {
        if (csvText[index + 1] === '"') {
          field += '"';
          index += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
      continue;
    }

    if (char === ",") {
      row.push(field);
      field = "";
      continue;
    }

    if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }

    if (char === "\r") {
      continue;
    }

    field += char;
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  return rows;
}

function normalizeRecords(records) {
  if (records.length === 0) {
    return { columns: [], rows: [] };
  }

  const columns = records[0].map((name, index) => {
    const header = String(name || "").trim();
    return header || `column_${index + 1}`;
  });

  const normalizedRows = records.slice(1).map((record, rowIndex) => {
    const mapped = {};
    columns.forEach((column, index) => {
      mapped[column] = record[index] ?? "";
    });
    mapped.__rowId = String(rowIndex);
    return mapped;
  });

  return { columns, rows: normalizedRows };
}

function applyFilterAndSort() {
  const query = searchInput.value.trim().toLowerCase();

  const filtered = state.rows.filter((row) => {
    if (!query) {
      return true;
    }

    return state.columns.some((column) => String(row[column] ?? "").toLowerCase().includes(query));
  });

  if (state.sortKey) {
    filtered.sort((left, right) => {
      const leftValue = left[state.sortKey] ?? "";
      const rightValue = right[state.sortKey] ?? "";

      const leftNumber = Number(leftValue);
      const rightNumber = Number(rightValue);
      const areNumbers = Number.isFinite(leftNumber) && Number.isFinite(rightNumber);

      let comparison;
      if (areNumbers) {
        comparison = leftNumber - rightNumber;
      } else {
        comparison = String(leftValue).localeCompare(String(rightValue));
      }

      return state.sortDirection === "asc" ? comparison : -comparison;
    });
  }

  state.filteredRows = filtered;
}

function buildHeaderCell(column) {
  const headerCell = document.createElement("th");
  const button = document.createElement("button");
  const isSorted = state.sortKey === column;
  const directionMarker = isSorted ? (state.sortDirection === "asc" ? " ▲" : " ▼") : "";

  button.textContent = `${column}${directionMarker}`;
  button.type = "button";
  button.addEventListener("click", () => {
    if (state.sortKey === column) {
      state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = column;
      state.sortDirection = "asc";
    }
    state.currentPage = 1;
    render();
  });

  headerCell.appendChild(button);
  return headerCell;
}

function renderTable() {
  tableContainer.innerHTML = "";
  const visibleColumns = getVisibleColumns();

  if (state.filteredRows.length === 0) {
    tableContainer.innerHTML = "<p style=\"padding:12px;margin:0\">No rows to display.</p>";
    return;
  }

  if (visibleColumns.length === 0) {
    tableContainer.innerHTML = "<p style=\"padding:12px;margin:0\">All columns are hidden.</p>";
    return;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const body = document.createElement("tbody");

  const headerRow = document.createElement("tr");
  visibleColumns.forEach((column) => {
    headerRow.appendChild(buildHeaderCell(column));
  });
  const responseHeader = document.createElement("th");
  responseHeader.textContent = "responses";
  headerRow.appendChild(responseHeader);
  thead.appendChild(headerRow);

  const start = (state.currentPage - 1) * state.rowsPerPage;
  const end = start + state.rowsPerPage;
  const pageRows = state.filteredRows.slice(start, end);

  pageRows.forEach((row) => {
    const rowId = String(row.__rowId ?? "");
    const isExpanded = state.expandedRowIds.has(rowId);

    const tr = document.createElement("tr");
    visibleColumns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = String(row[column] ?? "");
      tr.appendChild(td);
    });

    const actionCell = document.createElement("td");
    const toggleButton = document.createElement("button");
    toggleButton.type = "button";
    toggleButton.className = "row-response-toggle";
    toggleButton.textContent = isExpanded ? "Hide Responses" : "Show Responses";
    toggleButton.addEventListener("click", () => {
      if (state.expandedRowIds.has(rowId)) {
        state.expandedRowIds.delete(rowId);
      } else {
        state.expandedRowIds.add(rowId);
      }
      renderTable();
    });
    actionCell.appendChild(toggleButton);
    tr.appendChild(actionCell);

    body.appendChild(tr);

    if (isExpanded) {
      const detailRow = document.createElement("tr");
      detailRow.className = "response-detail-row";
      const detailCell = document.createElement("td");
      detailCell.colSpan = visibleColumns.length + 1;

      const detailWrap = document.createElement("div");
      detailWrap.className = "response-detail-wrap";

      const entries = [
        { label: "assistant_axis", value: row.assistant_axis },
        { label: "baseline", value: row.baseline },
        { label: "steered", value: row.steered },
      ];

      entries.forEach((entry) => {
        const section = document.createElement("div");
        section.className = "response-block";

        const label = document.createElement("p");
        label.className = "response-label";
        label.textContent = entry.label;

        const text = document.createElement("pre");
        text.className = "response-text";
        text.textContent = String(entry.value ?? "");

        section.appendChild(label);
        section.appendChild(text);
        detailWrap.appendChild(section);
      });

      detailCell.appendChild(detailWrap);
      detailRow.appendChild(detailCell);
      body.appendChild(detailRow);
    }
  });

  table.appendChild(thead);
  table.appendChild(body);
  tableContainer.appendChild(table);
}

function updatePaginationControls() {
  const totalPages = Math.max(1, Math.ceil(state.filteredRows.length / state.rowsPerPage));
  if (state.currentPage > totalPages) {
    state.currentPage = totalPages;
  }

  pageInfo.textContent = `Page ${state.currentPage} of ${totalPages}`;
  prevPageButton.disabled = state.currentPage <= 1;
  nextPageButton.disabled = state.currentPage >= totalPages;
}

function updateStatus() {
  const rowCount = state.filteredRows.length;
  const totalCount = state.rows.length;
  const visibleColumnCount = getVisibleColumns().length + 1;
  const totalColumnCount = state.columns.length + 1;

  if (totalCount === 0) {
    statusMessage.textContent = "Select a CSV file to begin.";
    return;
  }

  statusMessage.innerHTML = `<strong>${rowCount}</strong> visible rows of ${totalCount} total rows across ${visibleColumnCount} visible columns (${totalColumnCount} total).`;
}

function render() {
  applyFilterAndSort();
  updatePaginationControls();
  renderTable();
  updateStatus();
}

async function loadCsvFromFile(file) {
  const text = await file.text();
  const records = parseCsv(text);
  const { columns, rows } = normalizeRecords(records);

  state.columns = columns;
  state.rows = rows;
  state.filteredRows = rows;
  state.sortKey = null;
  state.sortDirection = "asc";
  state.currentPage = 1;
  state.expandedRowIds = new Set();

  render();
}

fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  try {
    await loadCsvFromFile(file);
  } catch (error) {
    statusMessage.textContent = `Failed to parse CSV: ${error instanceof Error ? error.message : String(error)}`;
    tableContainer.innerHTML = "";
    pageInfo.textContent = "Page 0 of 0";
  }
});

searchInput.addEventListener("input", () => {
  state.currentPage = 1;
  render();
});

rowsPerPageSelect.addEventListener("change", () => {
  state.rowsPerPage = Number(rowsPerPageSelect.value);
  state.currentPage = 1;
  render();
});

prevPageButton.addEventListener("click", () => {
  state.currentPage = Math.max(1, state.currentPage - 1);
  render();
});

nextPageButton.addEventListener("click", () => {
  const totalPages = Math.max(1, Math.ceil(state.filteredRows.length / state.rowsPerPage));
  state.currentPage = Math.min(totalPages, state.currentPage + 1);
  render();
});

render();
