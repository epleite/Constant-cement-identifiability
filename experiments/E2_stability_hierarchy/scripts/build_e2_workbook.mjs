import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const packageRoot = process.env.E2_PACKAGE_ROOT;
const dataPath = process.env.E2_WORKBOOK_DATA;
const outputPath = process.env.E2_WORKBOOK_OUTPUT;
const renderDir = process.env.E2_WORKBOOK_RENDER_DIR;
if (!packageRoot || !dataPath || !outputPath || !renderDir) {
  throw new Error("E2 workbook environment variables are required");
}
const data = JSON.parse(await fs.readFile(dataPath, "utf8"));
const workbook = Workbook.create();

const palette = {
  navy: "#17324D",
  blue: "#3478A6",
  teal: "#2A9D8F",
  gold: "#E9C46A",
  orange: "#F4A261",
  red: "#C94C4C",
  ink: "#1F2937",
  muted: "#667085",
  pale: "#EEF2F5",
  white: "#FFFFFF",
  line: "#D0D5DD",
};

function matrixFromRecords(records) {
  if (!records.length) return { headers: ["empty"], rows: [["No records"]] };
  const headers = Object.keys(records[0]);
  const rows = records.map((row) => headers.map((header) => row[header] ?? null));
  return { headers, rows };
}

function columnLetter(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    n -= 1;
    out = String.fromCharCode(65 + (n % 26)) + out;
    n = Math.floor(n / 26);
  }
  return out;
}

function titleBand(sheet, title, subtitle, lastColumn) {
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: palette.navy,
    font: { bold: true, color: palette.white, size: 16 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: palette.pale,
    font: { color: palette.ink, italic: true, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("1:1").format.rowHeight = 28;
  sheet.getRange("2:2").format.rowHeight = 31;
}

function formatColumn(sheet, columnIndex, header, rowCount, values) {
  const letter = columnLetter(columnIndex);
  const range = sheet.getRange(`${letter}4:${letter}${rowCount + 3}`);
  const lower = header.toLowerCase();
  if (lower.includes("vcem")) {
    range.format.numberFormat = "0.00000";
  } else if (lower.includes("fraction") || lower.includes("rate") || lower.includes("probability")) {
    range.format.numberFormat = "0.0%";
  } else if (lower.includes("cn") || lower.includes("gamma") || lower.startsWith("a_")) {
    range.format.numberFormat = "0.0000";
  } else if (
    lower.includes("objective") ||
    lower.includes("delta") ||
    lower.includes("error") ||
    lower.includes("beta") ||
    lower.includes("corr") ||
    lower.includes("lambda") ||
    lower.includes("mean") ||
    lower.includes("median") ||
    lower.includes("q02") ||
    lower.includes("q16") ||
    lower.includes("q84") ||
    lower.includes("q97") ||
    lower.includes("std") ||
    lower.includes("mad") ||
    lower.includes("estimate") ||
    lower.includes("cv_") ||
    lower.includes("value")
  ) {
    range.format.numberFormat = "0.0000";
  }
  const maxTextLength = Math.max(
    header.length,
    ...values.filter((value) => typeof value === "string").map((value) => value.length),
  );
  let width = Math.max(74, Math.min(190, 7.2 * maxTextLength + 18));
  if (["metric", "check", "detail", "note", "path", "artifact"].includes(lower)) width = 210;
  sheet.getRange(`${letter}:${letter}`).format.columnWidthPx = width;
}

function addDataSheet(name, title, subtitle, records, tableName) {
  const sheet = workbook.worksheets.getItem(name);
  sheet.showGridLines = false;
  const { headers, rows } = matrixFromRecords(records);
  const lastColumn = columnLetter(headers.length - 1);
  titleBand(sheet, title, subtitle, lastColumn);
  sheet.getRange(`A3:${lastColumn}${rows.length + 3}`).values = [headers, ...rows];
  sheet.getRange(`A3:${lastColumn}3`).format = {
    fill: palette.blue,
    font: { bold: true, color: palette.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: palette.navy },
  };
  sheet.getRange(`A4:${lastColumn}${rows.length + 3}`).format = {
    font: { color: palette.ink, size: 9 },
    verticalAlignment: "top",
    borders: {
      insideHorizontal: { style: "thin", color: "#E4E7EC" },
    },
  };
  const table = sheet.tables.add(`A3:${lastColumn}${rows.length + 3}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  sheet.freezePanes.freezeRows(3);
  headers.forEach((header, index) =>
    formatColumn(
      sheet,
      index,
      header,
      rows.length,
      rows.map((row) => row[index]),
    ),
  );
  sheet.getRange("A:A").format.columnWidthPx = Math.max(110, sheet.getRange("A:A").format.columnWidthPx ?? 110);
  return sheet;
}

const overview = workbook.worksheets.add("Overview");
for (const name of [
  "Key Results",
  "Bootstrap Summary",
  "Bootstrap Diagnostics",
  "Bootstrap Replicates",
  "LOTO Level",
  "LOTO Bootstrap",
  "LOTO Replicates",
  "LOTO Shape",
  "Hierarchy",
  "Residual ACF",
  "Verification",
  "Sources",
]) {
  workbook.worksheets.add(name);
}
overview.showGridLines = false;
overview.getRange("A1:L1").merge();
overview.getRange("A1").values = [["Constant-cement E2: stability, transport, and hierarchy"]];
overview.getRange("A1:L1").format = {
  fill: palette.navy,
  font: { bold: true, color: palette.white, size: 18 },
  verticalAlignment: "center",
};
overview.getRange("A2:L2").merge();
overview.getRange("A2").values = [["Frozen RPIA/E1 constant-cement Scheme 1; 20 m moving-block bootstrap is primary. Intervals are conditional on the two Hugin trajectories."]];
overview.getRange("A2:L2").format = {
  fill: palette.pale,
  font: { italic: true, color: palette.ink, size: 10 },
  wrapText: true,
};
overview.getRange("A4:D4").values = [["Coefficient", "Median", "95% low", "95% high"]];
overview.getRange("A5:A8").values = [["A raw"], ["Gamma raw"], ["A adjusted"], ["Gamma adjusted"]];
overview.getRange("B5:D8").formulas = [
  ["='Key Results'!B4", "='Key Results'!C4", "='Key Results'!D4"],
  ["='Key Results'!B5", "='Key Results'!C5", "='Key Results'!D5"],
  ["='Key Results'!B6", "='Key Results'!C6", "='Key Results'!D6"],
  ["='Key Results'!B7", "='Key Results'!C7", "='Key Results'!D7"],
];
overview.getRange("A4:D4").format = { fill: palette.blue, font: { bold: true, color: palette.white } };
overview.getRange("B5:D8").format.numberFormat = "0.0000";
overview.getRange("A4:D8").format.borders = { preset: "outside", style: "thin", color: palette.line };

overview.getRange("F4:H4").values = [["LOTO direction", "q-star ratio", "local tangent ratio"]];
overview.getRange("F5:F8").values = [["19A→BT2 raw"], ["19A→BT2 adjusted"], ["BT2→19A raw"], ["BT2→19A adjusted"]];
overview.getRange("G5:H8").formulas = [
  ["='LOTO Level'!M4", "='LOTO Level'!Q4"],
  ["='LOTO Level'!M5", "='LOTO Level'!Q5"],
  ["='LOTO Level'!M6", "='LOTO Level'!Q6"],
  ["='LOTO Level'!M7", "='LOTO Level'!Q7"],
];
overview.getRange("F4:H4").format = { fill: palette.teal, font: { bold: true, color: palette.white } };
overview.getRange("G5:H8").format.numberFormat = "0.000";
overview.getRange("F4:H8").format.borders = { preset: "outside", style: "thin", color: palette.line };

overview.getRange("A10:L10").merge();
overview.getRange("A10").values = [["Interpretation"]];
overview.getRange("A10:L10").format = { fill: palette.gold, font: { bold: true, color: palette.ink, size: 12 } };
overview.getRange("A11:L13").merge();
overview.getRange("A11").values = [["A and Gamma are stable under conditional within-trajectory resampling and the factored coordinate transports both level and finite ridge shape far better than the local exponential tangent. The hierarchical comparison is deliberately inconclusive: raw data slightly favor shared q-star at the point estimate, while nuisance adjustment and most block-bootstrap replicates can favor shared Cn. Both restrictions slide along weak ridges, so sharing Cn regularizes by assumption rather than creating sensitivity diversity."]];
overview.getRange("A11:L13").format = { fill: "#FFF8E1", font: { color: palette.ink, size: 10 }, wrapText: true, verticalAlignment: "top", borders: { preset: "outside", style: "thin", color: palette.gold } };

overview.getRange("A16:C16").values = [["Model", "Fixed nuisances", "Joint nuisance MAP"]];
overview.getRange("A17:A19").values = [["shared q-star"], ["shared Cn"], ["pooled theta"]];
overview.getRange("B17:C19").formulas = [
  ["='Hierarchy'!AJ5", "='Hierarchy'!AJ9"],
  ["='Hierarchy'!AJ6", "='Hierarchy'!AJ10"],
  ["='Hierarchy'!AJ7", "='Hierarchy'!AJ11"],
];
overview.getRange("A16:C16").format = { fill: palette.blue, font: { bold: true, color: palette.white } };
overview.getRange("B17:C19").format.numberFormat = "0.0000";
overview.getRange("A16:C19").format.borders = { preset: "outside", style: "thin", color: palette.line };

overview.getRange("A22:D22").values = [["Hierarchy bootstrap", "Value", "95% low", "95% high"]];
overview.getRange("A23:A24").values = [["q-star preferred fraction"], ["median Phi(Cn)-Phi(q-star)"]];
overview.getRange("B23").formulas = [["='Key Results'!B8"]];
overview.getRange("B24:D24").formulas = [["='Key Results'!B9", "='Key Results'!C9", "='Key Results'!D9"]];
overview.getRange("B23").format.numberFormat = "0.0%";
overview.getRange("B24:D24").format.numberFormat = "0.000";
overview.getRange("A22:D22").format = { fill: palette.orange, font: { bold: true, color: palette.ink } };
overview.getRange("A22:D24").format.borders = { preset: "outside", style: "thin", color: palette.line };

const hierarchyChart = overview.charts.add("bar", overview.getRange("A16:C19"));
hierarchyChart.title = "Constraint cost relative to separate model";
hierarchyChart.hasLegend = true;
hierarchyChart.yAxis = { numberFormatCode: "0.00" };
hierarchyChart.setPosition("E16", "L31");
overview.freezePanes.freezeRows(2);
overview.getRange("A:A").format.columnWidthPx = 205;
overview.getRange("B:B").format.columnWidthPx = 135;
overview.getRange("C:C").format.columnWidthPx = 165;
overview.getRange("D:D").format.columnWidthPx = 105;
overview.getRange("F:F").format.columnWidthPx = 170;
overview.getRange("G:G").format.columnWidthPx = 110;
overview.getRange("H:H").format.columnWidthPx = 155;
overview.getRange("1:1").format.rowHeight = 31;
overview.getRange("2:2").format.rowHeight = 34;

addDataSheet("Key Results", "Key results", "Values used by the Overview formulas.", data.key_results, "KeyResultsTable");
addDataSheet("Bootstrap Summary", "Bootstrap summary", "Percentile intervals for all successful and interior-only fits across IID and moving-block designs.", data.bootstrap_summary, "BootstrapSummaryTable");
addDataSheet("Bootstrap Diagnostics", "Bootstrap diagnostics", "Failure, boundary, correlation, and hierarchy-preference diagnostics by resampling design.", data.bootstrap_diagnostics, "BootstrapDiagnosticsTable");
addDataSheet("Bootstrap Replicates", "Bootstrap replicates", "Selected audit columns for all 760 coordinate replicates; full CSV remains in the package.", data.bootstrap_replicates, "BootstrapReplicatesTable");
addDataSheet("LOTO Level", "LOTO level transport", "Unrecentered transport from the training operating point to the held-out trajectory.", data.loto_level, "LotoLevelTable");
addDataSheet("LOTO Bootstrap", "LOTO bootstrap summary", "Training-trajectory 20 m moving-block uncertainty with held-out operating point fixed.", data.loto_bootstrap_summary, "LotoBootstrapSummaryTable");
addDataSheet("LOTO Replicates", "LOTO bootstrap replicates", "Selected audit columns for 300 train-only replicates in each direction.", data.loto_bootstrap_replicates, "LotoReplicatesTable");
addDataSheet("LOTO Shape", "LOTO ridge-shape comparison", "Recentered geometry test against raw and nuisance-profiled numerical ridges.", data.loto_shape, "LotoShapeTable");
addDataSheet("Hierarchy", "Hierarchical model comparison", "Equal-dimensional shared q-star and shared Cn models, with separate and pooled benchmarks.", data.hierarchy, "HierarchyTable");
addDataSheet("Residual ACF", "Residual autocorrelation", "Lag diagnostics motivating a conservative 20 m primary block length.", data.acf, "ResidualAcfTable");
const verificationSheet = addDataSheet("Verification", "Automated verification", "All scientific, identity, convergence, determinism, and output checks.", data.verification, "VerificationTable");
verificationSheet.getRange(`B4:B${data.verification.length + 3}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "TRUE", format: { fill: "#D1FAE5", font: { color: "#065F46", bold: true } } });
verificationSheet.getRange(`B4:B${data.verification.length + 3}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "FALSE", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } } });
addDataSheet("Sources", "Sources and lineage", "Package-relative sources; no external mutable inputs are used.", data.sources, "SourcesTable");

const overviewCheck = await workbook.inspect({ kind: "table", range: "Overview!A1:L31", include: "values,formulas", tableMaxRows: 31, tableMaxCols: 12 });
console.log(overviewCheck.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errors.ndjson);

await fs.mkdir(renderDir, { recursive: true });
for (const sheet of workbook.worksheets.items) {
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 0.8, format: "png" });
  const safe = sheet.name.replaceAll(" ", "_");
  await fs.writeFile(`${renderDir}/${safe}.png`, new Uint8Array(await preview.arrayBuffer()));
}
await fs.mkdir(new URL(".", `file://${outputPath}`).pathname, { recursive: true }).catch(() => {});
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: workbook.worksheets.items.map((sheet) => sheet.name) }));
