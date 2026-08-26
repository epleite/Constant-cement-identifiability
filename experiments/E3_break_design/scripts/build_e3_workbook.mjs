import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const packageRoot = process.env.E3_PACKAGE_ROOT;
const outputPath = process.env.E3_WORKBOOK_OUTPUT;
const renderDir = process.env.E3_WORKBOOK_RENDER_DIR;
if (!packageRoot || !outputPath || !renderDir) {
  throw new Error("E3_PACKAGE_ROOT, E3_WORKBOOK_OUTPUT, and E3_WORKBOOK_RENDER_DIR are required");
}

const resultsDir = path.join(packageRoot, "results");
const tablesDir = path.join(resultsDir, "tables");
const verificationDir = path.join(resultsDir, "verification");

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function coerce(value) {
  if (value === "") return null;
  if (value === "True" || value === "true") return true;
  if (value === "False" || value === "false") return false;
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

async function readCsv(fileName) {
  const rows = parseCsv(await fs.readFile(path.join(tablesDir, fileName), "utf8"));
  const headers = rows[0];
  return rows.slice(1).filter((row) => row.some((value) => value !== "")).map((row) =>
    Object.fromEntries(headers.map((header, index) => [header, coerce(row[index] ?? "")])),
  );
}

function columnLetter(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

const palette = {
  navy: "#17324D",
  blue: "#3478A6",
  teal: "#2A9D8F",
  gold: "#E9C46A",
  orange: "#F4A261",
  red: "#C94C4C",
  green: "#2E7D32",
  ink: "#1F2937",
  muted: "#667085",
  pale: "#EEF2F5",
  paleBlue: "#E8F1F7",
  paleTeal: "#E6F4F1",
  paleGold: "#FFF8E1",
  paleRed: "#FDECEC",
  line: "#D0D5DD",
  white: "#FFFFFF",
};

const summary = JSON.parse(await fs.readFile(path.join(resultsDir, "summary.json"), "utf8"));
const verification = JSON.parse(await fs.readFile(path.join(verificationDir, "E3_verification.json"), "utf8"));
const bestDesigns = await readCsv("E3_best_designs.csv");
const bootstrapSummary = await readCsv("E3_bootstrap_summary.csv");
const conditionalSelection = await readCsv("E3_conditional_design_selection.csv");
const operatingSensitivity = await readCsv("E3_operating_point_design_sensitivity.csv");
const referenceSensitivity = await readCsv("E3_reference_pressure_sensitivity.csv");
const trajectoryDesigns = await readCsv("E3_trajectory_specific_designs.csv");
const discrepancySensitivity = await readCsv("E3_model_discrepancy_sensitivity.csv");
const targetDiscrepancy = await readCsv("E3_target_aligned_discrepancy.csv");
const noGoAudit = await readCsv("E3_pressure_independence_audit.csv");
const noGoRepetition = await readCsv("E3_no_go_repetition.csv");
const multiFluid = await readCsv("E3_multi_fluid_control.csv");
const profileWidths = await readCsv("E3_profile_widths.csv");
const nonlinearProfiles = await readCsv("E3_nonlinear_profiles.csv");
const finiteDifference = await readCsv("E3_finite_difference_stability.csv");

const workbook = Workbook.create();
const sheetNames = [
  "Summary",
  "Fabric ablation",
  "Design",
  "Robustness",
  "Discrepancy controls",
  "Profiles",
  "Verification",
  "Source index",
];
for (const name of sheetNames) workbook.worksheets.add(name);

function titleBand(sheet, title, subtitle, lastColumn = "N") {
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: palette.navy,
    font: { bold: true, color: palette.white, size: 17 },
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: palette.pale,
    font: { italic: true, color: palette.ink, size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange("1:1").format.rowHeight = 30;
  sheet.getRange("2:2").format.rowHeight = 34;
  sheet.freezePanes.freezeRows(2);
}

function sectionBand(sheet, rangeAddress, label, fill = palette.blue, color = palette.white) {
  const range = sheet.getRange(rangeAddress);
  range.merge();
  range.values = [[label]];
  range.format = {
    fill,
    font: { bold: true, color, size: 11 },
    verticalAlignment: "center",
  };
}

function writeTable(sheet, startCell, headers, rows, tableName, options = {}) {
  const match = /^([A-Z]+)(\d+)$/.exec(startCell);
  if (!match) throw new Error(`Invalid start cell: ${startCell}`);
  const startColLetters = match[1];
  const startRow = Number(match[2]);
  let startCol = 0;
  for (const char of startColLetters) startCol = startCol * 26 + char.charCodeAt(0) - 64;
  startCol -= 1;
  const endCol = startCol + headers.length - 1;
  const endRow = startRow + rows.length;
  const endLetter = columnLetter(endCol);
  sheet.getRange(`${startCell}:${endLetter}${endRow}`).values = [headers, ...rows];
  const table = sheet.tables.add(`${startCell}:${endLetter}${endRow}`, true, tableName);
  table.style = options.style ?? "TableStyleMedium2";
  table.showBandedRows = true;
  sheet.getRange(`${startCell}:${endLetter}${startRow}`).format = {
    fill: options.headerFill ?? palette.blue,
    font: { bold: true, color: palette.white, size: 9 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: palette.navy },
  };
  if (rows.length) {
    sheet.getRange(`${columnLetter(startCol)}${startRow + 1}:${endLetter}${endRow}`).format = {
      font: { color: palette.ink, size: 9 },
      verticalAlignment: "top",
      borders: { insideHorizontal: { style: "thin", color: "#E4E7EC" } },
    };
  }
  return { table, startRow, endRow, startCol, endCol, endLetter };
}

function setWidths(sheet, widths) {
  for (const [column, widthPx] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidthPx = widthPx;
  }
}

function styleCard(sheet, rangeAddress, title, formulaOrValue, numberFormat, fill) {
  const match = /^([A-Z]+)(\d+):([A-Z]+)(\d+)$/.exec(rangeAddress);
  const topRow = Number(match[2]);
  const bottomRow = Number(match[4]);
  const left = match[1];
  const right = match[3];
  sheet.getRange(`${left}${topRow}:${right}${topRow}`).merge();
  sheet.getRange(`${left}${topRow}`).values = [[title]];
  sheet.getRange(`${left}${topRow}:${right}${topRow}`).format = {
    fill,
    font: { bold: true, color: palette.white, size: 10 },
    horizontalAlignment: "center",
  };
  sheet.getRange(`${left}${topRow + 1}:${right}${bottomRow}`).merge();
  if (typeof formulaOrValue === "string" && formulaOrValue.startsWith("=")) {
    sheet.getRange(`${left}${topRow + 1}`).formulas = [[formulaOrValue]];
  } else {
    sheet.getRange(`${left}${topRow + 1}`).values = [[formulaOrValue]];
  }
  sheet.getRange(`${left}${topRow + 1}:${right}${bottomRow}`).format = {
    fill: palette.white,
    font: { bold: true, color: palette.ink, size: 18 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: fill },
  };
  if (numberFormat) sheet.getRange(`${left}${topRow + 1}`).format.numberFormat = numberFormat;
}

function subset(records, predicate, headers) {
  return records.filter(predicate).map((record) => headers.map((header) => record[header] ?? null));
}

// Summary
{
  const sheet = workbook.worksheets.getItem("Summary");
  titleBand(
    sheet,
    "Constant-cement E3: nuisance-adjusted pressure design",
    "Prospective Break experiment. Primary result: expanded fabric nuisances, 39 MPa reference, and a finite 5–60 MPa candidate set.",
    "N",
  );
  styleCard(
    sheet,
    "A4:C8",
    "Selected added pressures",
    `${summary.primary_design.pressures_mpa[0].toFixed(1)} + ${summary.primary_design.pressures_mpa[1].toFixed(1)} MPa`,
    null,
    palette.blue,
  );
  sheet.getRange("A9:C9").merge();
  sheet.getRange("A9").values = [["Both are in addition to the 39 MPa reference"]];
  sheet.getRange("A9:C9").format = { font: { italic: true, color: palette.muted, size: 9 }, horizontalAlignment: "center" };
  styleCard(sheet, "E4:G7", "Adjusted λmin gain", "='Fabric ablation'!G9", '0.00"×"', palette.teal);
  styleCard(sheet, "I4:K7", "Worst-SD reduction", "='Fabric ablation'!H9", '0.00"×"', palette.orange);
  styleCard(sheet, "M4:N7", "Verification", "='Verification'!A5", null, palette.green);

  sectionBand(sheet, "A11:G11", "Primary assumptions");
  const assumptions = [
    ["Static model", summary.model.static, "—", "Frozen E1/RPIA forward model"],
    ["Pressure extension", summary.model.pressure_extension, "—", summary.model.status],
    ["Primary fabric mode", summary.model.primary_fabric_mode, "—", "Conservative Schur-adjusted analysis"],
    ["Reference pressure", summary.model.reference_pressure_mpa, "MPa", "Absolute reference state"],
    ["Per-state log-velocity σ", summary.primary_design.per_state_log_velocity_sigma, "fraction", "0.50% per absolute state"],
    ["Shared-reference error correlation", summary.primary_design.shared_reference_error_correlation, "correlation", "Two differences share the 39 MPa observation"],
    ["Trajectory discrepancy σ", summary.primary_design.trajectory_level_model_discrepancy_sigma, "fraction RMS", "Intercept + porosity + clay basis"],
  ];
  const a = writeTable(sheet, "A12", ["Assumption", "Value", "Unit", "Audit note"], assumptions, "SummaryAssumptions");
  sheet.getRange(`B13:B${a.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`A13:D${a.endRow}`).format.wrapText = true;
  sheet.getRange(`13:${a.endRow}`).format.rowHeight = 28;

  sectionBand(sheet, "I11:N11", "Local uncertainty before and after the design", palette.teal);
  writeTable(
    sheet,
    "I12",
    ["Quantity", "Static", "Post-design", "Change", "Unit", "Interpretation"],
    [
      ["SD Vcem", summary.operating_point.static_local_uncertainty.sd_Vcem_percentage_points, summary.primary_design.post_design_local_uncertainty.sd_Vcem_percentage_points, null, "percentage points", "Unconstrained local Gaussian"],
      ["SD ln Cn", summary.operating_point.static_local_uncertainty.sd_lnCn, summary.primary_design.post_design_local_uncertainty.sd_lnCn, null, "log units", "Unconstrained local Gaussian"],
      ["Multiplicative Cn factor", summary.operating_point.static_local_uncertainty.multiplicative_Cn_one_sigma, null, null, "factor", "Post factor is exp(post SD ln Cn)"],
      ["Corr(Vcem, ln Cn)", summary.operating_point.static_local_uncertainty.Vcem_lnCn_correlation, summary.primary_design.post_design_local_uncertainty.Vcem_lnCn_correlation, null, "correlation", "Near-perfect compensation remains"],
    ],
    "SummaryUncertainty",
    { headerFill: palette.teal },
  );
  sheet.getRange("L13:L14").formulas = [["=J13/K13"], ["=J14/K14"]];
  sheet.getRange("K15").formulas = [["=EXP(K14)"]];
  sheet.getRange("L15").formulas = [["=J15/K15"]];
  sheet.getRange("L16").formulas = [["=K16-J16"]];
  sheet.getRange("J13:L16").format.numberFormat = "0.000";

  sectionBand(sheet, "A21:N21", "Scientific interpretation", palette.gold, palette.ink);
  sheet.getRange("A22:N25").merge();
  sheet.getRange("A22").values = [[summary.scientific_conclusion]];
  sheet.getRange("A22:N25").format = {
    fill: palette.paleGold,
    font: { color: palette.ink, size: 10 },
    wrapText: true,
    verticalAlignment: "top",
    borders: { preset: "outside", style: "thin", color: palette.gold },
  };
  sectionBand(sheet, "A27:N27", "Decision boundary", palette.red);
  sheet.getRange("A28:N31").merge();
  sheet.getRange("A28").values = [["The 5 + 7.5 MPa pair is a candidate laboratory configuration, not a definitive acquisition. It touches the lower boundary of the tested pressure set, and the apparent gain collapses when compliant and stiff fabric variables and target-aligned model error can absorb the new direction."]];
  sheet.getRange("A28:N31").format = {
    fill: palette.paleRed,
    font: { color: palette.ink, size: 10 },
    wrapText: true,
    verticalAlignment: "top",
    borders: { preset: "outside", style: "thin", color: palette.red },
  };
  setWidths(sheet, { A: 210, B: 260, C: 100, D: 280, E: 85, F: 85, G: 95, H: 22, I: 165, J: 95, K: 95, L: 95, M: 115, N: 190 });
}

// Fabric ablation
{
  const sheet = workbook.worksheets.getItem("Fabric ablation");
  titleBand(sheet, "Fabric-link ablation", "All rows use the best two-state design at 0.50% per-state log-velocity uncertainty. Derived columns are live formulas.", "V");
  sheet.getRange("A3").values = [["Static baseline adjusted λmin"]];
  sheet.getRange("B3").values = [[summary.operating_point.baseline_adjusted_lambda_min]];
  sheet.getRange("B3").format.numberFormat = "0.000000";
  const primaryRows = ["shared", "fixed", "nuisance", "expanded_nuisance"].map((mode) => {
    const row = bestDesigns.find((record) => record.fabric_mode === mode && record.state_log_sigma === 0.005 && record.n_additional_pressures === 2);
    const assumptions = {
      shared: "Nominal and compliant Cn shared",
      fixed: "Compliant-contact Cn fixed (oracle)",
      nuisance: "Independent compliant-contact Cn with prior",
      expanded_nuisance: "Compliant Cn, stiff Cn, and critical porosity adjusted",
    };
    return [mode, assumptions[mode], row.lambda_min, row.lambda_max, null, null, null, null, null, null];
  });
  const tab = writeTable(sheet, "A5", ["Fabric mode", "Fabric assumption", "λmin", "λmax", "Spectral ratio", "Jacobian condition", "λmin gain", "Worst-SD reduction", "Gain retained vs shared", "Gain / expanded"], primaryRows, "FabricAblationTable", { headerFill: palette.blue });
  for (let row = 6; row <= 9; row += 1) {
    sheet.getRange(`E${row}`).formulas = [[`=C${row}/D${row}`]];
    sheet.getRange(`F${row}`).formulas = [[`=SQRT(D${row}/C${row})`]];
    sheet.getRange(`G${row}`).formulas = [[`=C${row}/$B$3`]];
    sheet.getRange(`H${row}`).formulas = [[`=SQRT(G${row})`]];
    sheet.getRange(`I${row}`).formulas = [[`=G${row}/$G$6`]];
    sheet.getRange(`J${row}`).formulas = [[`=G${row}/$G$9`]];
  }
  sheet.getRange("C6:H9").format.numberFormat = "0.0000";
  sheet.getRange("E6:E9").format.numberFormat = "0.000000";
  sheet.getRange("I6:I9").format.numberFormat = "0.0%";
  sheet.getRange("J6:J9").format.numberFormat = '0.0"×"';
  sheet.getRange("A9:J9").format = { fill: palette.paleRed, font: { bold: true, color: palette.ink, size: 9 }, borders: { preset: "outside", style: "medium", color: palette.red } };

  sectionBand(sheet, "A12:J12", "Interpretation", palette.gold, palette.ink);
  sheet.getRange("A13:J17").merge();
  sheet.getRange("A13").values = [["The strong 633.8× gain under shared fabric falls to 3.35× after expanding the nuisance space. The design therefore appears highly informative only when pressure-responsive fabric is linked to, fixed by, or tightly regularized around the nominal target parameters. The expanded mode is the primary conservative result."]];
  sheet.getRange("A13:J17").format = { fill: palette.paleGold, font: { color: palette.ink, size: 10 }, wrapText: true, verticalAlignment: "top", borders: { preset: "outside", style: "thin", color: palette.gold } };

  sheet.getRange("L4:M4").values = [["Fabric mode", "λmin gain"]];
  sheet.getRange("L5:L8").formulas = [["=A6"], ["=A7"], ["=A8"], ["=A9"]];
  sheet.getRange("M5:M8").formulas = [["=G6"], ["=G7"], ["=G8"], ["=G9"]];
  const chart = sheet.charts.add("bar", sheet.getRange("L4:M8"));
  chart.title = "Fabric assumptions dominate apparent λmin gain";
  chart.titleTextStyle.fontSize = 12;
  chart.hasLegend = false;
  chart.yAxis = { numberFormatCode: '0"×"', min: 0 };
  chart.setPosition("L10", "V27");
  setWidths(sheet, { A: 150, B: 300, C: 90, D: 90, E: 105, F: 110, G: 95, H: 115, I: 125, J: 105, K: 24, L: 155, M: 110, N: 90, O: 90, P: 90, Q: 90, R: 90, S: 90, T: 90, U: 90, V: 90 });
}

// Design
{
  const sheet = workbook.worksheets.getItem("Design");
  titleBand(sheet, "Finite pressure design", "E-optimal search over candidate pressures from 5 to 60 MPa; the 39 MPa state is the absolute reference.", "T");
  sectionBand(sheet, "A4:G4", "Primary design");
  const primaryDesignRows = [
    ["Reference pressure", summary.model.reference_pressure_mpa, "MPa", "Absolute state"],
    ["Candidate minimum", summary.primary_design.candidate_pressure_range_mpa[0], "MPa", "Finite search boundary"],
    ["Added pressure 1", summary.primary_design.pressures_mpa[0], "MPa", "Best pair"],
    ["Added pressure 2", summary.primary_design.pressures_mpa[1], "MPa", "Best pair"],
    ["Absolute state count", null, "states", "1 + number of added pressures"],
    ["Pressure span", null, "MPa", "Max absolute pressure − min absolute pressure"],
    ["Touches candidate boundary", summary.primary_design.touches_candidate_boundary, "boolean", "5 MPa is the lower tested boundary"],
    ["Pair / single λmin", summary.primary_design.best_single_state_comparator.pair_over_single_lambda_min_ratio, "ratio", "Second state adds 29.9% to λmin"],
  ];
  writeTable(sheet, "A5", ["Metric", "Value", "Unit", "Audit note"], primaryDesignRows, "PrimaryDesignTable");
  sheet.getRange("B10").formulas = [["=1+2"]];
  sheet.getRange("B11").formulas = [["=MAX(B6,B8,B9)-MIN(B6,B8,B9)"]];
  sheet.getRange("B5:B12").format.numberFormat = "0.000";

  const expandedRef = referenceSensitivity.filter((row) => row.fabric_mode === "expanded_nuisance");
  const refPressures = [...new Set(expandedRef.map((row) => row.reference_pressure_mpa))].sort((a, b) => a - b);
  const refRows = refPressures.map((pressure) => {
    const single = expandedRef.find((row) => row.reference_pressure_mpa === pressure && row.n_additional_pressures === 1);
    const pair = expandedRef.find((row) => row.reference_pressure_mpa === pressure && row.n_additional_pressures === 2);
    return [pressure, single.lambda_min_gain, pair.lambda_min_gain, pair.pressure_1_mpa, pair.pressure_2_mpa];
  });
  writeTable(sheet, "J4", ["Reference (MPa)", "Best single gain", "Best pair gain", "Pair P1", "Pair P2"], refRows, "ReferenceSensitivityTable", { headerFill: palette.teal });
  sheet.getRange(`J5:N${4 + refRows.length}`).format.numberFormat = "0.000";
  const refChart = sheet.charts.add("line", sheet.getRange(`J4:L${4 + refRows.length}`));
  refChart.title = "Reference pressure changes the useful contrast";
  refChart.titleTextStyle.fontSize = 12;
  refChart.hasLegend = true;
  refChart.yAxis = { numberFormatCode: '0.0"×"', min: 0 };
  refChart.xAxis = { axisType: "textAxis" };
  refChart.setPosition("J11", "T25");

  sectionBand(sheet, "A15:P15", "Best designs at the primary 0.50% per-state precision");
  const designHeaders = ["fabric_mode", "n_additional_pressures", "pressure_1_mpa", "pressure_2_mpa", "pressure_span_mpa", "lambda_min", "lambda_max", "spectral_ratio", "condition_number", "lambda_min_gain", "worst_sd_reduction", "determinant_gain", "strong_direction_rotation_deg"];
  const designRows = subset(bestDesigns, (row) => row.state_log_sigma === 0.005, designHeaders);
  const d = writeTable(sheet, "A16", designHeaders, designRows, "BestDesignsPrimaryTable");
  sheet.getRange(`C17:M${d.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`H17:H${d.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, `A${d.endRow + 3}:Q${d.endRow + 3}`, "Trajectory-specific design control", palette.teal);
  const trajectoryHeaders = ["trajectory", "n_samples", "fabric_mode", "Vcem_fraction", "Cn", "pressure_1_mpa", "pressure_2_mpa", "baseline_lambda_min", "lambda_min", "lambda_max", "spectral_ratio", "condition_number", "lambda_min_gain", "worst_sd_reduction", "strong_direction_rotation_deg"];
  const tStart = d.endRow + 4;
  const t = writeTable(sheet, `A${tStart}`, trajectoryHeaders, subset(trajectoryDesigns, (row) => row.fabric_mode === "expanded_nuisance", trajectoryHeaders), "TrajectoryDesignTable", { headerFill: palette.teal });
  sheet.getRange(`D${tStart + 1}:O${t.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`K${tStart + 1}:K${t.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, `A${t.endRow + 3}:P${t.endRow + 3}`, "Representative operating-point sensitivity", palette.orange, palette.ink);
  const opHeaders = ["E2_Vcem_quantile", "E2_replicate", "Vcem_fraction", "Cn", "pressure_1_mpa", "pressure_2_mpa", "baseline_lambda_min", "lambda_min", "lambda_max", "spectral_ratio", "condition_number", "lambda_min_gain", "worst_sd_reduction", "strong_direction_rotation_deg"];
  const oStart = t.endRow + 4;
  const o = writeTable(sheet, `A${oStart}`, opHeaders, subset(operatingSensitivity, () => true, opHeaders), "OperatingSensitivityTable", { headerFill: palette.orange });
  sheet.getRange(`C${oStart + 1}:N${o.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`J${oStart + 1}:J${o.endRow}`).format.numberFormat = "0.000000";
  setWidths(sheet, { A: 145, B: 115, C: 100, D: 100, E: 105, F: 105, G: 105, H: 105, I: 110, J: 115, K: 105, L: 110, M: 120, N: 105, O: 120, P: 115, Q: 100, R: 95, S: 95, T: 95 });
}

// Robustness
{
  const sheet = workbook.worksheets.getItem("Robustness");
  titleBand(sheet, "Robustness and conditional design stability", "Trajectory-stratified 20 m moving-block bootstrap and sensitivity to operating point and block length.", "T");
  sectionBand(sheet, "A4:J4", "Bootstrap summary (400 replicates per fabric mode/design)");
  const bHeaders = ["fabric_mode", "design", "pressures_mpa", "metric", "median", "q025", "q975", "minimum", "maximum", "n"];
  const b = writeTable(sheet, "A5", bHeaders, subset(bootstrapSummary, () => true, bHeaders), "BootstrapSummaryTable");
  sheet.getRange(`E6:I${b.endRow}`).format.numberFormat = "0.0000";

  const helperModes = ["shared", "nuisance", "expanded_nuisance"];
  sheet.getRange("L4:M4").values = [["Fabric mode", "Median worst-SD reduction"]];
  helperModes.forEach((mode, index) => {
    const sourceRow = 6 + bootstrapSummary.findIndex((record) => record.fabric_mode === mode && record.design === "best_pair" && record.metric === "worst_sd_reduction");
    sheet.getRange(`L${5 + index}`).values = [[mode]];
    sheet.getRange(`M${5 + index}`).formulas = [[`=E${sourceRow}`]];
  });
  const bootChart = sheet.charts.add("bar", sheet.getRange("L4:M7"));
  bootChart.title = "Expanded fabric leaves only ~1.84× SD reduction";
  bootChart.titleTextStyle.fontSize = 12;
  bootChart.hasLegend = false;
  bootChart.yAxis = { numberFormatCode: '0.0"×"', min: 0 };
  bootChart.setPosition("L10", "T25");

  const condStart = b.endRow + 3;
  sectionBand(sheet, `A${condStart}:I${condStart}`, "Conditional re-optimization by block length", palette.teal);
  const condRows = Object.entries(summary.conditional_design_selection).map(([key, value]) => [
    Number(key.split("_")[0]),
    Number(key.split("_")[0]) * 4,
    value.n,
    value.full_sample_pair_selection_frequency,
    value.median_regret_fraction,
    value.max_regret_fraction,
  ]);
  const c = writeTable(sheet, `A${condStart + 1}`, ["Block length (samples)", "Approx. block length (m)", "Replicates", "Full-pair selection frequency", "Median regret", "Maximum regret"], condRows, "ConditionalSelectionSummary", { headerFill: palette.teal });
  sheet.getRange(`D${condStart + 2}:F${c.endRow}`).format.numberFormat = "0.0%";

  sectionBand(sheet, `A${c.endRow + 3}:Q${c.endRow + 3}`, "Reference-pressure and trajectory controls", palette.orange, palette.ink);
  const controlRows = summary.trajectory_specific_primary.map((row) => ["trajectory", row.trajectory, row.n_samples, row.Vcem_fraction, row.Cn, row.pressure_1_mpa, row.pressure_2_mpa, row.lambda_min_gain, row.worst_sd_reduction, row.spectral_ratio]);
  controlRows.push(["operating-point ensemble", "9 representative E2 points", summary.operating_point_design_sensitivity.n_representative_E2_bootstrap_points, null, null, 5, 7.5, summary.operating_point_design_sensitivity.lambda_min_gain_range[0], summary.operating_point_design_sensitivity.lambda_min_gain_range[1], summary.operating_point_design_sensitivity.full_sample_pair_selection_frequency]);
  const rStart = c.endRow + 4;
  const rc = writeTable(sheet, `A${rStart}`, ["Control", "State", "n", "Vcem fraction", "Cn", "P1 (MPa)", "P2 (MPa)", "Gain / low", "Worst-SD / high", "Spectral ratio / selection freq."], controlRows, "RobustnessControlsTable", { headerFill: palette.orange });
  sheet.getRange(`D${rStart + 1}:J${rc.endRow}`).format.numberFormat = "0.0000";

  sectionBand(sheet, `A${rc.endRow + 3}:J${rc.endRow + 3}`, "Interpretation", palette.gold, palette.ink);
  sheet.getRange(`A${rc.endRow + 4}:J${rc.endRow + 7}`).merge();
  sheet.getRange(`A${rc.endRow + 4}`).values = [["The selected pair is stable under conditional re-optimization and across the representative E2 operating points that remain inside the convex validity domain. That stability does not remove the model-form limitation: the bootstrap is conditional on two Hugin trajectories and does not recalibrate the prospective pressure extension."]];
  sheet.getRange(`A${rc.endRow + 4}:J${rc.endRow + 7}`).format = { fill: palette.paleGold, wrapText: true, verticalAlignment: "top", font: { color: palette.ink, size: 10 }, borders: { preset: "outside", style: "thin", color: palette.gold } };
  setWidths(sheet, { A: 155, B: 120, C: 115, D: 170, E: 105, F: 105, G: 105, H: 105, I: 105, J: 115, K: 24, L: 170, M: 135, N: 90, O: 90, P: 90, Q: 90, R: 90, S: 90, T: 90 });
}

// Discrepancy and controls
{
  const sheet = workbook.worksheets.getItem("Discrepancy controls");
  titleBand(sheet, "Model discrepancy and falsification controls", "The design must add a direction that survives both nuisance adjustment and plausible model-error directions.", "T");
  const primaryBasis = discrepancySensitivity.filter((row) => row.basis === "intercept_plus_porosity_plus_clay");
  const discHeaders = ["trajectory_discrepancy_percent", "n_basis_terms", "lambda_min", "lambda_max", "spectral_ratio", "condition_number", "lambda_min_gain", "worst_sd_reduction", "strong_direction_rotation_deg"];
  sectionBand(sheet, "A4:I4", "Generic trajectory discrepancy: full three-term basis");
  const d = writeTable(sheet, "A5", discHeaders, subset(primaryBasis, () => true, discHeaders), "GenericDiscrepancyTable");
  sheet.getRange(`A6:I${d.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`E6:E${d.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, "K4:T4", "Target-aligned discrepancy", palette.teal);
  const targetHeaders = ["target_aligned_discrepancy_percent_rms", "raw_alignment_rms_per_scaled_parameter", "lambda_min", "lambda_max", "spectral_ratio", "condition_number", "lambda_min_gain", "worst_sd_reduction", "strong_direction_rotation_deg"];
  const td = writeTable(sheet, "K5", targetHeaders, subset(targetDiscrepancy, (row) => row.fabric_mode === "expanded_nuisance", targetHeaders), "TargetDiscrepancyTable", { headerFill: palette.teal });
  sheet.getRange(`K6:S${td.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`O6:O${td.endRow}`).format.numberFormat = "0.000000";

  sheet.getRange("K13:L13").values = [["Target-aligned RMS (%)", "λmin gain"]];
  for (let i = 0; i < td.endRow - 5; i += 1) {
    sheet.getRange(`K${14 + i}`).formulas = [[`=K${6 + i}`]];
    sheet.getRange(`L${14 + i}`).formulas = [[`=Q${6 + i}`]];
  }
  const targetChart = sheet.charts.add("line", sheet.getRange(`K13:L${13 + (td.endRow - 5)}`));
  targetChart.title = "A target-aligned error erases the nominal gain";
  targetChart.titleTextStyle.fontSize = 12;
  targetChart.hasLegend = false;
  targetChart.yAxis = { numberFormatCode: '0.0"×"', min: 0 };
  targetChart.xAxis = { axisType: "textAxis" };
  targetChart.setPosition("K20", "T35");

  const basisRows = Object.entries(summary.model_discrepancy_basis_ablation_at_1_percent).map(([basis, value]) => [basis, value.n_basis_terms, value.lambda_min_gain, value.worst_sd_reduction, value.spectral_ratio]);
  sectionBand(sheet, "A14:E14", "Basis ablation at 1% RMS", palette.orange, palette.ink);
  const ba = writeTable(sheet, "A15", ["Basis", "Terms", "λmin gain", "Worst-SD reduction", "Spectral ratio"], basisRows, "BasisAblationTable", { headerFill: palette.orange });
  sheet.getRange(`C16:E${ba.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, "A22:E22", "Exact no-go audit", palette.red);
  const noGoHeaders = ["pressure_mpa", "max_abs_delta_Vp_mps", "max_abs_delta_Vs_mps", "max_abs_delta_rho_gcc", "differential_target_jacobian_norm"];
  const ng = writeTable(sheet, "A23", noGoHeaders, subset(noGoAudit, () => true, noGoHeaders), "NoGoAuditTable", { headerFill: palette.red });
  sheet.getRange(`A24:E${ng.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, `A${ng.endRow + 3}:I${ng.endRow + 3}`, "Replication cannot rotate the raw target direction", palette.red);
  const repHeaders = ["n_identical_static_states", "lambda_min", "lambda_max", "lambda_min_gain", "spectral_ratio", "raw_strong_direction_rotation_deg", "target_rank", "new_sensitivity_direction"];
  const nrStart = ng.endRow + 4;
  const nr = writeTable(sheet, `A${nrStart}`, repHeaders, subset(noGoRepetition, () => true, repHeaders), "NoGoRepetitionTable", { headerFill: palette.red });
  sheet.getRange(`B${nrStart + 1}:F${nr.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, `A${nr.endRow + 3}:F${nr.endRow + 3}`, "Multi-fluid comparison control", palette.teal);
  const fluidHeaders = ["design", "n_additional_states", "lambda_min", "lambda_min_gain", "spectral_ratio", "strong_direction_rotation_deg"];
  const mfStart = nr.endRow + 4;
  const mf = writeTable(sheet, `A${mfStart}`, fluidHeaders, subset(multiFluid, () => true, fluidHeaders), "MultiFluidTable", { headerFill: palette.teal });
  sheet.getRange(`C${mfStart + 1}:F${mf.endRow}`).format.numberFormat = "0.000000";

  sectionBand(sheet, `A${mf.endRow + 3}:J${mf.endRow + 3}`, "Bounding-average validity", palette.gold, palette.ink);
  const bwStart = mf.endRow + 4;
  const weight = summary.controls.bounding_weight_stress;
  writeTable(sheet, `A${bwStart}`, ["Domain", "W_K min", "W_K max", "W_G min", "W_G max", "Maximum outside [0,1]"], [
    ["Pooled operating point + 1σ local nuisance scenarios", summary.controls.patchy_weight_ranges.W_K[0], summary.controls.patchy_weight_ranges.W_K[1], summary.controls.patchy_weight_ranges.W_G[0], summary.controls.patchy_weight_ranges.W_G[1], weight.pooled_state_maximum_fraction_outside_unit_interval],
    ["Wider E2 bootstrap stress test", weight.overall_W_K_range[0], weight.overall_W_K_range[1], weight.overall_W_G_range[0], weight.overall_W_G_range[1], weight.maximum_fraction_outside_unit_interval],
  ], "BoundingValidityTable", { headerFill: palette.gold });
  sheet.getRange(`B${bwStart + 1}:E${bwStart + 2}`).format.numberFormat = "0.000";
  sheet.getRange(`F${bwStart + 1}:F${bwStart + 2}`).format.numberFormat = "0.0%";
  sheet.getRange(`H${bwStart}:T${bwStart + 3}`).merge();
  sheet.getRange(`H${bwStart}`).values = [["Important distinction: target-aligned discrepancy = 0% still retains the primary generic 1%-RMS intercept + porosity + clay discrepancy. It is not the same experiment as generic discrepancy = 0%."]];
  sheet.getRange(`H${bwStart}:T${bwStart + 3}`).format = { fill: palette.paleGold, wrapText: true, verticalAlignment: "top", font: { color: palette.ink, size: 10 }, borders: { preset: "outside", style: "thin", color: palette.gold } };
  setWidths(sheet, { A: 175, B: 120, C: 110, D: 120, E: 120, F: 115, G: 115, H: 125, I: 125, J: 24, K: 140, L: 125, M: 110, N: 110, O: 110, P: 110, Q: 110, R: 110, S: 120, T: 105 });
}

// Profiles
{
  const sheet = workbook.worksheets.getItem("Profiles");
  titleBand(sheet, "Finite ridge profiles", "Nonlinear target grid with the nuisance tangent space frozen at the operating point. Objective values are dimensionless ΔΦ.", "T");
  sectionBand(sheet, "A4:N4", "Support widths at ΔΦ = 2.30 and 5.99");
  const widthRows = profileWidths.map((row) => [
    row.objective,
    row.threshold,
    row.Vcem_min_fraction,
    row.Vcem_max_fraction,
    null,
    row.Vcem_lower_censored,
    row.Vcem_upper_censored,
    row.Cn_min,
    row.Cn_max,
    null,
    row.Cn_lower_censored,
    row.Cn_upper_censored,
    row.truth_Vcem_fraction,
    row.truth_Cn,
  ]);
  const pw = writeTable(sheet, "A5", ["Objective", "Threshold ΔΦ", "Vcem min", "Vcem max", "Vcem width (pp; ≥ if censored)", "Vcem lower censored", "Vcem upper censored", "Cn min", "Cn max", "Cn width (≥ if censored)", "Cn lower censored", "Cn upper censored", "Truth Vcem", "Truth Cn"], widthRows, "ProfileWidthsTable");
  for (let row = 6; row <= pw.endRow; row += 1) {
    sheet.getRange(`E${row}`).formulas = [[`=(D${row}-C${row})*100`]];
    sheet.getRange(`J${row}`).formulas = [[`=I${row}-H${row}`]];
  }
  sheet.getRange(`B6:E${pw.endRow}`).format.numberFormat = "0.000";
  sheet.getRange(`H6:J${pw.endRow}`).format.numberFormat = "0.000";
  sheet.getRange(`M6:N${pw.endRow}`).format.numberFormat = "0.0000";
  sheet.getRange(`F6:G${pw.endRow}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "TRUE", format: { fill: palette.paleRed, font: { color: palette.red, bold: true } } });
  sheet.getRange(`K6:L${pw.endRow}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "TRUE", format: { fill: palette.paleRed, font: { color: palette.red, bold: true } } });

  const uniqueV = [...new Set(nonlinearProfiles.map((row) => row.Vcem_percent))].sort((a, b) => a - b);
  const uniqueCn = [...new Set(nonlinearProfiles.map((row) => row.Cn))].sort((a, b) => a - b);
  const vProfileRows = uniqueV.map((value) => {
    const rows = nonlinearProfiles.filter((row) => row.Vcem_percent === value);
    return [value.toFixed(2), Math.min(...rows.map((row) => row.static_adjusted_objective)), Math.min(...rows.map((row) => row.combined_adjusted_objective)), 2.3];
  });
  const cnProfileRows = uniqueCn.map((value) => {
    const rows = nonlinearProfiles.filter((row) => row.Cn === value);
    return [value.toFixed(2), Math.min(...rows.map((row) => row.static_adjusted_objective)), Math.min(...rows.map((row) => row.combined_adjusted_objective)), 2.3];
  });
  const profileStart = pw.endRow + 3;
  sectionBand(sheet, `A${profileStart}:D${profileStart}`, "Adjusted 1D profile minima over Cn");
  const vp = writeTable(sheet, `A${profileStart + 1}`, ["Vcem (%)", "Static adjusted ΔΦ", "Combined adjusted ΔΦ", "Threshold 2.30"], vProfileRows, "VcemProfileTable");
  sheet.getRange(`A${profileStart + 2}:D${vp.endRow}`).format.numberFormat = "0.000";
  sectionBand(sheet, `F${profileStart}:I${profileStart}`, "Adjusted 1D profile minima over Vcem", palette.teal);
  const cp = writeTable(sheet, `F${profileStart + 1}`, ["Cn", "Static adjusted ΔΦ", "Combined adjusted ΔΦ", "Threshold 2.30"], cnProfileRows, "CnProfileTable", { headerFill: palette.teal });
  sheet.getRange(`F${profileStart + 2}:I${cp.endRow}`).format.numberFormat = "0.000";

  const vChart = sheet.charts.add("line", sheet.getRange(`A${profileStart + 1}:D${vp.endRow}`));
  vChart.title = "No Vcem contraction is resolved on the grid";
  vChart.titleTextStyle.fontSize = 12;
  vChart.hasLegend = true;
  vChart.yAxis = { numberFormatCode: "0.0", min: 0 };
  vChart.xAxis = { axisType: "textAxis" };
  vChart.setPosition(`K${profileStart + 1}`, `T${profileStart + 17}`);
  const cChart = sheet.charts.add("line", sheet.getRange(`F${profileStart + 1}:I${cp.endRow}`));
  cChart.title = "Cn contraction remains boundary-censored";
  cChart.titleTextStyle.fontSize = 12;
  cChart.hasLegend = true;
  cChart.yAxis = { numberFormatCode: "0.0", min: 0 };
  cChart.xAxis = { axisType: "textAxis" };
  cChart.setPosition(`K${profileStart + 20}`, `T${profileStart + 36}`);

  sheet.getRange(`A${Math.max(vp.endRow, cp.endRow) + 3}:I${Math.max(vp.endRow, cp.endRow) + 6}`).merge();
  sheet.getRange(`A${Math.max(vp.endRow, cp.endRow) + 3}`).values = [["Censored widths are lower bounds. At ΔΦ = 2.30, adjusted Vcem support spans at least 3.300 percentage points for both static and combined objectives; adjusted Cn support spans at least 6.826 (static) and 6.460 (combined). The profiles are local-linear nuisance diagnostics, not fully nonlinear nuisance re-optimizations."]];
  sheet.getRange(`A${Math.max(vp.endRow, cp.endRow) + 3}:I${Math.max(vp.endRow, cp.endRow) + 6}`).format = { fill: palette.paleGold, wrapText: true, verticalAlignment: "top", font: { color: palette.ink, size: 10 }, borders: { preset: "outside", style: "thin", color: palette.gold } };
  setWidths(sheet, { A: 175, B: 110, C: 115, D: 115, E: 145, F: 130, G: 130, H: 105, I: 105, J: 145, K: 125, L: 125, M: 105, N: 105, O: 24, P: 90, Q: 90, R: 90, S: 90, T: 90 });
}

// Verification
{
  const sheet = workbook.worksheets.getItem("Verification");
  titleBand(sheet, "Automated verification and numerical checks", "Machine-readable E3 verification plus finite-difference stability of target and nuisance Jacobians.", "K");
  sheet.getRange("A4").values = [["Status"]];
  sheet.getRange("B4").values = [["Passed"]];
  sheet.getRange("C4").values = [["Failed"]];
  sheet.getRange("D4").values = [["Total"]];
  sheet.getRange("A5").formulas = [["=IF(C5=0,\"PASS\",\"FAIL\")"]];
  sheet.getRange("B5").formulas = [["=SUM(E10:E115)"]];
  sheet.getRange("D5").formulas = [["=MAX(A10:A115)"]];
  sheet.getRange("C5").formulas = [["=D5-B5"]];
  sheet.getRange("A4:D4").format = { fill: palette.navy, font: { bold: true, color: palette.white }, horizontalAlignment: "center" };
  sheet.getRange("A5:D5").format = { fill: palette.paleTeal, font: { bold: true, color: palette.green, size: 13 }, horizontalAlignment: "center", borders: { preset: "outside", style: "thin", color: palette.green } };

  sectionBand(sheet, "F4:K4", "Finite-difference stability", palette.teal);
  const fdHeaders = ["step", "target_relative_error_to_1e-4", "nuisance_relative_error_to_1e-4", "target_norm", "nuisance_norm", "n_nuisances"];
  const fd = writeTable(sheet, "F5", fdHeaders, subset(finiteDifference, () => true, fdHeaders), "FiniteDifferenceTable", { headerFill: palette.teal });
  sheet.getRange(`F6:K${fd.endRow}`).format.numberFormat = "0.000000E+00";
  sheet.getRange("F10:H10").values = [["Metric", "Maximum relative error", "Tolerance"]];
  sheet.getRange("F11:F12").values = [["Target Jacobian"], ["Nuisance Jacobian"]];
  sheet.getRange("G11").formulas = [["=MAX(G6:G8)"]];
  sheet.getRange("G12").formulas = [["=MAX(H6:H8)"]];
  sheet.getRange("H11:H12").values = [[1e-5], [1e-5]];
  sheet.getRange("G11:H12").format.numberFormat = "0.000E+00";

  sectionBand(sheet, "A8:E8", "Verification checks");
  const checks = verification.checks.map((check, index) => [index + 1, check.passed, check.name, check.detail || "", check.passed ? 1 : 0]);
  const vc = writeTable(sheet, "A9", ["ID", "Passed", "Check", "Detail", "Pass flag"], checks, "VerificationChecksTable");
  sheet.getRange(`B10:B${vc.endRow}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "TRUE", format: { fill: "#D1FAE5", font: { color: "#065F46", bold: true } } });
  sheet.getRange(`B10:B${vc.endRow}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "FALSE", format: { fill: "#FEE2E2", font: { color: "#991B1B", bold: true } } });
  sheet.freezePanes.freezeRows(9);
  setWidths(sheet, { A: 55, B: 80, C: 340, D: 300, E: 80, F: 115, G: 160, H: 175, I: 110, J: 115, K: 105 });
}

// Source index
{
  const sheet = workbook.worksheets.getItem("Source index");
  titleBand(sheet, "Sources, lineage, and package index", "All sources are immutable package-relative files. Full raw result tables remain in results/tables; this workbook presents curated scientific views.", "H");
  const destinationMap = {
    "E3_best_designs.csv": "Fabric ablation; Design",
    "E3_bootstrap_summary.csv": "Robustness",
    "E3_conditional_design_selection.csv": "Robustness (summary)",
    "E3_operating_point_design_sensitivity.csv": "Design; Robustness",
    "E3_reference_pressure_sensitivity.csv": "Design",
    "E3_trajectory_specific_designs.csv": "Design; Robustness",
    "E3_model_discrepancy_sensitivity.csv": "Discrepancy controls",
    "E3_target_aligned_discrepancy.csv": "Discrepancy controls",
    "E3_pressure_independence_audit.csv": "Discrepancy controls",
    "E3_no_go_repetition.csv": "Discrepancy controls",
    "E3_multi_fluid_control.csv": "Discrepancy controls",
    "E3_profile_widths.csv": "Profiles",
    "E3_nonlinear_profiles.csv": "Profiles (derived 1D minima)",
    "E3_finite_difference_stability.csv": "Verification",
  };
  const csvNames = (await fs.readdir(tablesDir)).filter((name) => name.endsWith(".csv")).sort();
  const sourceRows = [];
  for (const fileName of csvNames) {
    const records = await readCsv(fileName);
    sourceRows.push([fileName, `results/tables/${fileName}`, "CSV", records.length, destinationMap[fileName] ?? "Indexed only; see full package table", destinationMap[fileName] ? "Curated or summarized" : "Not duplicated in workbook"]);
  }
  sourceRows.unshift(
    ["summary.json", "results/summary.json", "JSON", 1, "Summary and cross-sheet assumptions", "Headline machine-readable result"],
    ["RESULTS.md", "results/RESULTS.md", "Markdown", null, "Interpretation reference", "Narrative result statement"],
    ["E3_verification.json", "results/verification/E3_verification.json", "JSON", verification.n_checks, "Verification", "Full automated check set"],
    ["METHODS.md", "docs/METHODS.md", "Markdown", null, "Method reference", "Definitions and computation"],
    ["SOURCES.md", "docs/SOURCES.md", "Markdown", null, "Citation reference", "Scientific sources"],
    ["VERSION", "VERSION", "Text", null, "Package metadata", "Frozen package version"],
  );
  const src = writeTable(sheet, "A4", ["Artifact", "Package-relative path", "Type", "Records / checks", "Workbook destination", "Scope note"], sourceRows, "SourceIndexTable");
  sheet.getRange(`D5:D${src.endRow}`).format.numberFormat = "#,##0";

  const hashStart = src.endRow + 3;
  sectionBand(sheet, `A${hashStart}:D${hashStart}`, "Upstream provenance anchors", palette.teal);
  const hashes = Object.entries(summary.provenance).map(([artifact, sha]) => [artifact, sha, "SHA-256", "Upstream E1/E2/core input anchor; not an E3 output-file hash"]);
  const ht = writeTable(sheet, `A${hashStart + 1}`, ["Artifact key", "Digest", "Algorithm", "Scope"], hashes, "ProvenanceHashesTable", { headerFill: palette.teal });
  sheet.getRange(`B${hashStart + 2}:B${ht.endRow}`).format.numberFormat = "@";

  const noteStart = ht.endRow + 3;
  sectionBand(sheet, `A${noteStart}:H${noteStart}`, "Units and formula conventions", palette.gold, palette.ink);
  sheet.getRange(`A${noteStart + 1}:H${noteStart + 6}`).merge();
  sheet.getRange(`A${noteStart + 1}`).values = [["Vcem fractions are converted to volume percent by multiplying by 100; Vcem uncertainty and profile width are reported in percentage points. Spectral ratio = λmin/λmax. Jacobian condition = sqrt(λmax/λmin). λmin gain = λmin/baseline λmin. Worst-SD reduction = sqrt(λmin gain). Multiplicative Cn uncertainty is exp(SD ln Cn). Eigen-information metrics are dimensionless after whitening and target scaling (0.015 for Vcem and 0.20 for ln Cn)."]];
  sheet.getRange(`A${noteStart + 1}:H${noteStart + 6}`).format = { fill: palette.paleGold, wrapText: true, verticalAlignment: "top", font: { color: palette.ink, size: 10 }, borders: { preset: "outside", style: "thin", color: palette.gold } };
  setWidths(sheet, { A: 220, B: 350, C: 95, D: 110, E: 240, F: 210, G: 100, H: 100 });
}

// Compact QA and render pass.
const keyInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:N31",
  include: "values,formulas",
  tableMaxRows: 31,
  tableMaxCols: 14,
  maxChars: 15000,
});
console.log(keyInspection.ndjson);

const formulaInspection = await workbook.inspect({
  kind: "formula",
  maxChars: 20000,
  options: { maxResults: 400 },
});
console.log(formulaInspection.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 500 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

await fs.mkdir(renderDir, { recursive: true });
const renderFiles = [];
for (const sheet of workbook.worksheets.items) {
  const preview = await workbook.render({ sheetName: sheet.name, autoCrop: "all", scale: 0.85, format: "png" });
  const safe = sheet.name.replaceAll(" ", "_");
  const previewPath = path.join(renderDir, `${safe}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  renderFiles.push(previewPath);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const qaReport = {
  outputPath,
  sheetNames,
  renders: renderFiles,
  verificationSource: { status: verification.status, passed: verification.n_passed, total: verification.n_checks },
  headlineReconciliation: {
    primaryFabricMode: summary.model.primary_fabric_mode,
    referencePressureMpa: summary.model.reference_pressure_mpa,
    addedPressuresMpa: summary.primary_design.pressures_mpa,
    lambdaMinGain: summary.primary_design.lambda_min_gain,
    bootstrapMedianGain: summary.bootstrap_primary.expanded_nuisance.lambda_min_gain.median,
  },
  formulaErrorScan: errors.ndjson,
};
await fs.writeFile(path.join(renderDir, "qa_report.json"), JSON.stringify(qaReport, null, 2));
console.log(JSON.stringify(qaReport));
