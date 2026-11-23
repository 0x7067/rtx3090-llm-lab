/** Literal model-output parsing only. No DOM repair, row padding or value repair. */
export type LiteralTableCell = {
  text: string;
  start: number;
  end: number;
  row: number;
  column: number;
  rowSpan: number;
  columnSpan: number;
};
export type LiteralTableRow = {
  start: number;
  end: number;
  /** Expanded slots refer to their actual origin; no cell geometry is inferred. */
  slots: LiteralTableCell[];
};
export type LiteralTable = {
  format: "otsl" | "html";
  rows: LiteralTableRow[];
};
const maxRows = 128;
const maxColumns = 64;
const maxSlots = 4096;
const maxCharacters = 10_000;

/** Reserved by the trusted worker for the fixed Table Recognition: task. */
export const isTableRecognitionBlock = (id: string) =>
  /^table-b[1-9][0-9]*$/.test(id);

/** Markup is never silently flattened through the ordinary line heuristics. */
export const hasTableMarkup = (text: string) =>
  /<(?:fcel|ecel|lcel|ucel|xcel|nl|\/?(?:table|thead|tbody|tfoot|tr|td|th))\b/i.test(
    text,
  );

function otsl(raw: string): LiteralTable | null {
  const rows: LiteralTableRow[] = [];
  const owners: LiteralTableCell[] = [];
  let slots: LiteralTableCell[] = [];
  let rowStart = -1;
  let cursor = 0;
  while (cursor < raw.length && /\s/.test(raw[cursor])) cursor++;
  while (cursor < raw.length) {
    const start = cursor;
    const token = /^<(fcel|ecel|lcel|ucel|xcel|nl)>/.exec(raw.slice(cursor));
    if (!token) return null;
    cursor += token[0].length;
    const next = raw.indexOf("<", cursor);
    const end = next === -1 ? raw.length : next;
    const content = raw.slice(cursor, end);
    const kind = token[1];
    if (kind === "nl") {
      if (!slots.length || rows.length >= maxRows || content.trim())
        return null;
      if (rows.length && slots.length !== rows[0].slots.length) return null;
      rows.push({ start: rowStart, end: cursor, slots });
      slots = [];
      rowStart = -1;
      cursor = end;
      continue;
    }
    if (rowStart === -1) rowStart = start;
    const row = rows.length;
    const column = slots.length;
    if (
      row >= maxRows ||
      column >= maxColumns ||
      row * (rows[0]?.slots.length ?? 0) + column >= maxSlots
    )
      return null;
    let owner: LiteralTableCell | undefined;
    if (kind === "fcel" || kind === "ecel") {
      if (kind === "fcel" ? !content.trim() : content !== "") return null;
      owner = {
        text: content.trim(),
        start,
        end,
        row,
        column,
        rowSpan: 1,
        columnSpan: 1,
      };
      owners.push(owner);
    } else {
      if (content !== "") return null;
      const left = slots[column - 1];
      const above = rows[row - 1]?.slots[column];
      if (kind === "lcel" && left?.row === row) owner = left;
      if (kind === "ucel" && above?.column === column) owner = above;
      if (kind === "xcel" && left && left === above) owner = left;
      // An explicit ecel is also an origin. Its blank merged rectangle is
      // structural evidence, never a financial value; absent owners still fail.
      if (!owner) return null;
      owner.rowSpan = Math.max(owner.rowSpan, row - owner.row + 1);
      owner.columnSpan = Math.max(owner.columnSpan, column - owner.column + 1);
    }
    slots.push(owner);
    cursor = end;
  }
  if (slots.length || !rows.length) return null;
  const width = rows[0].slots.length;
  if (rows.length * width > maxSlots) return null;
  // Every owner must occupy its complete rectangle, without crossing spans.
  for (const owner of owners)
    for (let row = owner.row; row < owner.row + owner.rowSpan; row++)
      for (
        let column = owner.column;
        column < owner.column + owner.columnSpan;
        column++
      )
        if (rows[row]?.slots[column] !== owner) return null;
  return { format: "otsl", rows };
}

function decodeEntities(raw: string): string | null {
  let value = "";
  let cursor = 0;
  const named: Record<string, string> = {
    amp: "&",
    lt: "<",
    gt: ">",
    quot: '"',
    apos: "'",
    nbsp: "\u00a0",
  };
  while (cursor < raw.length) {
    const amp = raw.indexOf("&", cursor);
    if (amp === -1) return value + raw.slice(cursor);
    value += raw.slice(cursor, amp);
    const match =
      /^&(#(?:[0-9]+|x[0-9a-fA-F]+)|amp|lt|gt|quot|apos|nbsp);/.exec(
        raw.slice(amp),
      );
    if (!match) return null;
    if (match[1].startsWith("#")) {
      const hex = match[1].startsWith("#x");
      const point = Number.parseInt(match[1].slice(hex ? 2 : 1), hex ? 16 : 10);
      if (
        !Number.isSafeInteger(point) ||
        point <= 0 ||
        point > 0x10ffff ||
        (point >= 0xd800 && point <= 0xdfff) ||
        (point < 32 && ![9, 10, 13].includes(point))
      )
        return null;
      value += String.fromCodePoint(point);
    } else value += named[match[1]];
    cursor = amp + match[0].length;
  }
  return value;
}

function html(raw: string): LiteralTable | null {
  let cursor = 0;
  const whitespace = () => {
    while (cursor < raw.length && /\s/.test(raw[cursor])) cursor++;
  };
  const consume = (tag: string) => {
    whitespace();
    const match = new RegExp(`^<${tag}>`, "i").exec(raw.slice(cursor));
    if (!match) return false;
    cursor += match[0].length;
    return true;
  };
  if (!consume("table")) return null;
  const pending: {
    start: number;
    end: number;
    group: string;
    cells: LiteralTableCell[];
  }[] = [];
  let section: string | undefined;
  const sections = new Set<string>();
  let lastSection = -1;
  let grouped: boolean | undefined;
  while (true) {
    whitespace();
    if (/^<\/table>/i.test(raw.slice(cursor))) break;
    const opening = /^<(thead|tbody|tfoot)>/i.exec(raw.slice(cursor));
    if (opening) {
      const name = opening[1].toLowerCase();
      const order = ["thead", "tbody", "tfoot"].indexOf(name);
      if (
        section ||
        grouped === false ||
        sections.has(name) ||
        order < lastSection
      )
        return null;
      grouped = true;
      section = name;
      sections.add(name);
      lastSection = order;
      cursor += opening[0].length;
      continue;
    }
    const closing = /^<\/(thead|tbody|tfoot)>/i.exec(raw.slice(cursor));
    if (closing) {
      if (closing[1].toLowerCase() !== section) return null;
      section = undefined;
      cursor += closing[0].length;
      continue;
    }
    if (grouped === true && !section) return null;
    if (grouped === undefined) grouped = false;
    const start = cursor;
    if (!consume("tr") || pending.length >= maxRows) return null;
    const cells: LiteralTableCell[] = [];
    while (true) {
      whitespace();
      if (/^<\/tr>/i.test(raw.slice(cursor))) break;
      const cellStart = cursor;
      const tag =
        /^<(td|th)((?:\s+(?:rowspan|colspan)\s*=\s*(?:"[1-9][0-9]*"|'[1-9][0-9]*'))*)\s*>/i.exec(
          raw.slice(cursor),
        );
      if (!tag || cells.length >= maxColumns) return null;
      const spans: Record<string, number> = {};
      for (const attr of tag[2].matchAll(
        /(rowspan|colspan)\s*=\s*["']([1-9][0-9]*)["']/gi,
      )) {
        const name = attr[1].toLowerCase();
        const span = Number(attr[2]);
        if (
          spans[name] ||
          !Number.isSafeInteger(span) ||
          span > (name === "rowspan" ? maxRows : maxColumns)
        )
          return null;
        spans[name] = span;
      }
      cursor += tag[0].length;
      const end = raw.indexOf("<", cursor);
      if (end === -1) return null;
      const text = decodeEntities(raw.slice(cursor, end));
      if (text === null) return null;
      cursor = end;
      if (!consume(`/${tag[1]}`)) return null;
      cells.push({
        text: text.trim(),
        start: cellStart,
        end: cursor,
        row: pending.length,
        column: -1,
        rowSpan: spans.rowspan ?? 1,
        columnSpan: spans.colspan ?? 1,
      });
    }
    if (!consume("/tr")) return null;
    pending.push({ start, end: cursor, group: section ?? "ungrouped", cells });
  }
  if (section || !consume("/table")) return null;
  whitespace();
  if (cursor !== raw.length || !pending.length) return null;
  const grid: LiteralTableCell[][] = pending.map(() => []);
  let allocated = 0;
  for (const [row, source] of pending.entries()) {
    let column = 0;
    for (const cell of source.cells) {
      while (grid[row][column]) column++;
      if (
        column + cell.columnSpan > maxColumns ||
        row + cell.rowSpan > grid.length ||
        pending
          .slice(row, row + cell.rowSpan)
          .some((target) => target.group !== source.group) ||
        allocated + cell.columnSpan * cell.rowSpan > maxSlots
      )
        return null;
      cell.column = column;
      for (let y = row; y < row + cell.rowSpan; y++)
        for (let x = column; x < column + cell.columnSpan; x++) {
          if (grid[y][x]) return null;
          grid[y][x] = cell;
          allocated++;
        }
      column += cell.columnSpan;
    }
  }
  const width = grid[0].length;
  if (!width || grid.length * width > maxSlots) return null;
  if (
    grid.some(
      (row) =>
        row.length !== width ||
        Array.from({ length: width }, (_, i) => row[i]).some((cell) => !cell),
    )
  )
    return null;
  return {
    format: "html",
    rows: pending.map((row, i) => ({
      start: row.start,
      end: row.end,
      slots: grid[i],
    })),
  };
}

/** Bounded, complete structures only; malformed raw output remains the caller's evidence. */
export function parseLiteralTable(raw: string): LiteralTable | null {
  if (raw.length > maxCharacters * 2 || Array.from(raw).length > maxCharacters)
    return null;
  return /^\s*<table>/i.test(raw) ? html(raw) : otsl(raw);
}
