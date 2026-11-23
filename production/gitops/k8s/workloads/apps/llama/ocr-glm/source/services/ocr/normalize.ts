import { z } from "zod";
import {
  hasTableMarkup,
  isTableRecognitionBlock,
  parseLiteralTable,
  type LiteralTable,
  type LiteralTableCell,
} from "./table.js";
import {
  hasIncompleteOcrCoverage,
  ocrRecognizedPageSchema,
  ocrRecognizedPagesSchema,
  type OcrJob,
} from "../../shared/ocr.js";

export type RecognizedPage = z.infer<typeof ocrRecognizedPageSchema>;
export const ocrNormalizationRevision = "receipt-core-ptbr-v28" as const;
export type OcrNormalizedResult = NonNullable<OcrJob["result"]>;
export type OcrNormalizationWarning = OcrNormalizedResult["warnings"][number];
export const recognizedPagesSchema = ocrRecognizedPagesSchema;
const warningSchema = ocrRecognizedPageSchema.shape.warnings.element;
const normalizationInputSchema = z
  .object({
    pages: recognizedPagesSchema,
    mode: z.enum(["real", "mock"]),
    methodsUsed: z
      .array(z.enum(["ocr", "pdf_text", "llm"]))
      .max(3)
      .refine((values) => new Set(values).size === values.length),
    warnings: z.array(warningSchema).max(100).optional(),
    documentType: z
      .enum(["receipt", "invoice", "fiscal_representation", "unknown"])
      .optional(),
    multipleDocuments: z.boolean().optional(),
  })
  .strict()
  .superRefine((input, ctx) => {
    if (
      input.mode === "mock"
        ? input.methodsUsed.length !== 0
        : !input.methodsUsed.includes("ocr")
    )
      ctx.addIssue({
        code: "custom",
        message:
          "Normalization requires explicit, consistent execution provenance.",
      });
    if (
      input.warnings?.some(
        (warning) => warning.page !== null && warning.page > input.pages.length,
      )
    )
      ctx.addIssue({
        code: "custom",
        message: "Normalization warnings must reference an original page.",
      });
  });
export const ocrNormalizationInputSchema = z
  .unknown()
  .superRefine((value, ctx) => {
    if (value === null || typeof value !== "object" || Array.isArray(value))
      return;
    const input = value as Record<string, unknown>;
    for (const [key, limit] of [
      ["methodsUsed", 3],
      ["warnings", 100],
    ] as const) {
      if (Array.isArray(input[key]) && input[key].length > limit) {
        ctx.addIssue({
          code: "custom",
          message: "Normalization array exceeds its contract limit.",
          path: [key],
        });
        return;
      }
    }
  })
  .pipe(normalizationInputSchema);
export type OcrNormalizationInput = z.infer<typeof ocrNormalizationInputSchema>;

type FieldName = keyof OcrNormalizedResult["fields"];
type Evidence =
  OcrNormalizedResult["fields"]["merchant"]["candidates"][number]["evidence"][number];
type Candidate<T> = {
  id: string;
  rawText: string;
  value: T;
  confidence: null;
  evidence: Evidence[];
};
type Field<T> = {
  status: "read" | "ambiguous" | "missing" | "unreadable" | "unsupported";
  reason: string | null;
  candidates: Candidate<T>[];
};
type Line = {
  page: number;
  block: RecognizedPage["blocks"][number];
  quote: string;
  /** Interpretation may decode a validated table row; quote is always literal. */
  text: string;
  table: boolean;
  supportsFields: boolean;
  folded: string;
  offsets: number[];
  /** Physical line index exists only for ordinary text in its original block. */
  sourceLine?: number;
  /** Literal-table rows retain structural ownership separately from OCR lines. */
  tableRow?: number;
  tableCompleteRow?: boolean;
};
type Proposal<T> = { value: T; lines: Line[]; rank?: number; goodsRole?: boolean; issuerIdentity?: string; printedIssuerId?: string };
type Observation<T> = { seen: boolean; proposals: Proposal<T>[] };
const fold = (value: string) =>
  value.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();

function lineFor(
  page: number,
  block: RecognizedPage["blocks"][number],
  quote: string,
  text = quote,
  table = false,
  supportsFields = true,
): Line {
  const offsets: number[] = [];
  let offset = 0,
    folded = "";
  for (const character of text) {
    offset += character.length;
    const value = fold(character);
    folded += value;
    for (let i = 0; i < value.length; i++) offsets.push(offset);
  }
  return { page, block, quote, text, table, supportsFields, folded, offsets };
}

function labelled(line: Line, pattern: RegExp): string | undefined {
  const match = pattern.exec(line.folded);
  if (!match) return undefined;
  return line.text.slice(line.offsets[match[0].length - 1] ?? 0).trim();
}

function valueSpan(
  lines: Line[],
  index: number,
  value: string,
): { value: string; lines: Line[] } {
  if (
    value !== "" ||
    lines[index].table ||
    !lines[index + 1] ||
    lines[index + 1].table ||
    lines[index + 1].page !== lines[index].page ||
    lines[index + 1].block !== lines[index].block ||
    lines[index].sourceLine === undefined ||
    lines[index + 1].sourceLine !== lines[index].sourceLine! + 1
  )
    return { value, lines: [lines[index]] };
  return {
    value: lines[index + 1].text,
    lines: [lines[index], lines[index + 1]],
  };
}

const merchantLabel =
  /^(?:nome fantasia|razao social|estabelecimento|emitente|fornecedor)(?=\s|[:\-]|$)\s*[:\-]?\s*/;
const dateLabel =
  /^(?:data (?:de |da |do )?(?:emissao|transacao|compra|venda|pagamento|efetivacao)|emissao|emitid[oa] em)(?=\s|[:\-]|$)\s*[:\-]?\s*/;
// Kept separate from relational labels: a table needs its own literal origin;
// ordinary OCR may use its explicit literal or one physical same-block line.
const ownOriginDocumentDateLabel =
  /^data do doc(?:umento)?\.?(?=\s|[:\-]|$)\s*[:\-]?\s*/;
const fiscalConsumerCaption =
  /^documento auxiliar da nota fiscal (?:de|do) consumidor eletronic[ao]$/u;
const fiscalDocumentPrefix = /^(?:nfc?-?e|nf-?ce)(?=\s|[.:/\-]|$)/iu;
const identityDocumentHeading =
  /^(?:documento(?: de)? identidade|documento\s*:\s*(?:identidade|rg|cpf)|identidade|rg|cpf)(?=\s|[:./\-]|$)/u;
const unrelatedDateHeading =
  /^(?:protocolo|autorizacao|data (?:de |do |da )?(?:autorizacao|nascimento|processamento|vencimento)|vencimento|nascimento|processamento)(?=\s|[:.\-]|$)/u;

const localDateIssuerHeading =
  /^(?:(?:dados do )?(?:emitente|estabelecimento|fornecedor)(?=\s|[:\-]|$)|documento auxiliar(?: da| de)? nota fiscal(?=\s|[:\-]|$)|danfe(?=\s|[:\-]|$)|nota fiscal(?=\s|[:\-]|$)|boleto(?: bancario)?\s*$)/u;
function localDatePartyBoundary(line: Line): boolean {
  const heading = line.folded.split(/\r\n?|\n|\\n/, 1)[0].trim();
  if (otherPartyHeading.test(heading)) return true;
  if (/^(?:(?:dados(?:\s+(?:do(?:\(s\))?|da(?:\(s\))?|dos|das))?)\s+)?(?:beneficiari[oa]s?|cedentes?|tomador(?:a|es|as)?|favorecid[oa]s?|recebedor(?:a|es|as)?|destinatari[oa]s?|clientes?|consumidor(?:a|es|as)?|pagador(?:a|es|as)?|sacad[oa]s?|remetentes?|transportador(?:a|es|as)?)(?=\s|[:./\-()]|$)/u.test(heading)) return true;
  // An unrecognized Dados heading cannot silently establish issuer scope.
  // Only an explicit issuer heading can distinguish an unknown Dados role.
  // This stricter boundary belongs to inferred date context; an explicit
  // own-origin document-date caption retains its existing separate semantics.
  return /^dados(?=\s|[:.\-]|$)/u.test(heading) && !localDateIssuerHeading.test(heading);
}
const totalLabel =
  /^(?:valor total da nota fiscal|valor total da nota|valor total(?: da)? nf-?e|valor total do documento|valor do documento|total do atendimento|total do documento|total da nota fiscal|total da nota|total(?: da)? nf-?e|total a pagar|valor a pagar|valor total|total geral|total nf|total)(?=\s|[:(\-]|$)\s*[:\-]?\s*/;
const serviceGrossLabel =
  /^(?:valor(?: total)? d(?:o|os) servicos?|total dos servicos)(?=\s|[:(\-]|$)\s*[:\-]?\s*/u;
const serviceNetLabel =
  /^valor liquido(?: da (?:nfs-?e|nota(?: fiscal)?))?(?=\s|[:(\-]|$)\s*[:\-]?\s*/u;
const serviceFinalLabel =
  /^(?:valor liquido da nfs-?e\s*\+\s*ibs\s*\/\s*cbs|valor total(?: da)? nfs-?e|total(?: da)? nfs-?e)(?=\s|[:(\-]|$)\s*[:\-]?\s*/u;
const legalBusinessTitle =
  /\b(?:comercio|representacoes|empresa|servicos|correios|transportes|sindicato|associacao|cooperativa|fundacao|ltda|eireli)\b|\bs\s*\/\s*a\b/;
const issuerNumber = /^cnpj[\s.:/\-]*\d/;
const ordinaryOtherPartyHeading =
  /^(?:(?:(?:dados (?:do|da|dos)\s+)?(?:destinatario|cliente|consumidor|operadora|bandeira|(?:nome (?:do|da) )?passageir[oa]s?)|remetente|pagador|sacado)(?=\s|[:/]|$)|(?:dados (?:do|da)\s+)?transportador\s*\/\s*volumes transportados(?=\s*[:.|\-\/]|\s*$|\s+(?:razao social|nome(?:\s*\/\s*razao social|\s+razao social)?|identificacao|dados|cnpj|cpf|endereco)(?=\s|[:.|\-\/]|$))|(?:dados (?:do|da)\s+)?transportador\s*[:.\-]?\s*$|volumes transportados\s*[:.\-]?\s*$)/;
const serviceRecipientHeading =
  /^(?:(?:dados (?:do|da|dos|das)\s+)?tomador(?:a|es|as)?(?:\s+(?:de|do|da|dos|das|do\(s\)|da\(s\))\s+servico(?:s|\(s\))?)?|(?:nome|razao social) d[oa] tomador(?:a)?)(?:\s*[:.\-]|\s*$)/u;
const otherPartyHeading = new RegExp(
  ordinaryOtherPartyHeading.source + "|" + serviceRecipientHeading.source, "u",
);
const currencyLabel = /^(?:moeda|currency)(?=\s|[:\-]|$)\s*[:\-]?\s*/;
const excludedMerchant =
  /^(?:cnpj|cpf|ie|inscricao|destinatario|cliente|consumidor|operadora|bandeira|autorizacao|documento auxiliar|danfe|cupom fiscal|nota fiscal|recibo|comprovante|recebemos|assinatura|total|subtotal|data|emissao|vencimento|moeda|nome fantasia|razao social)\b/;
const businessTitle =
  /\b(?:restaurante|pizzaria|churrascaria|lanchonete|padaria|supermercado|mercado|mercearia|loja|farmacia|hotel|pousada|posto|cafeteria|papelaria|livraria|estacionamento)\b/;

const operatorMetric = "(?:pedidos?|vendas?|compras?|atendimentos?|entregas?|produtos?|itens|clientes?)";
const addressedBusiness = "(?:na|no|nas|nos|em|da|do|das|dos|pela|pelo|para)\\s+(?:sua|seu|suas|seus|nossa|nosso|nossas|nossos)\\s+(?:lojas?|restaurantes?|estabelecimentos?|empresas?|comercios?)";
const operatorStatisticTime = "(?:hoje|ontem|neste mes|nesta semana|neste periodo|em andamento|pendentes?)";
const operatorStatistic = new RegExp(
    "^(?:[*•]\\s*)*(?:(?:voce\\s+)?(?:tem|recebeu|possui|registrou)\\s+)?" +
      "\\d{1,9}\\s+" + operatorMetric +
      "(?:\\s+(?:(?:recebid|registrad|realizad)[ao]s?|hoje|ontem)){0,2}\\s+" +
      addressedBusiness + "(?:\\s+" + operatorStatisticTime + ")?[.!]?$",
    "u",
  );
const operatorOrderInstruction = new RegExp(
    "^(?:[*•]\\s*)*(?:acompanhe|consulte|gerencie|confira|retire|receba|entregue|verifique|controle)\\s+" +
      "(?:(?:o|os|a|as|seu|seus|sua|suas)\\s+)?(?:\\d{1,9}\\s+)?" +
      operatorMetric + "\\s+" + addressedBusiness + "[.!]?$",
    "u",
  );

function unlabelledMerchantInstruction(value: string): boolean {
  // These complete addressed statistic/order-management phrases describe the
  // operator's activity. A business-kind word in them is not an issuer title.
  // Classify a leading literal heading only; never unescape a field value.
  const text = fold(value).split(/\\[nr]|[\r\n]/u, 1)[0].trim();
  return operatorStatistic.test(text) || operatorOrderInstruction.test(text);
}
// Empty fiscal forms can put the next field heading after "Razão social".
// A heading alone is never a merchant, even after an explicit issuer label.
const merchantFormLabel =
  /^(?:endereco|numero|bairro(?:\s*\/\s*distrito)?|cep|municipio|cidade|uf|pais|telefone(?:\s*\/\s*fax)?|fone|fax|contato|e-?mail|quantidade|especie|marca|numeracao|frete por conta|codigo antt|placa do veiculo|peso bruto|peso liquido)[\s:.\-]*$/;

function merchantFormLegend(value: string): boolean {
  const text = fold(value);
  // These are printed form roles/codes, not a ban on businesses named Frete,
  // Transportes or Transportadora. A business word does not repair a coded
  // role legend, while an ordinary business name with no such codes survives.
  const roles = new Set(
    text.match(
      /\b(?:emitente|destinatario|remetente|terceiros?|sem frete)\b/g,
    ) ?? [],
  );
  const numberedRole =
    /(?:^|[^\d])(?:\(\s*\d{1,3}\s*\)|\d{1,3}\s*[-–:=.)/]?)\s*(?:emitente|destinatario|remetente|terceiros?|sem frete)\b/.test(
      text,
    );
  const startsNumberedRole =
    /^(?:\(\s*\d{1,3}\s*\)|\d{1,3}\s*[-–:=.)/]?)\s*(?:emitente|destinatario|remetente|terceiros?|sem frete)\b/.test(
      text,
    );
  return (
    /^(?:emitente|destinatario|remetente|terceiros?|sem frete)[\s:.\-]*$/.test(
      text,
    ) ||
    (/^frete por conta(?=\s|[:.\-]|$)/.test(text) &&
      roles.size > 0 &&
      (numberedRole ||
        (!legalBusinessTitle.test(text) && !businessTitle.test(text)))) ||
    (numberedRole && (roles.size > 1 || startsNumberedRole))
  );
}

function merchantValue(value: string): string | undefined {
  const trimmed = value.trim();
  if (
    !trimmed ||
    Array.from(trimmed).length > 300 ||
    !/\p{L}/u.test(trimmed) ||
    excludedMerchant.test(fold(trimmed)) ||
    merchantFormLabel.test(fold(trimmed)) ||
    merchantFormLegend(trimmed) ||
    /https?:\/\/|www\.|@/i.test(trimmed)
  )
    return undefined;
  return trimmed;
}

function narrowOriginalEdgeRegion(line: Line): boolean {
  const { x, width } = line.block.box;
  // This is deliberately limited to a narrow detector region at a horizontal
  // original edge. A full-width recognition band necessarily touches both
  // edges; its coarse box says nothing about individual word clipping.
  return width < 0.25 && (x <= 0.001 || x + width >= 0.999);
}

function receiptIssueDate(line: Line): string | undefined {
  const movement =
    /^movimento[\s.:\-]*(\d{2}([/.\-])\d{2}\2\d{4})(?:\s+hora[\s.:\-]*([0-2]\d:[0-5]\d(?::[0-5]\d)?))?$/iu.exec(
      line.text,
    );
  if (movement)
    return calendarDate(movement[1] + (movement[3] ? " " + movement[3] : ""));
  // A printed NFC-e/NF-e identifier and series followed by its issue timestamp.
  // Never select a date from authorization, due-date or customer headings.
  const fiscal =
    /^(?:nfc-?e|nf-?ce|nf-?e)[\s.:noº°-]*\d+\s+s[eé]rie[\s.:]*\d+\s+(?:emiss[aã]o[\s.:]*)?(\d{2}([/.\-])\d{2}\2\d{4})(?:\s*[-–]?\s*([0-2]\d:[0-5]\d(?::[0-5]\d)?))?(?:\s*[-–]\s*via consumidor)?$/iu.exec(
      line.text,
    );
  return fiscal
    ? calendarDate(fiscal[1] + (fiscal[3] ? " " + fiscal[3] : ""))
    : undefined;
}

function adjacentFiscalIssueDate(lines: Line[], index: number): Proposal<string> | undefined {
  const header = lines[index], next = lines[index + 1];
  if (!next || header.table || next.table || !next.supportsFields ||
      header.page !== next.page || header.block !== next.block) return undefined;
  // This identifies the printed document context only. The bounded opaque
  // series token may contain OCR letters; it is neither repaired nor treated
  // as proof of fiscal validity or as evidence that the date is correct.
  const identity = /^(?:nfc-?e|nf-?ce|nf-?e)[ \t.:]*(?:(?:numero|n[.oº°]?)[ \t.:]*)?(\d{1,9})[ \t]+s[eé]rie[ \t.:]*([a-z0-9]{1,6})$/iu.exec(header.text);
  if (!identity || Number(identity[1]) === 0 || !/\d/.test(identity[2])) return undefined;
  // Adjacency is physical in the same raw block, not merely the flattened
  // list after empty lines/table rows have been filtered out.
  if (header.sourceLine === undefined || next.sourceLine !== header.sourceLine + 1) return undefined;
  // The date and time must both be complete literals. An optional bounded
  // "Via Consu…" copy-label prefix can be clipped/opaque; its spelling never
  // supplies, repairs or validates a date. Other trailing data is refused.
  const stamp = /^(?:emiss[aã]o[ \t.:]*)?(\d{2}([/.\-])\d{2}\2\d{4})[ \t]+(?:[-–][ \t]*)?((?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)(?:[ \t]*[-–][ \t]*via[ \t]+consu[\p{L}.]{0,16})?$/iu.exec(next.text);
  if (!stamp) return undefined;
  const value = calendarDate(stamp[1] + ' ' + stamp[3]);
  return value ? { value, lines: [header, next] } : undefined;
}

function ordinaryDocumentIssueDate(lines: Line[], index: number): Proposal<string> | undefined {
  const line = lines[index];
  if (line.table) return undefined;
  const printed = labelled(line, ownOriginDocumentDateLabel);
  if (printed === undefined) return undefined;
  if (printed !== '') {
    const value = calendarDate(printed);
    return value ? { value, lines: [line] } : undefined;
  }
  const next = lines[index + 1];
  if (!next || next.table || !next.supportsFields || next.page !== line.page || next.block !== line.block) return undefined;
  if (line.sourceLine === undefined || next.sourceLine !== line.sourceLine + 1) return undefined;
  const value = calendarDate(next.text);
  return value ? { value, lines: [line, next] } : undefined;
}

function captionedFiscalIssueDate(line: Line): string | undefined {
  if (line.table) return undefined;
  // A bare number/series pair has no date semantics. The caller must first
  // establish one exact prior fiscal caption in this same ordinary block,
  // within 128 physical lines and without another-party/date-role barrier.
  const printed = /^n[uú]mero[ \t.:]*(\d{1,9})[ \t]+s[eé]rie[ \t.:]*([a-z0-9]{1,6})[ \t]+(\d{2}([/.\-])\d{2}\4\d{4})(?:[ \t]*[-–][ \t]*|[ \t]+)((?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?)(?:[ \t]*[-–][ \t]*via[ \t]+consu[\p{L}.]{0,16})?$/iu.exec(line.text);
  if (!printed || Number(printed[1]) === 0 || !/\d/.test(printed[2])) return undefined;
  return calendarDate(printed[3] + ' ' + printed[5]);
}

function localReceiptHeaderDate(group: Line[], line: Line): Proposal<string> | undefined {
  const { box } = line.block;
  if (line.table || line.sourceLine === undefined || line.sourceLine >= 8 ||
      box.y > 0.25 || box.width < 0.25 || narrowOriginalEdgeRegion(line)) return undefined;
  const printed = labelled(line, /^data\s*:\s*/);
  if (printed === undefined) return undefined;
  const header = group.filter((item) => item.sourceLine !== undefined && item.sourceLine < 12);
  if (header.some((item) => /\bnao (?:e|constitui) (?:um |uma )?(?:recibo|comprovante|nota fiscal|cupom fiscal)\b/u.test(item.folded))) return undefined;
  const firstBarrier = header.find((item) => {
    const heading = item.folded.split(/\r\n?|\n|\\n/, 1)[0].trim();
    // A bare CPF after the early date can identify the individual issuer.
    // Before the date it remains a barrier; explicit party/identity/date-role
    // headings always block, even when a later receipt caption is present.
    const identity = identityDocumentHeading.test(heading);
    const bareCpf = /^cpf(?=\s|[:./\-]|$)/u.test(heading);
    return localDatePartyBoundary(item) || unrelatedDateHeading.test(heading) ||
      (identity && (!bareCpf || item.sourceLine! <= line.sourceLine!));
  });
  if (firstBarrier && firstBarrier.sourceLine! <= line.sourceLine) return undefined;
  const issuerHeader = header.filter((item) => !firstBarrier || item.sourceLine! < firstBarrier.sourceLine!);
  const receipts = issuerHeader.filter((item) => /^(?:recibo(?: de (?:taxi|venda(?: de produtos)?|compra|pagamento|servicos?|prestacao de servicos|aluguel|despesas?|reembolso|honorarios))?|comprovante de (?:pagamento|venda|compra|transacao|despesa)|cupom fiscal|nota fiscal(?: (?:de|do) consumidor eletronic[ao]| eletronica)?|documento auxiliar(?: da| de)? nota fiscal(?: (?:de|do) consumidor eletronic[ao]| eletronica)?)[\s:.]*$/u.test(item.folded));
  const sales = issuerHeader.filter((item) => /^pedido de venda(?:[ \t]+(?:(?:n[oº°.]?|numero)[ \t]*[:.]?[ \t]*)?#?\d{1,12})?$/u.test(item.folded));
  if (receipts.length + sales.length !== 1) return undefined;
  const receipt = receipts[0], sale = sales[0];
  const identifier = issuerHeader.find((item) => /^cnpj[\s.:/\-]*(\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2})$/u.test(item.folded) && item.sourceLine! < line.sourceLine!);
  const issuer = identifier && issuerHeader.find((item) => item.sourceLine! < identifier.sourceLine! &&
    identifier.sourceLine! - item.sourceLine! <= 4 && merchantValue(item.text) !== undefined &&
    (businessTitle.test(item.folded) || legalBusinessTitle.test(item.folded)));
  if (!receipt && !(sale && identifier && issuer)) return undefined;
  let valueText = printed;
  const evidence: Line[] = [line];
  if (valueText === '') {
    const next = group.find((item) => item.sourceLine === line.sourceLine! + 1);
    if (!next || next.table || !next.supportsFields) return undefined;
    const heading = next.folded.split(/\r\n?|\n|\\n/, 1)[0].trim();
    if (localDatePartyBoundary(next) || identityDocumentHeading.test(heading) || unrelatedDateHeading.test(heading)) return undefined;
    valueText = next.text; evidence.push(next);
  }
  const stamp = valueText.split(/[ \t]+(?:hor[aá]rio|hora)(?:[ \t]*:[ \t]*|[ \t]+)/iu);
  if (stamp.length > 2) return undefined;
  const value = calendarDate(stamp.length === 2 ? stamp[0].trim() + ' ' + stamp[1].trim() : valueText);
  if (!value) return undefined;
  if (receipt) evidence.push(receipt);
  else evidence.push(issuer!, identifier!, sale!);
  return { value, lines: [...new Set(evidence)].sort((a, b) => a.sourceLine! - b.sourceLine!) };
}

function receiptItemCountTotal(line: Line): string | undefined {
  if (line.table || /[\r\n]/.test(line.text)) return undefined;
  // One recognized receipt-summary prefix, then one complete explicit BRL
  // final-total literal on this same source line. The count is not summed or
  // multiplied, and an empty amount can never borrow a following line/block.
  const tail = labelled(
    line,
    /^(?:(?:qtd|qtde)\.?|quantidade)[ \t]+total[ \t]+de[ \t]+itens[ \t]*:[ \t]*\d{1,5}[ \t]+valor[ \t]+total[ \t]+/,
  );
  if (tail === undefined) return undefined;
  const printed = /^R\$[ \t]*:?[ \t]*(.+)$/u.exec(tail);
  const value = printed ? money(printed[1]) : undefined;
  return value && (!value.currency || value.currency === "BRL")
    ? tail
    : undefined;
}
function totalCurrencyText(value: string): {
  value: string;
  currency?: string;
} {
  // Currency can be on a total label before a colon/parenthesis or the amount.
  // Printed equality/arrow separators are common on totals. They are not
  // arithmetic instructions: only a remaining exact monetary literal parses.
  const stripSeparator = (text: string) =>
    text.replace(/^(?:\s|\(\s*=\s*\)|[=:>])+/u, "");
  const amount = stripSeparator(value);
  // A word beginning with three capitals does not declare a currency. Keep
  // complete tokens, directly attached amounts and existing separators.
  const prefix = /^\(?\s*(R\$|[A-Z]{3})(?=\s|[():.=>0-9]|$)\s*\)?\s*[:.]*\s*(.*)$/u.exec(amount);
  return prefix
    ? {
        value: stripSeparator(prefix[2]),
        currency: prefix[1] === "R$" ? "BRL" : prefix[1],
      }
    : { value: amount };
}
function calendarDate(value: string): string | undefined {
  const match =
    /^(?:(\d{2})([/.\-])(\d{2})\2(\d{4})|(\d{4})-(\d{2})-(\d{2}))(?:\s+(?:[àa]s\s+)?([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?)?$/iu.exec(
      value,
    );
  if (!match) return undefined;
  const year = +(match[4] ?? match[5]),
    month = +(match[3] ?? match[6]),
    day = +(match[1] ?? match[7]);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (
    year < 1 ||
    month < 1 ||
    month > 12 ||
    day < 1 ||
    day >
      [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
  )
    return undefined;
  return (
    `${year}`.padStart(4, "0") +
    "-" +
    `${month}`.padStart(2, "0") +
    "-" +
    `${day}`.padStart(2, "0")
  );
}

// Exact two-digit fractions only. A lone decimal point is unambiguous; it
// cannot be a grouped thousands separator. No rounding or currency default.
function money(
  value: string,
): { value: number; currency?: string } | undefined {
  const match =
    /^(?:(R\$|[A-Z]{3})\s*)?((?:\d{1,3}(?:\.\d{3})+|\d+)),(\d{2})(?:\s+(R\$|[A-Z]{3}))?$/.exec(
      value,
    ) ??
    /^(?:(R\$|[A-Z]{3})\s*)?(\d+)\.(\d{2})(?:\s+(R\$|[A-Z]{3}))?$/.exec(value);
  if (!match || (match[1] && match[4] && match[1] !== match[4]))
    return undefined;
  const major = match[2].replaceAll(".", "").replace(/^0+(?=\d)/, "");
  if (major.length > 7) return undefined;
  const minor = Number(major) * 100 + Number(match[3]);
  if (minor > 100_000_000) return undefined;
  const token = match[1] ?? match[4];
  return {
    value: minor,
    ...(token ? { currency: token === "R$" ? "BRL" : token } : {}),
  };
}

function explicitCellLiteral(label: Line, value: string): boolean {
  if (labelled(label, merchantLabel) === "")
    return (
      merchantValue(value) !== undefined &&
      !merchantLabel.test(fold(value)) &&
      !dateLabel.test(fold(value)) &&
      !totalLabel.test(fold(value)) &&
      !currencyLabel.test(fold(value)) &&
      !otherPartyHeading.test(fold(value))
    );
  if (labelled(label, dateLabel) === "")
    return calendarDate(value) !== undefined;
  if (labelled(label, currencyLabel) === "")
    return value === "R$" || /^[A-Za-z]{3}$/.test(value);
  if (labelled(label, /^cnpj[\s.:/\-]*/) === "")
    return /^\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2}$/.test(value);
  const total = labelled(label, totalLabel);
  const serviceAmount = labelled(label, serviceFinalLabel) ?? labelled(label, serviceGrossLabel) ?? labelled(label, serviceNetLabel);
  if (serviceAmount !== undefined)
    return totalCurrencyText(serviceAmount).value === "" && money(value) !== undefined;
  const adjustment = labelled(label, /^(?:descontos?|frete)(?=\s|[:\-]|$)\s*[:\-]?\s*/);
  if (adjustment !== undefined)
    return totalCurrencyText(adjustment).value === "" && money(value) !== undefined;
  return (
    total !== undefined &&
    totalCurrencyText(total).value === "" &&
    money(value) !== undefined
  );
}

function pureTableLabel(label: Line): boolean {
  return (
    [merchantLabel, dateLabel, currencyLabel, /^cnpj[\s.:/\-]*/].some(
      (pattern) => labelled(label, pattern) === "",
    ) ||
    (() => {
      return [totalLabel, serviceFinalLabel, serviceGrossLabel, serviceNetLabel].some(pattern => {
        const total = labelled(label, pattern);
        return total !== undefined && totalCurrencyText(total).value === "";
      });
    })()
  );
}

function tableEvidenceLines(
  page: number,
  block: RecognizedPage["blocks"][number],
  table: LiteralTable,
): Line[] {
  const origins = [...new Set(table.rows.flatMap((row) => row.slots))].sort(
    (left, right) => left.start - right.start,
  );
  const positions = new Map(origins.map((origin, index) => [origin, index]));
  const partyCounts = [0];
  for (const origin of origins)
    partyCounts.push(
      partyCounts[partyCounts.length - 1] +
        (otherPartyHeading.test(fold(origin.text)) ||
        merchantLabel.test(fold(origin.text))
          ? 1
          : 0),
    );
  const relations = (origin: LiteralTableCell) => {
    const index = positions.get(origin)!;
    const candidates = [
      {
        direction: "right" as const,
        value: table.rows[origin.row].slots[origin.column + origin.columnSpan],
      },
      {
        direction: "below" as const,
        value: table.rows[origin.row + origin.rowSpan]?.slots[origin.column],
      },
    ];
    return candidates.filter(
      (
        candidate,
      ): candidate is {
        direction: "right" | "below";
        value: LiteralTableCell;
      } => {
        const { direction, value } = candidate;
        // A single exact monetary literal may put its currency on its own
        // line inside one cell. Never join digits, multiple values or labels;
        // existing row/span/party ownership checks still govern the relation.
        if (!value || /[\r\n]/.test(origin.text) ||
            (/[\r\n]/.test(value.text) && money(value.text) === undefined))
          return false;
        const target = positions.get(value);
        return (
          target !== undefined &&
          target > index &&
          partyCounts[target] === partyCounts[index + 1] &&
          (direction === "right"
            ? value.row === origin.row &&
              value.rowSpan === origin.rowSpan &&
              value.column === origin.column + origin.columnSpan
            : value.row === origin.row + origin.rowSpan &&
              value.column === origin.column &&
              value.columnSpan === origin.columnSpan)
        );
      },
    );
  };
  // These labels assign context to other printed monetary fields only. They
  // never create a document-total candidate, sum, conversion or cell geometry.
  const auxiliaryMoneyHeading =
    /^(?:(?:valor\s+)?(?:total\s+)?(?:(?:do|da|dos|das|de)\s+)?(?:frete|seguro|descontos?|icms|ipi|pis|cofins|tributos?|impostos?|produtos?|itens)|subtotal|tax(?:es)?|shipping|insurance|discount|(?:outras\s+)?despesas acessorias|base (?:de )?calculo(?: do)? (?:icms|ipi|pis|cofins))(?:\s*\(\s*(?:r\$|[a-z]{3})\s*\))?[\s:.\-]*$/u;
  const assignments = new Map<LiteralTableCell, Set<LiteralTableCell>>();
  for (const origin of origins) {
    const label = lineFor(page, block, origin.text);
    const tail = labelled(label, totalLabel);
    if (
      !(tail !== undefined && totalCurrencyText(tail).value === "") &&
      !auxiliaryMoneyHeading.test(label.folded)
    )
      continue;
    for (const { value } of relations(origin)) {
      if (!money(value.text)) continue;
      const owners = assignments.get(value) ?? new Set<LiteralTableCell>();
      owners.add(origin);
      assignments.set(value, owners);
    }
  }
  const itemHeading =
    /^(?:descri[cç][aã]o|description|produtos?|products?|item|itens|items|quantidade|quantity|qty\.?|qtd\.?|qtde\.?|c[oó]d(?:igo)?\.?|code|sku|un(?:idade)?\.?|units?|(?:valor|pre[cç]o|vlr?\.?)\s+unit(?:[aá]rio)?\.?|unit\s+(?:price|cost))(?=\s|:|$)/iu;
  return origins.flatMap((origin) => {
    const label = lineFor(page, block, origin.text);
    const right =
      table.rows[origin.row].slots[origin.column + origin.columnSpan];
    const literalRelations = relations(origin).filter(({ value }) =>
      explicitCellLiteral(label, value.text),
    );
    const relatedValues = new Set(literalRelations.map(({ value }) => value));
    const totalTail = labelled(label, totalLabel);
    const totalRelation =
      totalTail !== undefined && totalCurrencyText(totalTail).value === "";
    const itemRow = table.rows[origin.row].slots.some((cell) =>
      itemHeading.test(cell.text),
    );
    const genericTotal =
      /^(?:total|valor total)(?:\s*[:\-]?\s*(?:\(\s*(?:r\$|[a-z]{3})\s*\)|r\$|[a-z]{3}))?\s*[:\-]?\s*$/u.test(
        label.folded,
      );
    const genericTotalContextUnclear =
      genericTotal &&
      table.rows[origin.row].slots.some(
        (cell) =>
          cell !== origin &&
          cell.text !== "" &&
          !(
            cell === right &&
            cell.row === origin.row &&
            cell.rowSpan === origin.rowSpan &&
            cell.column === origin.column + origin.columnSpan &&
            explicitCellLiteral(label, cell.text)
          ) &&
          !pureTableLabel(lineFor(page, block, cell.text)),
      );
    const proposals: Line[] = [];
    for (const { direction, value } of literalRelations) {
      const competingMoney =
        totalRelation &&
        [...new Set([origin.row, value.row])].some((row) =>
          table.rows[row].slots.some(
            (cell) =>
              cell !== origin &&
              !relatedValues.has(cell) &&
              money(cell.text) !== undefined &&
              ![...(assignments.get(cell) ?? [])].some(
                (owner) => owner !== origin,
              ),
          ),
        );
      if (
        (direction === "below" &&
          genericTotal &&
          (itemRow || genericTotalContextUnclear)) ||
        competingMoney
      )
        continue;
      const row = table.rows[origin.row];
      const wholeRow =
        direction === "right" &&
        origin.rowSpan === 1 &&
        origin.column === 0 &&
        value.column + value.columnSpan === row.slots.length;
      proposals.push(
        { ...lineFor(
          page,
          block,
          block.text.slice(
            wholeRow ? row.start : origin.start,
            wholeRow ? row.end : value.end,
          ),
          `${origin.text}${/[:\-]$/.test(origin.text) ? " " : ": "}${value.text}`,
          true,
        ), tableRow: origin.row, tableCompleteRow: wholeRow },
      );
    }
    if (proposals.length) return proposals;
    const parts = origin.text
      .split(/\r\n?|\n/)
      .map((part) => part.trim())
      .filter(Boolean);
    const multiline = /[\r\n]/.test(origin.text);
    // Some completed table reads retain a literal backslash+n between the
    // caption and its own date. Interpret only this exact two-part caption
    // form; do not unescape the table, other cells, quotes or arbitrary text.
    const documentDateParts = origin.text
      .split(/\r\n?|\n|\\n/)
      .map((part) => part.trim());
    const documentDatePaired =
      documentDateParts.length === 2 &&
      labelled(lineFor(page, block, documentDateParts[0]), ownOriginDocumentDateLabel) === "" &&
      calendarDate(documentDateParts[1]) !== undefined;
    const paired =
      (multiline && parts.length === 2 && explicitCellLiteral(lineFor(page, block, parts[0]), parts[1])) ||
      documentDatePaired;
    const pairedParts = documentDatePaired ? documentDateParts : parts;
    const inlinePrefix =
      totalTail === undefined
        ? ""
        : origin.text.slice(0, origin.text.length - totalTail.length).trim();
    const genericInline = /^(?:total|valor total)[\s:\-]*$/u.test(
      fold(inlinePrefix),
    );
    const mergedInlineMoney =
      origin.rowSpan === 1 &&
      origin.columnSpan > 1 &&
      !multiline &&
      totalTail !== undefined &&
      money(totalCurrencyText(totalTail).value) !== undefined &&
      (!genericInline ||
        (!itemRow &&
          !table.rows[origin.row].slots.some(
            (cell) =>
              cell !== origin &&
              cell.text !== "" &&
              !pureTableLabel(lineFor(page, block, cell.text)),
          )));
    return [
      { ...lineFor(
        page,
        block,
        block.text.slice(origin.start, origin.end),
        paired
          ? `${pairedParts[0]}${/[:\-]$/.test(pairedParts[0]) ? " " : ": "}${pairedParts[1]}`
          : origin.text,
        true,
        (origin.rowSpan === 1 &&
          origin.columnSpan === 1 &&
          (!multiline || paired)) ||
          mergedInlineMoney,
      ), tableRow: origin.row, tableCompleteRow: origin.rowSpan === 1 && origin.column === 0 && origin.columnSpan === table.rows[origin.row].slots.length },
    ];
  });
}

function printedIssuerNumber(lines: Line[], index: number): string | undefined {
  if (!lines[index].supportsFields) return undefined;
  const printed = labelled(lines[index], /^cnpj[\s.:/\-]*/);
  if (printed === undefined) return undefined;
  const span = valueSpan(lines, index, printed);
  const match = /^(\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2})(?=$|\D)/.exec(
    span.value,
  );
  return match?.[1].replace(/\D/g, "");
}

/** Literal issuer titles in the ordinary header before one fiscal caption.
 * A CNPJ may precede its title or share the same physical line. A registered
 * individual need not have a business-kind word in their printed legal name.
 * No glyph, word or identifier is repaired; wrapped titles use literal spaces.
 */
function fiscalHeaderIssuers(lines: Line[]): Proposal<string>[] {
  const caption = /^(?:documento|doc\.) auxiliar (?:da|de) nota fiscal (?:de|do) consumidor eletronic[ao]$/u;
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    if (line.table || line.block.box.y > 0.2) continue;
    const group = groups.get(line.block) ?? [];
    group.push(line); groups.set(line.block, group);
  }
  const result: Proposal<string>[] = [];
  const metadata = /^(?:endereco|rua|r\.|avenida|av\.?|rodovia|rod\.?|estrada|travessa|alameda|cep|fone|telefone|tel\.?|fax|ie|inscricao|cnpj|cpf|codigo|item|quantidade|qtd|total|subtotal|data|emissao|vencimento)(?=\s|[.:/\-]|$)/u;
  const nameLine = (text: string, origin: Line) => merchantValue(text) !== undefined &&
    !metadata.test(fold(text)) && !merchantLabel.test(fold(text)) &&
    !localDatePartyBoundary(lineFor(origin.page, origin.block, text)) &&
    !unlabelledMerchantInstruction(text) && !fiscalDocumentPrefix.test(text) &&
    !/(?:[-,/]\s*|\s)(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)$/u.test(text) &&
    !/[\r\n]|\\[nr]/u.test(text);
  for (const group of groups.values()) {
    const captions: { index: number; lines: Line[] }[] = [];
    for (let i = 0; i < group.length; i++) {
      for (let length = 1; length <= 3 && i + length <= group.length; length++) {
        const chunk = group.slice(i, i + length);
        if (chunk.some((line, j) => j > 0 && line.sourceLine !== chunk[j - 1].sourceLine! + 1)) break;
        if (caption.test(chunk.map(line => line.folded).join(" "))) captions.push({ index: i, lines: chunk });
      }
    }
    // Repeated fiscal captions can indicate distinct source documents. Only
    // a bounded own header, with one literal identifier, supplies a new title.
    if (captions.length !== 1 || captions[0].index > 12) continue;
    const witness = captions[0], header = group.slice(0, witness.index);
    if (!header.length || header.some(line => line.sourceLine === undefined || line.sourceLine > 20 ||
      !line.supportsFields || narrowOriginalEdgeRegion(line) || goodsHeaderBoundary(line))) continue;
    const identifiers = header.flatMap((line, index) => {
      const id = printedIssuerNumber(header, index);
      return id ? [{ line, index, id }] : [];
    });
    if (identifiers.length !== 1 || header.some(line =>
      (line.folded.match(/\bcnpj\b/g)?.length ?? 0) > 1)) continue;
    const identifier = identifiers[0];
    const printed = labelled(identifier.line, /^cnpj[\s.:/\-]*/)!;
    const tail = /^(?:\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2})(?=$|\D)(.*)$/u.exec(printed)?.[1].trim();
    if (tail === undefined) continue;
    const idOnly = !tail || /^(?:ie|ins(?:cricao)?\.? est(?:adual)?\.?)[\s.:/\-]*[\d./\-]+$/u.test(fold(tail));
    let names: { text: string; line: Line }[] = [];
    if (identifier.index === 1 && idOnly && header[1].sourceLine === header[0].sourceLine! + 1 && nameLine(header[0].text, header[0])) {
      names = [{ text: header[0].text, line: header[0] }];
    } else if (identifier.index === 0) {
      if (!idOnly) {
        if (!nameLine(tail, identifier.line)) continue;
        names.push({ text: tail, line: identifier.line });
      }
      for (let i = 1; i < header.length && names.length < 3; i++) {
        if (/\b(?:ltda|eireli|s\s*\/\s*a)\.?$/iu.test(names.at(-1)?.text ?? "")) break;
        const line = header[i];
        if (line.sourceLine !== header[i - 1].sourceLine! + 1 || !nameLine(line.text, line)) break;
        names.push({ text: line.text, line });
      }
      // A bounded wrapped title must reach a printed legal suffix. Otherwise
      // a fourth/unowned line or an address could be hidden by truncation.
      if (names.length > 1 && !/\b(?:ltda|eireli|s\s*\/\s*a)\.?$/iu.test(names.at(-1)!.text)) continue;
    }
    const value = merchantValue(names.map(item => item.text).join(" "));
    if (!value || !names.length) continue;
    const evidence = [...new Set([...names.map(item => item.line), identifier.line, ...witness.lines])]
      .sort((a, b) => a.sourceLine! - b.sourceLine!);
    result.push({ value, lines: evidence, rank: 0, printedIssuerId: identifier.id });
  }
  return result;
}

/** NFS-e provider scope is after its caption and before the recipient/services.
 * CPF/CNPJ is contextual printed evidence, not identity or tax validation.
 * Gross service value, net value and tax-base/tax fields are distinct roles.
 * Retain different printed service/net values as alternatives; never calculate
 * an amount, borrow a detached label value, or promote ISS/base-of-tax amounts.
 */
function serviceInvoiceEvidence(lines: Line[]): {
  providers: Proposal<string>[];
  totals: Proposal<number>[];
  currencies: Proposal<string>[];
  totalSeen: boolean;
  unresolved: boolean;
  identityConflict: boolean;
  ownedNameOrigins: Map<Line, string>;
  differingAmounts: boolean;
} {
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    const group = groups.get(line.block) ?? [];
    group.push(line);
    groups.set(line.block, group);
  }
  const caption =
    /^(?:nfs-?e\s*[-–:]\s*)?nota fiscal de servicos? eletronica$|^documento auxiliar da nfs-?e$/u;
  const providerHeading =
    /^(?:(?:dados|identificacao) d[oa]\s+)?prestador(?:a)?(?:\s*(?:\/\s*fornecedor|d[eo](?:\(s\)|s)? servico(?:s|\(s\))?))?\s*[:.\-]*$/u;
  const serviceBody =
    /^(?:discriminacao|descricao|detalhamento) d[eo](?:\(s\)|s)? servico(?:s|\(s\))?(?=\s|[:.\-]|$)|^(?:valores da nota|valor(?: total)? d(?:o|os) servicos?|valor liquido)(?=\s|[:.\-]|$)/u;
  const nameLabel =
    /^(?:nome fantasia|razao social(?: d[oa] prestador(?:a)?)?|nome(?:\s*\/\s*nome empresarial|\s*\/\s*razao social|\s+empresarial|\s+d[oa] prestador(?:a)?)|nome(?=\s*[:.\-]|\s*$))(?=\s|[:.\-]|$)\s*[:.\-]?\s*/u;
  const idLabel = /^(?:cpf\s*\/\s*cnpj|cnpj\s*\/\s*cpf|cnpj|cpf)[\s:.\-]*/u;
  const metadata =
    /^(?:emitida? em|data|competencia|codigo|numero|n[º°o.]|inscricao|endereco|rua|r\.|avenida|av\.?|rodovia|cep|municipio|cidade|uf|pais|telefone|fone|fax|e-?mail|cnpj|cpf|valor|total|base de calculo|aliquota|prestador)(?=\s|[:./\-]|$)/u;
  const nameLine = (line: Line) =>
    merchantValue(line.text) !== undefined &&
    !metadata.test(line.folded) &&
    !otherPartyHeading.test(line.folded) &&
    !/^nome\s+(?:d[oa]|de)(?=\s|:|$)/u.test(line.folded) &&
    !providerHeading.test(line.folded) &&
    !serviceBody.test(line.folded) &&
    !nameLabel.test(line.folded) &&
    !unlabelledMerchantInstruction(line.text) &&
    !caption.test(line.folded) &&
    !/^[\d:]/u.test(line.text) &&
    !/^(?:AC|AL|AP|AM|BA|CE|DF|ES|GO|MA|MT|MS|MG|PA|PB|PR|PE|PI|RJ|RN|RS|RO|RR|SC|SP|SE|TO)$/u.test(
      line.text,
    );
  const id = (line: Line) => {
    const text = labelled(line, idLabel);
    if (text === undefined) return undefined;
    const m =
      /^(\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2}|\d{3}\.?\d{3}\.?\d{3}-?\d{2})(?=$|\s)(.*)$/u.exec(
        text,
      );
    if (
      !m ||
      (m[2].trim() &&
        !/^inscricao municipal[\s:.\-]*(?:[\d./\-]+|nao informad[oa])$/u.test(
          fold(m[2].trim()),
        ))
    )
      return undefined;
    const digits = m[1].replace(/\D/gu, "");
    if (
      (/^cnpj(?!\s*\/)/u.test(line.folded) && digits.length !== 14) ||
      (/^cpf(?!\s*\/)/u.test(line.folded) && digits.length !== 11)
    )
      return undefined;
    return (digits.length === 11 ? "cpf:" : "") + digits;
  };
  const owners: { group: Line[]; caption: Line[] }[] = [];
  for (const group of groups.values()) {
    if (group[0].table || group[0].block.box.y > 0.25) continue;
    const captions: { start: number; lines: Line[] }[] = [];
    for (let i = 0; i < Math.min(group.length, 20); i++)
      for (let count = 1; count <= 3 && i + count <= group.length; count++) {
        const chunk = group.slice(i, i + count);
        if (
          chunk.some(
            (line, j) =>
              line.sourceLine === undefined ||
              line.sourceLine > 24 ||
              (j > 0 && line.sourceLine !== chunk[j - 1].sourceLine! + 1),
          )
        )
          break;
        if (caption.test(chunk.map((line) => line.folded).join(" ")))
          captions.push({ start: i, lines: chunk });
      }
    if (captions.length !== 1) continue;
    const c = captions[0];
    owners.push({
      group: group.slice(c.start + c.lines.length),
      caption: c.lines,
    });
  }
  const providers: Proposal<string>[] = [],
    ownedNameOrigins = new Map<Line, string>();
  const addProviders = (group: Line[], context: Line[]) => {
    const end = group.findIndex(
      (line) =>
        otherPartyHeading.test(line.folded) || serviceBody.test(line.folded),
    );
    const header = group.slice(0, end < 0 ? 28 : Math.min(end, 28));
    const identifiers = header.flatMap((line, index) =>
      id(line) ? [{ line, index, id: id(line)! }] : [],
    );
    for (const identifier of identifiers) {
      const named: {
        value: string;
        lines: Line[];
        rank: number;
        origin?: Line;
      }[] = [];
      for (let i = 0; i < header.length; i++) {
        const printed = labelled(header[i], nameLabel);
        if (printed === undefined) continue;
        const span = valueSpan(header, i, printed),
          value = merchantValue(span.value);
        if (
          !value ||
          !span.lines.every((line) => header.includes(line)) ||
          otherPartyHeading.test(fold(value)) ||
          providerHeading.test(fold(value)) ||
          metadata.test(fold(value))
        )
          continue;
        const textLines = [...span.lines];
        let full = value;
        let next = i + span.lines.length;
        while (
          next < header.length &&
          textLines.length < 4 &&
          !/\b(?:ltda|eireli|s\s*\/\s*a)\.?$/iu.test(full) &&
          header[next].sourceLine === textLines.at(-1)!.sourceLine! + 1 &&
          nameLine(header[next])
        ) {
          full += " " + header[next].text;
          textLines.push(header[next++]);
        }
        if (full !== value && !/\b(?:ltda|eireli|s\s*\/\s*a)\.?$/iu.test(full))
          continue;
        if (!merchantValue(full)) continue;
        ownedNameOrigins.set(header[i], full);
        named.push({
          value: full,
          lines: textLines,
          rank: header[i].folded.startsWith("nome fantasia") ? 3 : 1,
        });
      }
      if (!named.length) {
        const before = header[identifier.index - 1],
          after = header[identifier.index + 1];
        const chosen =
          before &&
          before.sourceLine === identifier.line.sourceLine! - 1 &&
          nameLine(before)
            ? before
            : after &&
                after.sourceLine === identifier.line.sourceLine! + 1 &&
                nameLine(after)
              ? after
              : undefined;
        if (chosen) {
          const names = [chosen];
          let pos = header.indexOf(chosen) - 1;
          while (
            chosen === before &&
            names.length < 3 &&
            pos >= 0 &&
            header[pos].sourceLine === names[0].sourceLine! - 1 &&
            nameLine(header[pos])
          )
            names.unshift(header[pos--]);
          const value = merchantValue(names.map((line) => line.text).join(" "));
          if (
            value &&
            (names.length === 1 ||
              /\b(?:ltda|eireli|s\s*\/\s*a)\.?$/iu.test(value))
          )
            named.push({ value, lines: names, rank: 0 });
        }
      }
      for (const name of named)
        providers.push({
          value: name.value,
          rank: name.rank,
          printedIssuerId: identifier.id,
          lines: [
            ...new Set([...context, ...name.lines, identifier.line]),
          ].sort((a, b) =>
            a.block === b.block ? a.sourceLine! - b.sourceLine! : 0,
          ),
        });
    }
  };
  for (const owner of owners) addProviders(owner.group, owner.caption);
  const invoicePages = new Set(owners.map((owner) => owner.caption[0].page));
  if (!invoicePages.size)
    return {
      providers,
      totals: [],
      currencies: [],
      totalSeen: false,
      unresolved: false,
      identityConflict: false,
      ownedNameOrigins,
      differingAmounts: false,
    };
  const anchoredPages = new Set(providers.map((p) => p.lines[0].page));
  // A bounded header ROI may counter-read a name/identifier before its own
  // recipient heading. Merely sharing a page or identifier supplies no name.
  for (const group of groups.values()) {
    const box = group[0].block.box;
    if (
      !anchoredPages.has(group[0].page) ||
      group[0].table ||
      box.y > 0.25 ||
      box.height > 0.65 ||
      box.width < 0.25 ||
      box.width >= 0.95 ||
      !group.some((line) => serviceRecipientHeading.test(line.folded))
    )
      continue;
    if (
      owners.some((owner) =>
        owner.group.some((line) => line.block === group[0].block),
      )
    )
      continue;
    addProviders(group, []);
  }
  const ids = new Map<number, Set<string>>();
  for (const p of providers) {
    const set = ids.get(p.lines[0].page) ?? new Set<string>();
    set.add(p.printedIssuerId!);
    ids.set(p.lines[0].page, set);
  }
  const totals: Proposal<number>[] = [],
    currencies: Proposal<string>[] = [];
  const geometry = geometricTotalValueSpans(lines, [
    serviceFinalLabel,
    serviceGrossLabel,
    serviceNetLabel,
    totalLabel,
  ]);
  let totalSeen = false,
    unresolved = false,
    differingAmounts = false;
  const roles = new Map<number, { gross: Set<number>; net: Set<number> }>();
  const explicitFinal = new Set<number>();
  const explicitFinalLabel =
    /^(?:valor total da nota(?: fiscal)?|valor total do documento|total do documento|total da nota(?: fiscal)?|valor total(?: da)? nfs-?e|total(?: da)? nfs-?e|valor a pagar|total a pagar)(?=\s|[:(\-]|$)\s*[:\-]?\s*/u;
  for (const line of lines)
    if (
      invoicePages.has(line.page) &&
      (labelled(line, explicitFinalLabel) !== undefined ||
        labelled(line, serviceFinalLabel) !== undefined)
    )
      explicitFinal.add(line.page);
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (!invoicePages.has(line.page) || !line.supportsFields) continue;
    const final = labelled(line, serviceFinalLabel);
    if (explicitFinal.has(line.page) && final === undefined) continue;
    const gross = labelled(line, serviceGrossLabel),
      net = labelled(line, serviceNetLabel);
    const text = final ?? gross ?? net;
    if (text === undefined) continue;
    totalSeen = true;
    const parts = totalCurrencyText(text),
      local = valueSpan(lines, i, parts.value);
    const span = local.value === "" ? (geometry.get(line) ?? local) : local,
      value = money(span.value);
    if (!value) {
      unresolved = true;
      continue;
    }
    totals.push({ value: value.value, lines: span.lines });
    for (const currency of [parts.currency, value.currency])
      if (currency) currencies.push({ value: currency, lines: span.lines });
    const pageRoles = roles.get(line.page) ?? {
      gross: new Set<number>(),
      net: new Set<number>(),
    };
    (gross !== undefined ? pageRoles.gross : pageRoles.net).add(value.value);
    roles.set(line.page, pageRoles);
  }
  for (const values of roles.values())
    if (
      values.gross.size &&
      values.net.size &&
      new Set([...values.gross, ...values.net]).size > 1
    )
      differingAmounts = true;
  return {
    providers,
    totals,
    currencies,
    totalSeen,
    unresolved,
    ownedNameOrigins,
    differingAmounts,
    identityConflict: [...ids.values()].some((set) => set.size > 1),
  };
}

/** A taximeter header can name an individual provider without a CNPJ.
 * Require the complete local Taxi/name/own-CPF/phone/receipt sequence. A
 * passenger section, detached identifier or incomplete crop cannot own it.
 */
function taximeterHeaderProviders(lines: Line[]): Proposal<string>[] {
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    if (line.table || line.block.box.y > 0.2) continue;
    const group = groups.get(line.block) ?? [];
    group.push(line); groups.set(line.block, group);
  }
  const proposals: Proposal<string>[] = [];
  const taxi = /^taxi\s*:\s*\d{1,8}$/u;
  const receipt = /^recibo de taxi[\s:.]*$/u;
  const boundary = /^(?:passageir[oa]s?|solicitante|usuario|destino|origem|corrida|trajeto|percurso|inicio|fim|pagamento|valor|total|cnpj|rg|documento)(?=\s|[:./\-]|$)|^nome\s*:|^(?:nome|cpf|fone|telefone)\s+(?:do|da)\s+(?:passageir[oa]|cliente|consumidor)(?=\s|[:./\-]|$)/u;
  for (const group of groups.values()) {
    const captions = group.flatMap((line, index) => receipt.test(line.folded) ? [index] : []);
    if (captions.length !== 1 || captions[0] > 12) continue;
    const captionIndex = captions[0], header = group.slice(0, captionIndex);
    const taxis = header.flatMap((line, index) => taxi.test(line.folded) ? [index] : []);
    if (taxis.length !== 1 || taxis[0] + 4 !== captionIndex) continue;
    const start = taxis[0], [vehicle, name, identifier, phone] = header.slice(start);
    if (header.some(line => !line.supportsFields || line.sourceLine === undefined || line.sourceLine > 18 ||
        narrowOriginalEdgeRegion(line) || localDatePartyBoundary(line) || boundary.test(line.folded))) continue;
    if (header.filter(line => /^cpf(?=\s|[:./\-]|$)/u.test(line.folded)).length !== 1) continue;
    const literal = merchantValue(labelled(name, /^(?:motorista|condutora?|prestadora?)(?=\s|[:\-]|$)\s*[:\-]?\s*/) ?? name.text);
    if (!literal || fiscalDocumentPrefix.test(literal) || unlabelledMerchantInstruction(literal) ||
        !/^\p{L}[\p{L}\p{M}'’\-]*(?:[ \t]+\p{L}[\p{L}\p{M}'’\-]*){1,11}$/u.test(literal)) continue;
    const cpf = /^cpf(?:\s+(?:do|da)\s+(?:motorista|condutora?|prestadora?))?[\s:.\-]*(\d{3}\.?\d{3}\.?\d{3}-?\d{2})$/u.exec(identifier.folded);
    const contact = /^(?:fone|telefone|tel\.?)\s*[:.]?\s*([+\d() .\-]{7,30})$/u.exec(phone.folded);
    const phoneDigits = contact?.[1].replace(/\D/gu, "").length ?? 0;
    if (!cpf || phoneDigits < 8 || phoneDigits > 15 ||
        [name, identifier, phone].some((line, index) => line.sourceLine !== header[start + index].sourceLine! + 1) ||
        group[captionIndex].sourceLine! - phone.sourceLine! > 3) continue;
    proposals.push({ value: literal, rank: 0,
      printedIssuerId: "cpf:" + cpf[1].replace(/\D/gu, ""),
      lines: [vehicle, name, identifier, phone, group[captionIndex]] });
  }
  return proposals;
}

/** One literal fiscal summary may distinguish merchandise from final payable.
 * Exact cent arithmetic validates that local relationship; only the printed
 * payable becomes a candidate. Other blocks/tables/totals stay independent.
 */
function fiscalPayableSummaries(lines: Line[]): {
  merchandise: Set<Line>; evidence: Map<Line, Line[]>; unresolved: boolean;
} {
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    const group = groups.get(line.block) ?? [];
    group.push(line); groups.set(line.block, group);
  }
  const merchandise = new Set<Line>(), evidence = new Map<Line, Line[]>();
  let unresolved = false;
  const merchandiseLabel = /^valor total(?=\s|[:\-]|$)\s*[:\-]?\s*/;
  const payableLabel = /^(?:valor|total) a pagar(?=\s|[:\-]|$)\s*[:\-]?\s*/;
  const amount = (group: Line[], index: number, label: RegExp) => {
    const line = group[index];
    const text = labelled(line, label);
    if (!line.supportsFields || text === undefined) return undefined;
    const parts = totalCurrencyText(text), span = valueSpan(group, index, parts.value), value = money(span.value);
    return value && (parts.currency === "BRL" || value.currency === "BRL") &&
      (!parts.currency || parts.currency === "BRL") && (!value.currency || value.currency === "BRL")
      ? { value: value.value, lines: span.lines } : undefined;
  };
  const summaries: { first: Line; final: Line; lines: Line[]; owner?: Proposal<string>; key: string; inheritable: boolean }[] = [];
  const position = (line: Line) => line.table ? line.tableRow : line.sourceLine;
  const inputs = [...groups.values()].map(original => ({ original, geometric: false }));
  // Layout OCR may return each complete printed summary row as a separate
  // region. Require a unique tight vertical chain of original single-line
  // regions, then the same complete signature as an independently owned
  // summary. A lone repeated merchandise amount cannot inherit that ownership.
  const regions = new Map<number, Line[]>();
  for (const line of lines) {
    const box = line.block.box;
    if (line.table || !line.supportsFields || /[\r\n]|\\[nr]/u.test(line.block.text.trim()) ||
        box.width >= 0.9 || box.height > 0.1 || narrowOriginalEdgeRegion(line)) continue;
    const page = regions.get(line.page) ?? []; page.push(line); regions.set(line.page, page);
  }
  const aligned = (a: Line, b: Line) => {
    if (a.block === b.block) return false;
    const x = a.block.box, y = b.block.box;
    const overlap = Math.max(0, Math.min(x.x + x.width, y.x + y.width) - Math.max(x.x, y.x));
    const gap = y.y - x.y - x.height;
    return y.y >= x.y + x.height * 0.5 && overlap >= Math.min(x.width, y.width) * 0.85 &&
      Math.abs(x.x - y.x) <= Math.max(x.width, y.width) * 0.1 &&
      Math.abs(x.x + x.width - y.x - y.width) <= Math.max(x.width, y.width) * 0.15 &&
      gap >= -Math.min(x.height, y.height) * 0.25 && gap <= Math.max(x.height, y.height) * 0.75;
  };
  for (const page of regions.values()) {
    if (page.length > 256) continue;
    const next = new Map(page.map(line => [line, page.filter(other => aligned(line, other))]));
    const previous = new Map(page.map(line => [line, page.filter(other => aligned(other, line))]));
    for (const first of page) {
      if (labelled(first, merchandiseLabel) === undefined) continue;
      let before = first, foreign = false;
      for (let n = 0; n < 8; n++) {
        const parents = previous.get(before)!;
        if (parents.length !== 1) break;
        before = parents[0];
        if (localDatePartyBoundary(before) || merchantLabel.test(before.folded) ||
            fiscalDocumentPrefix.test(before.text) ||
            /^(?:cnpj|cpf|recibo|cupom|nota fiscal|documento auxiliar|danfe)(?=\s|[:./\-]|$)/u.test(before.folded) ||
            (merchantValue(before.text) && (businessTitle.test(before.folded) || legalBusinessTitle.test(before.folded)))) {
          foreign = true; break;
        }
      }
      if (foreign) continue;
      const chain = [first];
      while (chain.length < 4) {
        const children = next.get(chain.at(-1)!)!;
        if (children.length !== 1 || previous.get(children[0])!.length !== 1) break;
        const child = children[0]; chain.push(child);
        if (labelled(child, payableLabel) !== undefined) {
          inputs.push({ original: chain, geometric: true }); break;
        }
        if (!/^(?:descontos?|frete)(?=\s|[:\-]|$)/u.test(child.folded)) break;
      }
    }
  }
  for (const { original, geometric } of inputs) {
    const role = (line: Line) => /^(?:valor total|(?:valor|total) a pagar|descontos?|frete)(?=\s|[:\-]|$)/u.test(line.folded);
    const counts = new Map<number, number>();
    if (original[0].table) for (const line of original) if (role(line) && line.tableRow !== undefined)
      counts.set(line.tableRow, (counts.get(line.tableRow) ?? 0) + 1);
    const group = original[0].table ? original.filter(line => role(line) && line.tableCompleteRow &&
      line.tableRow !== undefined && counts.get(line.tableRow) === 1) : original;
    if (!group.length) continue;
    const starts = group.flatMap((line, index) => amount(group, index, merchandiseLabel) !== undefined ? [index] : []);
    const ends = group.filter(line => labelled(line, payableLabel) !== undefined);
    if (starts.length !== 1 || ends.length !== 1) continue;
    const issuers = fiscalHeaderIssuers(original);
    const start = starts[0], first = group[start], initial = amount(group, start, merchandiseLabel)!;
    const owner = issuers.length === 1 && issuers[0].lines.at(-1)!.sourceLine! < first.sourceLine!
      ? issuers[0] : undefined;
    const context = [...initial.lines], adjustments = new Map<string, number>();
    let expected = initial.value, previous = initial.lines.at(-1)!;
    let index = start + initial.lines.length;
    for (let role = 0; role < 3 && index < group.length; role++) {
      const line = group[index];
      if (line.page !== first.page || !line.supportsFields ||
          (geometric ? !aligned(previous, line) : position(line) === undefined ||
            position(line) !== position(previous)! + 1)) break;
      const final = amount(group, index, payableLabel);
      const payableText = labelled(line, payableLabel);
      if (owner && payableText !== undefined && final === undefined) {
        const raw = totalCurrencyText(payableText), span = valueSpan(group, index, raw.value);
        if (!money(span.value)) unresolved = true;
      }
      if (final !== undefined) {
        if (adjustments.size && final.value !== initial.value && final.value === expected) {
          const box = first.block.box;
          const inheritable = geometric || (first.table || (box.width < 0.9 && box.height < 0.8)) && original.filter(item => position(item)! < position(first)!).every(item =>
            !localDatePartyBoundary(item) && !merchantLabel.test(item.folded) &&
            !fiscalDocumentPrefix.test(item.text) &&
            !/^(?:cnpj|cpf|recibo|cupom|nota fiscal|documento auxiliar|danfe)(?=\s|[:./\-]|$)/u.test(item.folded) &&
            !(merchantValue(item.text) && (businessTitle.test(item.folded) || legalBusinessTitle.test(item.folded))));
          summaries.push({ first, final: line, owner, inheritable,
            lines: [...context, ...final.lines],
            key: JSON.stringify([first.page, initial.value, [...adjustments].sort(([a], [b]) => a.localeCompare(b)), final.value]) });
        }
        break;
      }
      const kind = /^(descontos?|frete)(?=\s|[:\-]|$)/u.exec(line.folded)?.[1];
      if (!kind) break;
      const key = kind.startsWith("desconto") ? "discount" : "freight";
      const adjustment = amount(group, index, /^(?:descontos?|frete)(?=\s|[:\-]|$)\s*[:\-]?\s*/);
      if (adjustments.has(key) || adjustment === undefined) break;
      adjustments.set(key, adjustment.value); expected += key === "discount" ? -adjustment.value : adjustment.value;
      context.push(...adjustment.lines); previous = adjustment.lines.at(-1)!; index += adjustment.lines.length;
    }
  }
  const owned = new Map<string, Proposal<string>[]>();
  for (const summary of summaries) if (summary.owner) {
    const owners = owned.get(summary.key) ?? []; owners.push(summary.owner); owned.set(summary.key, owners);
  }
  for (const summary of summaries) {
    const owners = owned.get(summary.key) ?? [];
    const inherited = summary.inheritable && owners.length > 0 && new Set(owners.map(owner => owner.printedIssuerId)).size === 1
      ? owners[0] : undefined;
    const owner = summary.owner ?? inherited;
    if (!owner) continue;
    merchandise.add(summary.first);
    evidence.set(summary.final, [...new Set([...owner.lines, ...summary.lines])]);
  }
  return { merchandise, evidence, unresolved };
}

/** A detached amount needs a unique, tightly aligned source region beside/below a
 * single-line total label. Recognition-array order cannot establish ownership.
 * No geometry or values are manufactured and no table cells are flattened.
 */
function geometricTotalValueSpans(
  lines: Line[], patterns: RegExp[] = [totalLabel],
): Map<Line, { value: string; lines: Line[] }> {
  const textFor = (line: Line) => {
    for (const pattern of patterns) { const text = labelled(line, pattern); if (text !== undefined) return text; }
    return undefined;
  };
  const byPage = new Map<number, Line[]>();
  for (const line of lines) {
    if (line.table || !line.supportsFields || /[\r\n]|\\[nr]/u.test(line.block.text.trim())) continue;
    const box = line.block.box;
    if (box.width > 0.8 || box.height > 0.1) continue;
    const page = byPage.get(line.page) ?? []; page.push(line); byPage.set(line.page, page);
  }
  const spans = new Map<Line, { value: string; lines: Line[] }>();
  for (const page of byPage.values()) {
    if (page.length > 256) continue;
    const labels = page.filter(line => {
      const text = textFor(line);
      if (text === undefined) return false;
      const parts = totalCurrencyText(text);
      return parts.value === "" && (!parts.currency || parts.currency === "BRL");
    });
    if (labels.length > 128) continue;
    const amounts = page.filter(line => money(line.text) !== undefined);
    const matches = new Map<Line, Line[]>(), owners = new Map<Line, Set<Line>>();
    for (const label of labels) {
      const parts = totalCurrencyText(textFor(label)!);
      const candidates = amounts.filter(value => {
        if (value.block === label.block) return false;
        const parsed = money(value.text)!;
        if ((parts.currency !== "BRL" && parsed.currency !== "BRL") ||
            (parsed.currency && parsed.currency !== "BRL")) return false;
        const a = label.block.box, b = value.block.box;
        const overlap = Math.max(0, Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x));
        const gap = b.y - a.y - a.height;
        const below = b.y >= a.y + a.height * 0.5 &&
          overlap >= Math.min(a.width, b.width) * 0.6 &&
          gap >= -Math.min(a.height, b.height) * 0.35 && gap <= Math.max(a.height, b.height) * 0.5;
        const overlapY = Math.max(0, Math.min(a.y + a.height, b.y + b.height) - Math.max(a.y, b.y));
        const gapX = b.x - a.x - a.width;
        const beside = b.x >= a.x + a.width * 0.5 &&
          overlapY >= Math.min(a.height, b.height) * 0.6 &&
          gapX >= -Math.min(a.width, b.width) * 0.2 && gapX <= Math.max(a.height, b.height) * 0.5;
        return below || beside;
      });
      matches.set(label, candidates);
      for (const value of candidates) { const set = owners.get(value) ?? new Set<Line>(); set.add(label); owners.set(value, set); }
    }
    for (const [label, candidates] of matches) if (candidates.length === 1 && owners.get(candidates[0])?.size === 1)
      spans.set(label, { value: candidates[0].text, lines: [label, candidates[0]] });
  }
  return spans;
}

/** A flattened fiscal logo cannot supply an inferred legal-name prefix.
 * Never manufacture a title by stripping that prefix: a separate literal
 * proposal must own the same printed CNPJ on the same page. Otherwise the
 * issuer remains unresolved, even if some other clean company is present.
 */
function fiscalCaptionMerchantReview(lines: Line[], proposals: Proposal<string>[]): {
  excluded: Set<Proposal<string>>; unresolved: boolean;
} {
  const unlabelled = (proposal: Proposal<string>) => !proposal.goodsRole &&
    fiscalDocumentPrefix.test(proposal.value) &&
    !proposal.lines.some(line => labelled(line, merchantLabel) !== undefined);
  const excluded = new Set(proposals.filter(unlabelled));
  if (!excluded.size) return { excluded, unresolved: false };
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    if (line.table) continue;
    const group = groups.get(line.block) ?? [];
    group.push(line); groups.set(line.block, group);
  }
  const positions = new Map<Line, { group: Line[]; index: number }>();
  for (const group of groups.values())
    group.forEach((line, index) => positions.set(line, { group, index }));
  const ownId = (proposal: Proposal<string>): string | undefined => {
    if (proposal.printedIssuerId) return proposal.printedIssuerId;
    const last = proposal.lines.at(-1)!;
    const position = positions.get(last);
    if (!position) return undefined;
    const { group, index } = position;
    for (let at = index + 1; at <= index + 4 && at < group.length; at++) {
      const line = group[at];
      if (line.page !== last.page || !line.supportsFields ||
          (line.sourceLine !== undefined && last.sourceLine !== undefined && line.sourceLine - last.sourceLine > 4) ||
          localDatePartyBoundary(line) ||
          labelled(line, merchantLabel) !== undefined) break;
      const printed = labelled(line, /^cnpj[\s.:/\-]*/);
      if (printed !== undefined) {
        if ((line.folded.match(/\bcnpj\b/g)?.length ?? 0) !== 1) return undefined;
        const span = valueSpan(group, at, printed);
        if (span.lines.some((part, i) => i > 0 &&
            part.sourceLine !== span.lines[i - 1].sourceLine! + 1)) return undefined;
        const tail = /^(?:\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2})(?=$|\D)(.*)$/u.exec(span.value)?.[1].trim();
        if (tail === undefined || (tail &&
            !/^(?:ie|ins(?:cricao)?\.? est(?:adual)?\.?)[\s.:/\-]*[\d./\-]+$/u.test(fold(tail)))) return undefined;
        return printedIssuerNumber(group, at);
      }
    }
    return undefined;
  };
  const key = (page: number, id: string, value: string) =>
    JSON.stringify([page, id, value.normalize("NFC").toLocaleLowerCase("pt-BR")]);
  const literals = new Set<string>();
  for (const proposal of proposals) {
    if (excluded.has(proposal)) continue;
    const id = ownId(proposal);
    if (id) literals.add(key(proposal.lines[0].page, id, proposal.value));
  }
  let unresolved = false;
  for (const proposal of excluded) {
    const marker = fiscalDocumentPrefix.exec(proposal.value)!;
    const suffix = proposal.value.slice(marker[0].length).replace(/^[\s.:/\-]+/u, "");
    const id = ownId(proposal);
    // This is only an identity comparison with an independently proposed
    // literal. The suffix is never added as a candidate or source quote.
    if (!id || (!literals.has(key(proposal.lines[0].page, id, proposal.value)) &&
        !literals.has(key(proposal.lines[0].page, id, suffix)))) unresolved = true;
  }
  return { excluded, unresolved };
}

function incompleteSmallHeaderPrefixes(
  lines: Line[],
  proposals: Proposal<string>[],
): { incomplete: Set<Proposal<string>>; budgetExhausted: boolean } {
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    if (line.table || line.block.box.y > 0.2) continue;
    const group = groups.get(line.block) ?? [];
    if (group.length < 12) group.push(line);
    groups.set(line.block, group);
  }
  const incomplete = new Set<Proposal<string>>();
  // The public normalizer accepts far more blocks than the real worker emits.
  // Bound this additional comparison pass instead of permitting P×G work.
  // Each comparison inspects at most the existing twelve-line header window.
  let comparisons = 0;
  const comparisonLimit = 4096;
  const copyHeader = /^(?:recibo (?:do|de) (?:pagador|sacado|beneficiario)|ficha de compensacao)[\s.:\-]*$/;
  const partyBoundary = (line: Line) => {
    // A literal escaped newline can delimit a printed/OCR party heading.
    // Classify its leading heading only; never unescape a financial value or
    // replace the retained source text/quote with synthetic line breaks.
    const heading = line.folded.split(/\\[nr]|[\r\n]/u, 1)[0];
    return otherPartyHeading.test(heading) || merchantLabel.test(heading) ||
      /^(?:beneficiario|cedente|tomador|favorecido|recebedor)(?=\s|[:/]|$)/.test(heading);
  };
  for (const proposal of proposals) {
    if (proposal.goodsRole) continue;
    if (proposal.lines.length !== 1 || (proposal.rank ?? 0) !== 0) continue;
    const primary = proposal.lines[0];
    const box = primary.block.box;
    if (
      primary.table || labelled(primary, merchantLabel) !== undefined ||
      box.y > 0.2 || box.height > 0.12 || narrowOriginalEdgeRegion(primary)
    ) continue;
    const own = groups.get(primary.block);
    const at = own?.indexOf(primary) ?? -1;
    if (!own || at < 0 || at > 2) continue;
    let ownId: string | undefined;
    for (let i = at + 1; i <= at + 4 && i < own.length; i++) {
      if (partyBoundary(own[i])) break;
      if (labelled(own[i], /^cnpj[\s.:/\-]*/) === undefined) continue;
      ownId = printedIssuerNumber(own, i);
      break;
    }
    if (!ownId) continue;
    for (const group of groups.values()) {
      if (comparisons++ >= comparisonLimit)
        return { incomplete, budgetExhausted: true };
      const wider = group[0];
      const region = wider?.block.box;
      if (!wider || wider.block === primary.block || wider.page !== primary.page ||
        region.width <= box.width || region.x > box.x || region.y > box.y ||
        region.x + region.width < box.x + box.width ||
        region.y + region.height < box.y + box.height) continue;
      const ids = group.flatMap((line, index) => {
        const id = printedIssuerNumber(group, index);
        return id ? [{ id, index }] : [];
      });
      // This matching printed ID establishes a bounded context comparison,
      // never tax authenticity or ownership of a new/full issuer candidate.
      if (!ids.length || ids.some((item) => item.id !== ownId) ||
        group.some((line, index) =>
          (labelled(line, /^cnpj[\s.:/\-]*/) !== undefined && !printedIssuerNumber(group, index)) ||
          (line.folded.match(/\bcnpj\b/g)?.length ?? 0) > 1)) continue;
      for (const [index, line] of group.entries()) {
        const value = merchantValue(line.text);
        if (!value || value === proposal.value || !value.startsWith(proposal.value) ||
          !/\p{L}$/u.test(proposal.value) ||
          !/^\p{L}/u.test(value.slice(proposal.value.length))) continue;
        const preceding = group.slice(0, index);
        if (preceding.some(partyBoundary)) continue;
        const immediateIdAfter = ids.some((item) => item.index > index && item.index <= index + 4);
        const ordinaryHeader = index === 0 && immediateIdAfter;
        const boundedCopyHeader = index > 0 && copyHeader.test(group[index - 1].folded) &&
          ids.some((item) => item.index < 4 && item.index < index);
        if (!ordinaryHeader && !boundedCopyHeader) continue;
        // Only suppress the incomplete small-ROI proposal. Do not create a
        // longer name candidate, join glyphs, infer a cell box or pick a winner.
        incomplete.add(proposal);
        break;
      }
      if (incomplete.has(proposal)) break;
    }
  }
  return { incomplete, budgetExhausted: false };
}

function crossPageIssuerNamesConflict(proposals: Proposal<string>[]): boolean {
  const pages = new Map<number, { rank: number; names: Set<string> }>();
  for (const proposal of proposals) {
    const page = proposal.lines[0].page, rank = proposal.rank ?? 0;
    const prior = pages.get(page);
    const name = (proposal.issuerIdentity ?? proposal.value).normalize("NFC").toLocaleLowerCase("pt-BR");
    if (!prior || rank > prior.rank) pages.set(page, { rank, names: new Set([name]) });
    else if (rank === prior.rank) prior.names.add(name);
  }
  return pages.size > 1 && new Set([...pages.values()].flatMap(value => [...value.names])).size > 1;
}

/** One complete declared goods-supplier role; never a cash/payment payer. */
function completeGoodsStubName(text: string): { value: string; identity: string } | undefined {
  if (text.length > 1024 || /[\r\n]|\\[nr]/u.test(text)) return undefined;
  // A closing metadata parenthesis can supply the printed boundary before OS;
  // fixed INDICADA/AO words may also touch in a table read. Names and spelling
  // stay literal, and arbitrary letters/cell origins are never split or joined.
  const match = /^recebemos[\p{Zs}\t]+de[\p{Zs}\t]+(.+?)(?:[\p{Zs}\t]+|(?<=\)))(?:os[\p{Zs}\t]+)?produtos[\p{Zs}\t]+constantes[\p{Zs}\t]+na[\p{Zs}\t]+nota[\p{Zs}\t]+fiscal[\p{Zs}\t]+indicada(?:[\p{Zs}\t]*ao[\p{Zs}\t]+lado)?[.\p{Zs}\t]*$/iu.exec(text);
  if (!match) return undefined;
  return goodsStubIdentity(match[1]);
}
function goodsStubIdentity(raw: string): { value: string; identity: string } | undefined {
  const identity = merchantValue(raw);
  if (!identity || merchantLabel.test(fold(identity)) ||
      /(?:recebemos\s+de|produtos\s+constantes|(?:cnpj|cpf|total)\s*[:=])/iu.test(identity)) return undefined;
  // Only complete terminal store metadata after an explicit LTDA suffix is
  // separate from the legal name. Never repair spelling or clipped codes.
  const metadata = goodsStoreMetadata(identity);
  return { value: metadata?.[1] ?? identity, identity };
}
/** Counter-review only: a complete legal title can precede a shorter fiscal clause.
 * No positive issuer is created, no missing suffix is appended, and different
 * retained names/store declarations still require independent resolution.
 */
function shortGoodsStubCounterclaim(text: string): { value: string; identity: string } | undefined {
  if (text.length > 1024 || /[\r\n]|\\[nr]/u.test(text)) return undefined;
  const match = /^recebemos[\p{Zs}\t]+de[\p{Zs}\t]+(.+?)(?:[\p{Zs}\t]+|(?<=\)))(?:os[\p{Zs}\t]+)?produtos[\p{Zs}\t]+constantes[\p{Zs}\t]+na[\p{Zs}\t]+nota[\p{Zs}\t]+fiscal[.\p{Zs}\t]*$/iu.exec(text);
  if (!match) return undefined;
  const name = goodsStubIdentity(match[1]);
  if (!name || !(goodsStoreMetadata(name.identity) || /\b(?:LTDA|EIRELI|S\/?A|S\.A\.)$/iu.test(name.identity))) return undefined;
  return name;
}
function goodsStoreMetadata(identity: string): RegExpExecArray | null {
  return /^(.+\bLTDA)[\p{Zs}\t]*\(LJ[\p{Zs}\t]*([0-9]{1,6})\)(?:[\p{Zs}\t]*\(([0-9]{1,6})\))?$/iu.exec(identity);
}
function goodsIssuerIdentityKey(identity: string): string {
  const metadata = goodsStoreMetadata(identity);
  // Compare only harmless spacing around complete printed store tokens. Store
  // numbers, leading zeros and every letter in the issuer name still matter.
  return metadata
    ? `${metadata[1]} (LJ${metadata[2]})${metadata[3] ? ` (${metadata[3]})` : ""}`
    : identity;
}
function goodsHeaderBoundary(line: Line): boolean {
  const headings = line.text.split(/\r\n?|\n|\\[nr]/u, 17);
  if (headings.length > 16) return true;
  return headings.some(text => localDatePartyBoundary(lineFor(line.page, line.block, text)));
}
function goodsRoleProposal(name: {value: string; identity: string}, evidence: Line[], ranks: Map<number, number>): Proposal<string> {
  return { value: name.value, issuerIdentity: goodsIssuerIdentityKey(name.identity), goodsRole: true,
    lines: evidence, rank: ranks.get(evidence[0].page) ?? 0 };
}
/** An ordinary header can own the same complete supplier declaration. */
function ordinaryGoodsStubMerchants(lines: Line[], existing: Proposal<string>[]): Proposal<string>[] {
  const ranks = new Map<number, number>();
  for (const proposal of existing) ranks.set(proposal.lines[0].page,
    Math.max(ranks.get(proposal.lines[0].page) ?? 0, proposal.rank ?? 0));
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    if (line.table || line.block.box.y > 0.2 || line.block.box.width < 0.5) continue;
    const group = groups.get(line.block) ?? [];
    group.push(line); groups.set(line.block, group);
  }
  const proposals: Proposal<string>[] = [];
  for (const group of groups.values()) {
    const witnesses = group.filter(line => line.folded === "danfe");
    if (witnesses.length !== 1) continue;
    const witness = witnesses[0], position = group.indexOf(witness);
    if (position >= 32 || witness.sourceLine === undefined || witness.sourceLine >= 48) continue;
    const header = group.slice(0, Math.max(position + 1, 3));
    const boundary = header.findIndex(goodsHeaderBoundary);
    const owned = boundary < 0 ? header : header.slice(0, boundary);
    if (!owned.includes(witness)) continue;
    for (const line of owned.slice(0, 3)) {
      if (!line.supportsFields || line.sourceLine === undefined || line.sourceLine > 4) continue;
      const name = completeGoodsStubName(line.text);
      if (!name) continue;
      const evidence = line.sourceLine < witness.sourceLine ? [line, witness] : [witness, line];
      proposals.push(goodsRoleProposal(name, evidence, ranks));
    }
  }
  return proposals;
}
/** Complete single-origin trusted table roles only; no row/cell/name repair. */
function typedGoodsStubMerchants(pages: RecognizedPage[], existing: Proposal<string>[]): Proposal<string>[] {
  const ranks = new Map<number, number>();
  for (const proposal of existing) ranks.set(proposal.lines[0].page,
    Math.max(ranks.get(proposal.lines[0].page) ?? 0, proposal.rank ?? 0));
  const proposals: Proposal<string>[] = [];
  for (const page of pages) for (const block of page.blocks) {
    if (!isTableRecognitionBlock(block.id) || block.box.y > 0.2 || block.box.width < 0.5) continue;
    const parsed = parseLiteralTable(block.text);
    if (!parsed) continue;
    const cells = [...new Set(parsed.rows.flatMap(row => row.slots))]
      .filter(cell => cell.row < 12).sort((a,b) => a.start - b.start);
    const header = cells.map(cell => ({ cell,
      line: lineFor(page.page, block, block.text.slice(cell.start,cell.end),cell.text,true) }));
    const boundary = header.findIndex(item => goodsHeaderBoundary(item.line));
    const owned = boundary < 0 ? header : header.slice(0,boundary);
    const witnesses = owned.filter(item => item.line.folded === "danfe");
    if (witnesses.length !== 1) continue;
    const roles = owned.flatMap(item => {
      const name = completeGoodsStubName(item.cell.text);
      return name ? [{...item,name}] : [];
    });
    // A late/body-only role never becomes a sole issuer. Once an early own-cell
    // role exists, other complete header roles remain conflict evidence too.
    if (!roles.some(item => item.cell.row < 3)) continue;
    for (const role of roles) {
      const witness = witnesses[0];
      const evidence = role.cell.start < witness.cell.start
        ? [role.line,witness.line] : [witness.line,role.line];
      proposals.push(goodsRoleProposal(role.name,evidence,ranks));
    }
  }
  return proposals;
}
/** Retained producer claims can oppose an owned positive, never create one. */
function retainedGoodsStubConflicts(pages: RecognizedPage[], existing: Proposal<string>[]): {
  proposals: Proposal<string>[]; incomplete: boolean;
} {
  const ranks = new Map<number, number>();
  for (const proposal of existing) ranks.set(proposal.lines[0].page,
    Math.max(ranks.get(proposal.lines[0].page) ?? 0, proposal.rank ?? 0));
  const proposals: Proposal<string>[] = [];
  let units=0, blocks=0, characters=0, incomplete=false;
  const producerStarter=/\brecebemos\b/iu;
  const review=(line: Line): boolean => {
    // One review unit is a physical plain line or one strict table origin.
    // No party, position, width or fiscal-witness gate can erase a counterclaim.
    if (units++>=4096) { incomplete=true;return false; }
    if (line.text.length>1024) { incomplete=true;return true; }
    if (!producerStarter.test(line.text)) return true;
    const name=completeGoodsStubName(line.text) ?? shortGoodsStubCounterclaim(line.text);
    // Escaped/wrapped/embedded/partial/cross-cell roles or unsafe values are
    // uncertainty only. Do not join/unescape values or invent a candidate.
    if (!name) { incomplete=true;return true; }
    proposals.push(goodsRoleProposal(name,[line],ranks));
    return true;
  };
  const neutralGap=/^(?:\s|\\[nr])*$/u;
  const reviewGap=(gap: string): boolean => {
    if ((characters+=gap.length)>1_000_000) return false;
    // Page text may legally contain unboxed text. It supplies no owned value,
    // but cannot hide a counterclaim. Only structural delimiters are neutral.
    if (!neutralGap.test(gap)) incomplete=true;
    return true;
  };
  for (const page of pages) {
    let offset=0;
    for (const block of page.blocks) {
      const start=page.text.indexOf(block.text,offset);
      if (start<0 || !reviewGap(page.text.slice(offset,start)))
        return {proposals,incomplete:true};
      offset=start+block.text.length;
      // Bound raw review and structural parsing too, including blank/invalid
      // channels that may not produce normalizer lines. Unknown remainder closes
      // merchant selection rather than treating unexamined text as absent.
      if (blocks++>=4096 || (characters+=block.text.length)>1_000_000)
        return {proposals,incomplete:true};
      const table=isTableRecognitionBlock(block.id);
      if (table || hasTableMarkup(block.text)) {
        const parsed=table?parseLiteralTable(block.text):null;
        // Untrusted/malformed structure cannot establish complete origin ownership.
        // It is never flattened, including for counter-only candidate creation.
        if (!parsed) { incomplete=true;continue; }
        const origins=[...new Set(parsed.rows.flatMap(row=>row.slots))]
          .sort((a,b)=>a.start-b.start);
        for (const cell of origins) if (!review(lineFor(page.page,block,
          block.text.slice(cell.start,cell.end),cell.text,true)))
          return {proposals,incomplete:true};
        continue;
      }
      for (const [sourceLine,value] of block.text.split("\n").entries()) {
        const quote=value.trim();
        if (!quote) continue;
        // The shared 4096-unit/one-million-character bounds cover the entire
        // review. A second per-block cap must not hide later counterclaims or
        // reject ordinary invoices just because they exceed 128 printed lines.
        if (!review({...lineFor(page.page,block,quote),sourceLine}))
          return {proposals,incomplete:true};
      }
    }
    if (!reviewGap(page.text.slice(offset))) return {proposals,incomplete:true};
  }
  return {proposals,incomplete};
}
/** Display normalization cannot erase different complete source declarations. */
function normalizedIssuerIdentityConflict(proposals: Proposal<string>[]): boolean {
  const byPage = new Map<number,Map<string,Set<string>>>();
  for (const proposal of proposals) {
    if (!proposal.goodsRole || !proposal.issuerIdentity) continue;
    const page=proposal.lines[0].page, names=byPage.get(page)??new Map<string,Set<string>>();
    const key=proposal.value.normalize("NFC").toLocaleLowerCase("pt-BR"), sources=names.get(key)??new Set<string>();
    sources.add(proposal.issuerIdentity.normalize("NFC").toLocaleLowerCase("pt-BR"));
    names.set(key,sources);byPage.set(page,names);
    if (sources.size>1) return true;
  }
  return false;
}

function overlappingHeaderEvidence(lines: Line[]): {
  proposals: Proposal<string>[];
  uncertainEdgeIssuer: boolean;
} {
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    // This is a bounded header-discovery heuristic, not proof of document
    // boundaries or universal header coverage. Body/footer ROIs get no new
    // first-line inference merely because another recognition read began.
    if (line.block.box.y > 0.2) continue;
    const group = groups.get(line.block) ?? [];
    if (group.length < 12) group.push(line);
    groups.set(line.block, group);
  }
  const proposals: Proposal<string>[] = [];
  let uncertainEdgeIssuer = false;
  for (const group of groups.values()) {
    let scope: "issuer" | "other" | undefined;
    for (let index = 0; index < Math.min(8, group.length); index++) {
      const line = group[index];
      if (otherPartyHeading.test(line.folded)) scope = "other";
      if (
        /^(?:emitente|estabelecimento|fornecedor|dados do emitente)(?=\s|:|$)/.test(
          line.folded,
        )
      )
        scope = "issuer";
      if (scope === "other") continue;
      const merchantText = labelled(line, merchantLabel);
      // A typed table gets only explicit issuer labels, never first-cell title
      // inference. A supported later top-ROI issuer still needs its own CNPJ
      // before a local customer/operator boundary, just like other overlaps.
      if (!line.supportsFields || (line.table && merchantText === undefined))
        continue;
      const span =
        merchantText === undefined
          ? { value: line.text, lines: [line] }
          : valueSpan(group, index, merchantText);
      const value = merchantValue(span.value);
      const positiveHeader =
        value !== undefined &&
        (merchantText !== undefined ||
          (!unlabelledMerchantInstruction(line.text) &&
            (businessTitle.test(line.folded) ||
              legalBusinessTitle.test(line.folded))));
      if (narrowOriginalEdgeRegion(line)) {
        if (positiveHeader || labelled(line, merchantLabel) !== undefined)
          uncertainEdgeIssuer = true;
        continue;
      }
      if (!positiveHeader) continue;
      // Additional overlapping reads require identifier context in this same
      // block. Do not turn every late crop's company-like first line into an
      // issuer, inherit customer scope from a different overlapping read, or
      // correct one spelling from another read/the evaluation truth.
      for (
        let next = index + 1;
        next <= index + 4 && next < group.length;
        next++
      ) {
        const following = group[next];
        if (otherPartyHeading.test(following.folded)) break;
        if (labelled(following, /^cnpj[\s.:/\-]*/) === undefined) continue;
        if (printedIssuerNumber(group, next))
          proposals.push({
            value,
            lines: span.lines,
            rank:
              merchantText === undefined
                ? 0
                : line.folded.startsWith("nome fantasia")
                  ? 3
                  : line.folded.startsWith("razao social")
                    ? 1
                    : 2,
          });
        break;
      }
    }
  }
  return { proposals, uncertainEdgeIssuer };
}

function neighboringHeaderAlternatives(
  pages: RecognizedPage[],
  lines: Line[],
  primary: Proposal<string>[],
): Proposal<string>[] {
  const groups = new Map<Line["block"], Line[]>();
  for (const line of lines) {
    const group = groups.get(line.block) ?? [];
    group.push(line);
    groups.set(line.block, group);
  }
  const primaryIds = new Map<number, Set<string>>();
  for (const proposal of primary) {
    if (proposal.printedIssuerId) {
      const page = proposal.lines[0].page;
      const ids = primaryIds.get(page) ?? new Set<string>();
      ids.add(proposal.printedIssuerId); primaryIds.set(page, ids);
      continue;
    }
    const last = proposal.lines.at(-1)!;
    const group = groups.get(last.block)!;
    const start = group.indexOf(last);
    // Identity must be printed next to the already supported primary name in
    // its own recognition block, never borrowed from an overlapping broad ROI.
    for (
      let index = start + 1;
      index <= start + 4 && index < group.length;
      index++
    ) {
      const line = group[index];
      if (
        otherPartyHeading.test(line.folded) ||
        merchantLabel.test(line.folded)
      )
        break;
      if (labelled(line, /^cnpj[\s.:/\-]*/) === undefined) continue;
      const id = printedIssuerNumber(group, index);
      if (id) {
        const ids = primaryIds.get(last.page) ?? new Set<string>();
        ids.add(id);
        primaryIds.set(last.page, ids);
      }
      break;
    }
  }
  const smallTop = (block: Line["block"]) =>
    block.box.y <= 0.2 && block.box.height <= 0.12;
  const nearby = (first: Line["block"], second: Line["block"]) => {
    const a = first.box,
      b = second.box;
    const overlap = Math.min(a.x + a.width, b.x + b.width) - Math.max(a.x, b.x);
    const gap = b.y - (a.y + a.height);
    return (
      smallTop(first) &&
      smallTop(second) &&
      b.y >= a.y &&
      Math.abs(gap) <= 0.05 &&
      overlap >= Math.min(a.width, b.width) * 0.5
    );
  };
  const proposals: Proposal<string>[] = [];
  for (const page of pages) {
    const ids = primaryIds.get(page.page);
    if (page.status !== "read" || !ids?.size) continue;
    const primaryRank = primary
      .filter((item) => item.lines[0].page === page.page)
      .reduce((rank, item) => Math.max(rank, item.rank ?? 0), 0);
    for (let index = 0; index + 1 < page.blocks.length; index++) {
      const block = page.blocks[index],
        idBlock = page.blocks[index + 1];
      const group = groups.get(block),
        idGroup = groups.get(idBlock);
      if (
        !group?.length ||
        group.length > 4 ||
        !idGroup?.length ||
        idGroup.length > 4 ||
        group.some((line) => line.table) ||
        idGroup.some((line) => line.table) ||
        !nearby(block, idBlock)
      )
        continue;
      const line = group[0];
      if (
        otherPartyHeading.test(line.folded) ||
        narrowOriginalEdgeRegion(line) ||
        narrowOriginalEdgeRegion(idGroup[0])
      )
        continue;
      const label = labelled(line, merchantLabel);
      const span =
        label === undefined
          ? { value: line.text, lines: [line] }
          : valueSpan(group, 0, label);
      const value = merchantValue(span.value);
      if (
        !value ||
        (label === undefined &&
          (unlabelledMerchantInstruction(line.text) ||
            (!businessTitle.test(line.folded) &&
              !legalBusinessTitle.test(line.folded))))
      )
        continue;
      if (
        group
          .slice(group.indexOf(span.lines.at(-1)!) + 1)
          .some(
            (next) =>
              otherPartyHeading.test(next.folded) ||
              merchantLabel.test(next.folded),
          )
      )
        continue;
      let barrier = false,
        nextBlock = block;
      // Inspect at most two immediately preceding small local elements. A
      // full-frame read's trailing customer section is unrelated to this top
      // spatial context and must not be inherited across overlapping reads.
      for (let prior = index - 1; prior >= Math.max(0, index - 2); prior--) {
        const previous = page.blocks[prior],
          previousLines = groups.get(previous);
        if (
          !previousLines ||
          previousLines.some((item) => item.table) ||
          !nearby(previous, nextBlock)
        )
          break;
        if (previousLines.some((item) => otherPartyHeading.test(item.folded))) {
          barrier = true;
          break;
        }
        nextBlock = previous;
      }
      if (barrier) continue;
      const prefix: Line[] = [];
      for (const item of idGroup) {
        if (
          otherPartyHeading.test(item.folded) ||
          merchantLabel.test(item.folded)
        )
          break;
        prefix.push(item);
      }
      if (!prefix.length) continue;
      const id = printedIssuerNumber(prefix, 0);
      const printedIds = new Set(
        [
          ...prefix
            .map((item) => item.text)
            .join("\n")
            .matchAll(
              /\bcnpj[\s.:/\-]*(\d{2}[. \t]?\d{3}[. \t]?\d{3}[\/ \t]?\d{4}[- \t]?\d{2})(?!\d)/gi,
            ),
        ].map((match) => match[1].replace(/\D/g, "")),
      );
      // Agreement concerns literal recognized identifiers, not tax authenticity.
      if (!id || !ids.has(id) || printedIds.size !== 1 || !printedIds.has(id))
        continue;
      if (
        primary.some(
          (item) =>
            item.lines[0].page === page.page &&
            item.value.normalize("NFC").toLocaleLowerCase("pt-BR") ===
              value.normalize("NFC").toLocaleLowerCase("pt-BR"),
        )
      )
        continue;
      const rank =
        label === undefined
          ? 0
          : line.folded.startsWith("nome fantasia")
            ? 3
            : line.folded.startsWith("razao social")
              ? 1
              : 2;
      proposals.push({
        value,
        lines: span.lines,
        // An additive cross-block reading cannot replace the primary set by
        // gaining priority. Existing lower legal/trading ranks stay lower.
        rank: Math.min(rank, primaryRank),
      });
    }
  }
  return proposals;
}

function contextualIssuerConflict(
  lines: Line[],
  proposals: Proposal<string>[],
): boolean {
  const positions = new Map(lines.map((line, index) => [line, index]));
  const byPage = new Map<number, Map<string, Set<string>>>();
  const retain = (proposal: Proposal<string>, page: number, id: string) => {
    const identities = byPage.get(page) ?? new Map<string, Set<string>>();
    const names = identities.get(id) ?? new Set<string>();
    names.add((proposal.issuerIdentity ?? proposal.value).normalize("NFC").toLocaleLowerCase("pt-BR"));
    identities.set(id, names); byPage.set(page, identities);
  };
  for (const proposal of proposals) {
    if (proposal.goodsRole) continue;
    if (proposal.printedIssuerId) {
      retain(proposal, proposal.lines[0].page, proposal.printedIssuerId);
      continue;
    }
    const last = proposal.lines[proposal.lines.length - 1];
    const start = positions.get(last);
    if (start === undefined) continue;
    // Associate a supported issuer header only with its first nearby printed
    // CNPJ. Never count arbitrary CNPJs, cross a customer/operator boundary,
    // or use coarse band boxes to infer individual document geometry.
    for (let index = start + 1; index <= start + 4; index++) {
      const line = lines[index];
      if (
        !line ||
        line.page !== last.page ||
        otherPartyHeading.test(line.folded)
      )
        break;
      const printed = labelled(line, /^cnpj[\s.:/\-]*/);
      if (printed === undefined) continue;
      const id = printedIssuerNumber(lines, index);
      if (id) retain(proposal, line.page, id);
      // A repeated/footer/branch CNPJ with no new named issuer header is not
      // a second associated issuer. Printed IDs are evidence, not validation.
      break;
    }
  }
  return [...byPage.values()].some(
    (identities) =>
      identities.size > 1 &&
      new Set([...identities.values()].flatMap((names) => [...names])).size > 1,
  );
}

type AnchoredTransactionPanel = {
  box: RecognizedPage["blocks"][number]["box"];
  quantity: string;
  total: number;
  receiptId?: string;
};

function independentTransactionPanelReview(pages: RecognizedPage[]): {
  suspected: boolean;
  incomplete: boolean;
} {
  const localRoi = (block: RecognizedPage["blocks"][number]) => {
    const { width, height } = block.box;
    // These are conservative ROI prerequisites, not document segmentation.
    // Full-frame reads, full-width fallback bands and tiny header/body crops
    // cannot independently establish an entire transaction panel here.
    return !isTableRecognitionBlock(block.id) && !hasTableMarkup(block.text) &&
      width >= 0.12 && width <= 0.8 && height >= 0.1 && height <= 0.9;
  };
  const partyBoundary = (text: string) => {
    const heading = fold(text).split(/\\[nr]|[\r\n]/u, 1)[0].trim();
    const issuerHeading = /^(?:dados (?:do|da) )?(?:emitente|estabelecimento|fornecedor)(?=\s|[:\-]|$)/u;
    if (otherPartyHeading.test(heading)) return true;
    if (/^(?:(?:dados(?:\s+(?:do(?:\(s\))?|da(?:\(s\))?|dos|das))?)\s+)?(?:beneficiari[oa]s?|cedentes?|tomador(?:a|es|as)?|favorecid[oa]s?|recebedor(?:a|es|as)?|destinatari[oa]s?|clientes?|consumidor(?:a|es|as)?|pagador(?:a|es|as)?|sacad[oa]s?|remetentes?|transportador(?:a|es|as)?)(?=\s|[:./\-()]|$)/u.test(heading)) return true;
    if (/^dados(?=\s|[:.\-]|$)/u.test(heading) && !issuerHeading.test(heading)) return true;
    return /^(?:beneficiario|cedente|tomador|favorecido|recebedor|vencimento|autorizacao|processamento|nascimento|forma de pagamento|pagamento|comprovante de pagamento|transacao aprovada|(?:data|hora|horario) (?:de |do |da )?(?:vencimento|autorizacao|processamento|nascimento|pagamento)|due date|authorization|processing date|payment date|date of birth)(?=\s|[:/]|$)/u.test(heading);
  };
  const bodyHeader = (text: string) => /^(?:produtos?|item|itens|descricao(?: (?:do|de) (?:produto|item|servico))?|servicos?|combustivel|quantidade|qtd\.?|qtde\.?|codigo|bico|preco|valor(?: (?:total|unitario|do|da|de|dos|das|documento|produto|frete|pagar)){0,5}|subtotal|total|impostos?|tributos?|base de calculo|frete|desconto|taxa)(?=\s*[:=]|\s*$)/u.test(fold(text));
  const receiptId = (rawText: string): string | undefined => {
    const text = fold(rawText);
    const id = /^(?:recibo|cupom|ticket|comprovante|documento)\s+(?:n[.º°o]?|numero|id)\s*[:#.\-]?\s*([A-Z0-9][A-Z0-9/\-]{0,39})$/iu.exec(text);
    if (id) return fold(id[1]);
    const fiscal = /^(nfc-?e|nf-?ce|nf-?e)\s*(?:n[.º°o]?|numero)?\s*[:#.]?\s*(\d{1,12})\s+s[eé]rie\s*[:.]?\s*([A-Z0-9]{1,4})$/iu.exec(text);
    // Normalize recognized label variants only. Numeric ID/series bytes remain
    // exact; label punctuation/spacing cannot manufacture a new transaction.
    return fiscal ? `${fiscal[1].replace(/-/gu, "")}:${fiscal[2]}:${fold(fiscal[3])}` : undefined;
  };
  const ordinaryLines = (text: string) => text.split(/\r?\n/u).map(text => text.trim()).filter(Boolean);
  const headerContext = (page: number, block: RecognizedPage["blocks"][number], rawLines: string[]) => {
    const boundary = rawLines.findIndex(partyBoundary);
    const texts = boundary < 0 ? rawLines : rawLines.slice(0, boundary);
    const local = texts.map(text => lineFor(page, block, text));
    const header = local.findIndex((line, index) => {
      if (index > 1) return false;
      const label = labelled(line, merchantLabel), value = label ?? line.text;
      if (label === undefined && (bodyHeader(line.text) || local.slice(0, index).some(item => bodyHeader(item.text)))) return false;
      return merchantValue(value) !== undefined &&
        (label !== undefined || (/^\p{L}/u.test(value) &&
          (businessTitle.test(line.folded) || legalBusinessTitle.test(line.folded) ||
            /\bltd\.?$/u.test(line.folded))));
    });
    if (header < 0) return undefined;
    const dateLines = local.flatMap((line, index) => {
      if (index <= header || index > header + 3) return [];
      const labelledDate = labelled(line, dateLabel) ?? labelled(line, /^data\s*:\s*/);
      // A bare timestamp has header ownership only across strict receipt-ID
      // metadata. Never hop over an unknown or nontransaction date heading.
      if (labelledDate === undefined && local.slice(header + 1, index).some(item => !receiptId(item.text))) return [];
      const value = labelledDate ?? line.text;
      return /(?:^|\s)(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$/u.test(value) &&
        calendarDate(value) ? [index] : [];
    });
    if (dateLines.length !== 1) return undefined;
    const timestampIndex = dateLines[0];
    return { local, header, timestampIndex };
  };
  const panel = (page: number, block: RecognizedPage["blocks"][number],
    rawLines = ordinaryLines(block.text), box = block.box): AnchoredTransactionPanel | undefined => {
    if (rawLines.length < 4 || rawLines.length > 12) return undefined;
    const context = headerContext(page, block, rawLines);
    if (!context) return undefined;
    const { local, header, timestampIndex } = context;
    let hasDescription = false, malformedReceiptId = false;
    const quantities = new Set<string>(), totals = new Set<number>(), ids = new Set<string>();
    for (const [index, line] of local.entries()) {
      if (index > header && (labelled(line, merchantLabel) !== undefined ||
        ((businessTitle.test(line.folded) || legalBusinessTitle.test(line.folded)) &&
          merchantValue(line.text) !== undefined && !/^(?:combustivel|produto|item|descricao|servico)\s*:/u.test(line.folded))))
        return undefined;
      const id = receiptId(line.text);
      if (id) ids.add(id);
      if ((/^(?:recibo|cupom|ticket|comprovante|documento)\s+(?:n[.º°o]?|numero|id)(?=\s|[:#.\-]|$)/iu.test(line.folded) ||
        /^(?:nfc-?e|nf-?ce|nf-?e)(?=\s|[:#.\d]|n(?:[.º°o]|\s|umero\b|\d|$)|$)/iu.test(line.text)) && !id) malformedReceiptId = true;
      if (index <= timestampIndex) continue;
      const description = /^(?:combustivel|produto|item|descricao|servico)\s*:\s*(.+)$/u.exec(line.folded);
      if (description && /\p{L}/u.test(description[1]) &&
        !/^(?:pagamento|autorizacao|debito|credito|bandeira|parcelas?|vencimento|impostos?|tributos?|icms|base de calculo)\b/u.test(description[1]))
        hasDescription = true;
      const quantity = /^(?:quantidade|qtd\.?|qtde\.?)\s*:\s*(\d{1,9})(?:,(\d{1,3}))?(?:\s+(?:L|KG|UN|PC|H))?$/iu.exec(line.text);
      if (quantity) {
        const integer = quantity[1].replace(/^0+(?=\d)/u, "");
        const fraction = (quantity[2] ?? "").replace(/0+$/u, "");
        if (/[1-9]/u.test(integer + fraction)) quantities.add(integer + (fraction ? "." + fraction : ""));
      }
      const totalText = labelled(line, totalLabel);
      if (totalText === undefined || !/^\(?\s*(?:R\$|BRL(?=\s|[):=.]|$))/u.test(totalText)) continue;
      const currency = totalCurrencyText(totalText), amount = money(currency.value);
      if (currency.currency === "BRL" && amount && (!amount.currency || amount.currency === "BRL")) totals.add(amount.value);
    }
    if (!hasDescription || quantities.size !== 1 || totals.size !== 1 || ids.size > 1 || malformedReceiptId) return undefined;
    return { box, quantity: [...quantities][0], total: [...totals][0], receiptId: [...ids][0] };
  };
  const separated = (a: AnchoredTransactionPanel, b: AnchoredTransactionPanel) => {
    const first = a.box, second = b.box;
    const overlapX = Math.max(0, Math.min(first.x + first.width, second.x + second.width) - Math.max(first.x, second.x));
    const overlapY = Math.max(0, Math.min(first.y + first.height, second.y + second.height) - Math.max(first.y, second.y));
    const minimumWidth = Math.min(first.width, second.width);
    return overlapX <= minimumWidth * 0.05 &&
      overlapY >= Math.min(first.height, second.height) * 0.6 &&
      Math.abs(first.x + first.width / 2 - second.x - second.width / 2) >= minimumWidth * 0.75;
  };
  for (const page of pages) {
    if (page.status !== "read") continue;
    const panels: AnchoredTransactionPanel[] = [];
    let scanned = 0, comparisons = 0;
    for (const block of page.blocks) {
      if (!localRoi(block)) continue;
      if (++scanned > 64) return { suspected: false, incomplete: true };
      const current = panel(page.page, block);
      if (!current) continue;
      if (panels.length >= 16) return { suspected: false, incomplete: true };
      for (const prior of panels) {
        if (++comparisons > 120) return { suspected: false, incomplete: true };
        if (!separated(current, prior)) continue;
        const different = current.receiptId && prior.receiptId
          ? current.receiptId !== prior.receiptId
          : !current.receiptId && !prior.receiptId && current.quantity !== prior.quantity && current.total !== prior.total;
        if (different) return { suspected: true, incomplete: false };
      }
      panels.push(current);
    }
    // Layout detectors commonly return a header, description, quantity and
    // total as separate small regions. Review those original fragments only
    // when two independently dated headers establish disjoint columns. These
    // temporary groups are never added to raw pages, blocks or field evidence.
    const fragments: RecognizedPage["blocks"][number][] = [];
    const headers: RecognizedPage["blocks"][number][] = [];
    for (const block of page.blocks) {
      const { width, height } = block.box;
      if (isTableRecognitionBlock(block.id) || hasTableMarkup(block.text) ||
          width < 0.04 || width > 0.8 || height < 0.005 || height > 0.12) continue;
      if (fragments.length >= 256) return { suspected: false, incomplete: true };
      if (block.text.length > 4_096) return { suspected: false, incomplete: true };
      fragments.push(block);
      if (width < 0.12 || height > 0.08) continue;
      const text = ordinaryLines(block.text);
      if (text.length < 2 || text.length > 5) continue;
      const context = headerContext(page.page, block, text);
      if (!context || context.timestampIndex !== text.length - 1) continue;
      if (headers.length >= 16) return { suspected: false, incomplete: true };
      headers.push(block);
    }
    let fragmentAssignments = 0, groupingIncomplete = false;
    for (let i = 0; i < headers.length; i++) for (let j = i + 1; j < headers.length; j++) {
      const [left, right] = [headers[i], headers[j]].sort((a, b) => a.box.x - b.box.x);
      const a = left.box, b = right.box;
      if (a.x + a.width > b.x || Math.abs(a.y - b.y) > 0.06 ||
          Math.abs(a.y - b.y) > Math.max(a.height, b.height) * 1.5) continue;
      const split = (a.x + a.width + b.x) / 2;
      const assembled = [left, right].map((header, side) => {
        const h = header.box, start = side === 0 ? 0 : split, end = side === 0 ? split : 1;
        // Titles are often centred while item labels start at the receipt's
        // left edge. Use centres to propose ownership, then require the full
        // union boxes to pass the same strict separation check below. A line
        // that bridges the two columns therefore cannot validate two panels.
        const owned = fragments.filter(part => {
          if (++fragmentAssignments > 8_192) { groupingIncomplete = true; return false; }
          return part.box.x + part.box.width / 2 >= start &&
            part.box.x + part.box.width / 2 <= end &&
            part.box.y >= h.y - 0.12 && part.box.y + part.box.height <= h.y + 0.42;
        });
        if (groupingIncomplete) return undefined;
        // A nearby recipient/payment heading belongs to the same original
        // column; starting a new group at its company name must not erase it.
        if (owned.some(part => part !== header && part.box.y < h.y &&
            ordinaryLines(part.text).some(partyBoundary))) return undefined;
        let below = owned.filter(part => part === header || part.box.y >= h.y + h.height - 0.005)
          .sort((a, b) => a.box.y - b.box.y || a.box.x - b.box.x);
        const boundary = below.findIndex(part => part !== header && ordinaryLines(part.text).some(partyBoundary));
        if (boundary >= 0) below = below.slice(0, boundary + 1);
        if (below[0] !== header || below.length < 4 || below.length > 16) return undefined;
        const x = Math.min(...below.map(part => part.box.x)), y = h.y;
        const box = { x, y, width: Math.max(...below.map(part => part.box.x + part.box.width)) - x,
          height: Math.max(...below.map(part => part.box.y + part.box.height)) - y };
        if (box.height < 0.1 || box.width > 0.8) return undefined;
        return panel(page.page, header, below.flatMap(part => ordinaryLines(part.text)), box);
      });
      if (groupingIncomplete) return { suspected: false, incomplete: true };
      const [first, second] = assembled;
      if (!first || !second || !separated(first, second)) continue;
      const different = first.receiptId && second.receiptId
        ? first.receiptId !== second.receiptId
        : !first.receiptId && !second.receiptId && first.quantity !== second.quantity && first.total !== second.total;
      if (different) return { suspected: true, incomplete: false };
    }
  }
  return { suspected: false, incomplete: false };
}

function field<T>(name: FieldName, observation: Observation<T>): Field<T> {
  const unique = new Map<string, Proposal<T>>();
  for (const proposal of observation.proposals) {
    if (
      Array.from(proposal.lines.map((line) => line.quote).join("\n")).length >
      10_000
    )
      continue;
    const key =
      name === "merchant"
        ? String(proposal.value).normalize("NFC").toLocaleLowerCase("pt-BR")
        : String(proposal.value);
    if (!unique.has(key)) unique.set(key, proposal);
  }
  if (unique.size > 5)
    return {
      status: "unsupported",
      reason:
        "More than five distinct supported alternatives require manual inspection.",
      candidates: [],
    };
  const candidates: Candidate<T>[] = [...unique.values()].map(
    (proposal, index) => ({
      id: `${name}-${index + 1}`,
      value: proposal.value,
      confidence: null,
      rawText: proposal.lines.map((line) => line.quote).join("\n"),
      evidence: proposal.lines.map((line) => ({
        page: line.page,
        blockIds: [line.block.id],
        box: { ...line.block.box },
        quote: line.quote,
      })),
    }),
  );
  return candidates.length === 1
    ? { status: "read", reason: null, candidates }
    : candidates.length > 1
      ? {
          status: "ambiguous",
          reason: "Distinct printed alternatives require human selection.",
          candidates,
        }
      : {
          status: observation.seen ? "unreadable" : "missing",
          reason: observation.seen
            ? "The field label is visible but its value cannot be normalized reliably."
            : "No supported printed field evidence was found.",
          candidates: [],
        };
}

/** Pure deterministic normalization. Text, URLs and document instructions never cause I/O. */
export function normalizeOcrPages(raw: OcrNormalizationInput): {
  state: "completed" | "partial";
  result: OcrNormalizedResult;
} {
  const parsed = ocrNormalizationInputSchema.safeParse(raw);
  if (!parsed.success) throw new Error("OCR normalization input is invalid.");
  const input = parsed.data;
  if (!input.pages.some((page) => page.status === "read"))
    throw new Error(
      "OCR normalization requires at least one readable original page.",
    );
  const warnings = [...(input.warnings ?? [])];
  const warn = (
    code: OcrNormalizationWarning["code"],
    message: string,
    page: number | null = null,
  ) => {
    if (
      !warnings.some(
        (warning) =>
          warning.code === code &&
          warning.page === page &&
          warning.message === message,
      )
    )
      warnings.push({ code, message, page });
  };
  for (const page of input.pages)
    if (
      page.status === "unreadable" &&
      !page.warnings.some((warning) => warning.code === "page_unreadable")
    )
      warn("page_unreadable", "An original page could not be read.", page.page);
  let multiple =
    input.multipleDocuments ||
    warnings.some((warning) => warning.code === "multiple_documents") ||
    input.pages.some((page) =>
      page.warnings.some((warning) => warning.code === "multiple_documents"),
    );
  const panelReview = multiple
    ? { suspected: false, incomplete: false }
    : independentTransactionPanelReview(input.pages);
  multiple ||= panelReview.suspected;
  const unsupportedFields = (reason: string): OcrNormalizedResult["fields"] => {
    const unsupported = (): Field<never> => ({
      status: "unsupported",
      reason,
      candidates: [],
    });
    return {
      merchant: unsupported(),
      date: unsupported(),
      total: unsupported(),
      currency: unsupported(),
    };
  };
  let fields: OcrNormalizedResult["fields"];
  let crossPageIssuerConflict = false;
  let samePageIssuerConflict = false;
  let normalizedIssuerConflict = false;
  let goodsConflictReviewIncomplete = false;
  if (multiple) {
    fields = unsupportedFields(
      panelReview.suspected
        ? "Conflicting independently anchored transaction panels require manual boundary review before core fields can be selected."
        : "Explicit document boundaries identify multiple documents; separation is required.",
    );
    warn(
      "multiple_documents",
      panelReview.suspected
        ? "Independently anchored, spatially separated transaction panels conflict. Manual boundary or identity review is required; no physical document count or separation was inferred."
        : "Multiple documents require explicit separation before field extraction.",
    );
  } else {
    const lines = input.pages
      .filter((page) => page.status === "read")
      .flatMap((page) =>
        page.blocks.flatMap((block) => {
          const table = isTableRecognitionBlock(block.id);
          if (table || hasTableMarkup(block.text)) {
            // Only the fixed trusted task ID enables structural association.
            // Malformed/ordinary OCR markup stays raw and cannot fall through
            // into line-based totals, dates, headers or issuer detection.
            const parsed = table ? parseLiteralTable(block.text) : null;
            if (!parsed) {
              warn(
                "low_quality",
                "Table-like output has no supported complete literal structure; its raw evidence requires manual inspection.",
                page.page,
              );
              return [];
            }
            return tableEvidenceLines(page.page, block, parsed);
          }
          return block.text.split("\n").flatMap((value, sourceLine) => {
            const quote = value.trim();
            return quote ? [{ ...lineFor(page.page, block, quote), sourceLine }] : [];
          });
        }),
      );
    const merchants: Observation<string> = { seen: false, proposals: [] };
    const merchantLabelPages = new Set<number>();
    const dates: Observation<string> = { seen: false, proposals: [] };
    const totals: Observation<number> = { seen: false, proposals: [] };
    const currencies: Observation<string> = { seen: false, proposals: [] };
    const payableSummaries = fiscalPayableSummaries(lines);
    const serviceInvoices = serviceInvoiceEvidence(lines);
    const geometricTotals = geometricTotalValueSpans(lines);
    let uncertainEdgeIssuer = false;
    let bareCurrency = false;
    let merchantScope: "issuer" | "other" | undefined;
    let scopePage = 0;
    // Only the new document-date caption uses this local party scope. An
    // overlapping block cannot inherit a customer section from another ROI.
    const documentDateScopes = new Map<Line["block"], "other" | "document">();
    const fiscalCaptionCounts = new Map<Line["block"], number>();
    for (const line of lines) if (!line.table && fiscalConsumerCaption.test(line.folded))
      fiscalCaptionCounts.set(line.block, (fiscalCaptionCounts.get(line.block) ?? 0) + 1);
    const fiscalDateScopes = new Map<Line["block"], { captionLine: number; caption: Line; blocked: boolean }>();
    const ordinaryBlockHeaders = new Map<Line["block"], Line[]>();
    for (const line of lines) if (!line.table && line.sourceLine !== undefined && line.sourceLine < 12) {
      const header = ordinaryBlockHeaders.get(line.block) ?? [];
      header.push(line); ordinaryBlockHeaders.set(line.block, header);
    }
    for (const [index, line] of lines.entries()) {
      if (line.page !== scopePage) {
        merchantScope = undefined;
        scopePage = line.page;
      }
      if (otherPartyHeading.test(line.folded)) merchantScope = "other";
      const localHeading = line.folded.split(/\r\n?|\n|\\n/, 1)[0].trim();
      const inferredDatePartyHere = localDatePartyBoundary(line) || identityDocumentHeading.test(localHeading);
      // Data do Doc/Documento explicitly names the document's own date. A
      // beneficiary may be the issuer of a boleto; treating every such role
      // as a customer would erase this stronger own-origin field. Preserve
      // the existing customer/identity guard independently of generic DATA.
      const documentDatePartyHere = otherPartyHeading.test(localHeading) || identityDocumentHeading.test(localHeading);
      if (documentDatePartyHere)
        documentDateScopes.set(line.block, "other");
      if (localDateIssuerHeading.test(localHeading))
        documentDateScopes.set(line.block, "document");
      const documentDateAllowed = documentDateScopes.get(line.block) !== "other";
      if (!line.table && fiscalConsumerCaption.test(line.folded) && fiscalCaptionCounts.get(line.block) === 1 && line.sourceLine !== undefined)
        fiscalDateScopes.set(line.block, { captionLine: line.sourceLine, caption: line, blocked: false });
      const fiscalScope = fiscalDateScopes.get(line.block);
      if (fiscalScope && (inferredDatePartyHere || unrelatedDateHeading.test(localHeading))) fiscalScope.blocked = true;
      const captionedFiscalDateAllowed = !line.table && fiscalCaptionCounts.get(line.block) === 1 && fiscalScope && !fiscalScope.blocked &&
        line.sourceLine !== undefined && line.sourceLine > fiscalScope.captionLine && line.sourceLine - fiscalScope.captionLine <= 128;
      if (
        /^(?:emitente|estabelecimento|fornecedor|dados do emitente)(?=\s|:|$)/.test(
          line.folded,
        )
      )
        merchantScope = "issuer";
      const merchantText = labelled(line, merchantLabel);
      // Currency needs no label/value or cell-size inference. This line is
      // either ordinary OCR or an origin of a validated literal table; raw
      // malformed/untyped table blocks never reach this evidence scan.
      if (/(?:^|[^\p{L}\p{N}])R\$(?!\p{L})/u.test(line.text))
        currencies.proposals.push({ value: "BRL", lines: [line] });
      if (
        merchantScope !== "other" &&
        narrowOriginalEdgeRegion(line) &&
        (merchantText !== undefined ||
          (merchantValue(line.text) !== undefined &&
            !unlabelledMerchantInstruction(line.text) &&
            (businessTitle.test(line.folded) ||
              legalBusinessTitle.test(line.folded))))
      )
        uncertainEdgeIssuer = true;
      if (!line.supportsFields) {
        // Context boundaries above still apply, but merged/multi-value rows
        // never borrow a value from another row or supply core candidates.
        if (merchantText !== undefined && merchantScope !== "other")
          merchants.seen = true;
        if (labelled(line, dateLabel) !== undefined) dates.seen = true;
        if (labelled(line, totalLabel) !== undefined) totals.seen = true;
        if (labelled(line, currencyLabel) !== undefined) currencies.seen = true;
        continue;
      }
      if (merchantText !== undefined && merchantScope !== "other") {
        merchants.seen = true;
        merchantLabelPages.add(line.page);
        const span = valueSpan(lines, index, merchantText),
          value = merchantValue(span.value);
        if (value && !narrowOriginalEdgeRegion(line))
          merchants.proposals.push({
            value,
            lines: span.lines,
            rank: line.folded.startsWith("nome fantasia")
              ? 3
              : line.folded.startsWith("razao social")
                ? 1
                : 2,
          });
      }
      const dateText = labelled(line, dateLabel) ??
        (line.table && documentDateAllowed ? labelled(line, ownOriginDocumentDateLabel) : undefined);
      if (dateText !== undefined) {
        dates.seen = true;
        const span = valueSpan(lines, index, dateText),
          value = calendarDate(span.value);
        if (value) dates.proposals.push({ value, lines: span.lines });
      }
      // Bare DATA needs one local financial-header witness. Page-wide word
      // matches, compressed line counters and dropped time suffixes could
      // promote a customer/body/birth/authorization date into an issue date.
      const localHeaderDate = localReceiptHeaderDate(ordinaryBlockHeaders.get(line.block) ?? [], line);
      if (localHeaderDate) {
        dates.seen = true;
        dates.proposals.push(localHeaderDate);
      }
      const receiptDate = line.table ? undefined : receiptIssueDate(line);
      if (receiptDate) {
        dates.seen = true;
        dates.proposals.push({ value: receiptDate, lines: [line] });
      }
      const adjacentFiscalDate = adjacentFiscalIssueDate(lines, index);
      if (adjacentFiscalDate) {
        dates.seen = true;
        dates.proposals.push(adjacentFiscalDate);
      }
      const documentIssueDate = documentDateAllowed ? ordinaryDocumentIssueDate(lines, index) : undefined;
      if (documentIssueDate) {
        dates.seen = true;
        dates.proposals.push(documentIssueDate);
      }
      const captionedFiscalDate = captionedFiscalDateAllowed ? captionedFiscalIssueDate(line) : undefined;
      if (captionedFiscalDate && fiscalScope) {
        dates.seen = true;
        dates.proposals.push({ value: captionedFiscalDate, lines: [fiscalScope.caption, line] });
      }
      const totalText =
        labelled(line, totalLabel) ?? receiptItemCountTotal(line);
      if (
        totalText !== undefined &&
        !/^(?:dos? |de )?(?:produtos?|itens|descontos?|tributos?|impostos?|parcelas?)\b/i.test(
          fold(totalText),
        )
      ) {
        totals.seen = true;
        const totalParts = totalCurrencyText(totalText),
          local = valueSpan(lines, index, totalParts.value),
          span = local.value === "" ? geometricTotals.get(line) ?? local : local,
          value = money(span.value);
        if (totalParts.currency)
          currencies.proposals.push({
            value: totalParts.currency,
            lines: [line],
          });
        if (value) {
          if (!payableSummaries.merchandise.has(line))
            totals.proposals.push({ value: value.value,
              lines: payableSummaries.evidence.get(line) ?? span.lines });
          if (value.currency)
            currencies.proposals.push({
              value: value.currency,
              lines: span.lines,
            });
        }
        const codedAmount = /^([A-Z]{3})\s+\d/.exec(span.value);
        if (codedAmount)
          currencies.proposals.push({
            value: codedAmount[1],
            lines: span.lines,
          });
        if (/(?:^|\s)\$\s*\d/.test(span.value)) bareCurrency = true;
      }
      const currencyText = labelled(line, currencyLabel);
      if (currencyText !== undefined) {
        currencies.seen = true;
        const span = valueSpan(lines, index, currencyText),
          value =
            span.value === "R$"
              ? "BRL"
              : /^[A-Za-z]{3}$/.test(span.value)
                ? span.value.toUpperCase()
                : undefined;
        if (value) currencies.proposals.push({ value, lines: span.lines });
        else if (span.value === "$") bareCurrency = true;
      }
    }
    if (!merchantLabelPages.has(1)) {
      const firstPage = lines
        .filter((line) => line.page === 1 && !line.table)
        .slice(0, 8);
      for (const [index, line] of firstPage.entries()) {
        if (
          firstPage
            .slice(0, index)
            .some((previous) => otherPartyHeading.test(previous.folded))
        )
          continue;
        const value = merchantValue(line.text);
        if (unlabelledMerchantInstruction(line.text)) continue;
        if (otherPartyHeading.test(line.folded)) continue;
        if (narrowOriginalEdgeRegion(line)) continue;
        const firstIssuerTitle = firstPage
          .slice(0, index)
          .every((previous) => merchantValue(previous.text) === undefined);
        const issuerIdNearby =
          firstIssuerTitle &&
          (businessTitle.test(line.folded) ||
            legalBusinessTitle.test(line.folded)) &&
          firstPage
            .slice(index + 1, index + 3)
            .some((_, offset) => printedIssuerNumber(firstPage, index + 1 + offset) !== undefined);
        const legalIssuerNearby =
          index < 6 &&
          legalBusinessTitle.test(line.folded) &&
          firstPage
            .slice(index + 1, index + 5)
            .some((_, offset) => printedIssuerNumber(firstPage, index + 1 + offset) !== undefined);
        if (
          value &&
          ((index < 3 && businessTitle.test(line.folded)) ||
            issuerIdNearby ||
            legalIssuerNearby)
        ) {
          merchants.proposals.push({ value, lines: [line], rank: 0 });
          // Equally supported issuer headers stay alternatives. Choosing the
          // first could combine one document's issuer with another's amounts.
        }
      }
    }
    const additionalHeaders = overlappingHeaderEvidence(lines);
    merchants.proposals.push(...additionalHeaders.proposals);
    merchants.proposals.push(...fiscalHeaderIssuers(lines));
    merchants.proposals.push(...taximeterHeaderProviders(lines));
    // An owned NFS-e name replaces only the generic parse of that same
    // original label. Independent regions and identities remain alternatives.
    merchants.proposals = merchants.proposals.filter(proposal => !proposal.lines.some(line => {
      const full = serviceInvoices.ownedNameOrigins.get(line);
      return full !== undefined && full !== proposal.value;
    }));
    merchants.proposals.push(...serviceInvoices.providers);
    samePageIssuerConflict ||= serviceInvoices.identityConflict;
    totals.seen ||= serviceInvoices.totalSeen;
    totals.proposals.push(...serviceInvoices.totals);
    currencies.proposals.push(...serviceInvoices.currencies);
    merchants.proposals.push(
      ...neighboringHeaderAlternatives(input.pages, lines, merchants.proposals),
    );
    const goodsIssuers = [
      ...typedGoodsStubMerchants(input.pages, merchants.proposals),
      ...ordinaryGoodsStubMerchants(lines, merchants.proposals),
    ];
    if (goodsIssuers.length) {
      merchants.seen = true;
      merchants.proposals.push(...goodsIssuers);
      const counter = retainedGoodsStubConflicts(input.pages, merchants.proposals);
      merchants.proposals.push(...counter.proposals);
      goodsConflictReviewIncomplete = counter.incomplete;
      normalizedIssuerConflict = normalizedIssuerIdentityConflict(merchants.proposals);
    }
    const fiscalCaptionReview = fiscalCaptionMerchantReview(lines, merchants.proposals);
    if (fiscalCaptionReview.excluded.size) {
      // Excluding a malformed inferred title cannot erase a different printed
      // issuer or let one page's clean company suppress another document.
      samePageIssuerConflict ||= contextualIssuerConflict(lines, merchants.proposals);
      crossPageIssuerConflict ||= crossPageIssuerNamesConflict(merchants.proposals);
      merchants.seen = true;
      merchants.proposals = merchants.proposals.filter(proposal => !fiscalCaptionReview.excluded.has(proposal));
      warn("low_quality", "An unlabelled issuer-like reading starts with a fiscal document marker. It remains in raw evidence; no company name was created by stripping the marker.");
    }
    const headerPrefixReview = incompleteSmallHeaderPrefixes(
      lines,
      merchants.proposals,
    );
    const incompleteHeaderProposals = headerPrefixReview.incomplete;
    if (incompleteHeaderProposals.size && !headerPrefixReview.budgetExhausted) {
      // Invalidating a fragment does not remove its printed issuer identity or
      // separate documents. Preserve the original conflicts before filtering.
      samePageIssuerConflict ||= contextualIssuerConflict(lines, merchants.proposals);
      crossPageIssuerConflict ||= crossPageIssuerNamesConflict(merchants.proposals);
      merchants.seen = true;
      merchants.proposals = merchants.proposals.filter(
        (proposal) => !incompleteHeaderProposals.has(proposal),
      );
      warn(
        "low_quality",
        "A small-header issuer read ends inside a longer literal word retained in another bounded header context; the incomplete read requires inspection.",
      );
    }
    uncertainEdgeIssuer ||= additionalHeaders.uncertainEdgeIssuer;
    // Postal receipts explicitly call the shipping charge "Valor do Porte".
    // Require a postal issuer/header and equal final amount received, so postage
    // is never mistaken for one component of another invoice or insured total.
    if (!totals.proposals.length) {
      const postalPages = new Set(
        input.pages
          .filter(
            (p) =>
              lines.some(
                (line) =>
                  line.page === p.page &&
                  !line.table &&
                  /correios e tel[eé]grafos/i.test(line.text),
              ) &&
              lines.some(
                (line) =>
                  line.page === p.page &&
                  !line.table &&
                  /comprovante do cliente/i.test(line.text),
              ),
          )
          .map((p) => p.page),
      );
      const postageByPage = new Map<number, Map<number, Line>>(),
        receivedByPage = new Map<number, Map<number, Line>>();
      for (const line of lines) {
        if (!postalPages.has(line.page)) continue;
        const amount = /^valor do porte\s*\(\s*R\$\s*\)[\s.:\-]*(.+)$/iu.exec(
            line.text,
          ),
          paid =
            /^valor recebido\s*\(\s*R\$\s*(?:=>?)?\s*\)\s*(?:=>?)?[\s.:\-]*(.+)$/iu.exec(
              line.text,
            );
        for (const [match, table] of [
          [amount, postageByPage],
          [paid, receivedByPage],
        ] as const) {
          if (!match) continue;
          const parsed = money(match[1]);
          if (!parsed) continue;
          const byValue = table.get(line.page) ?? new Map<number, Line>();
          if (!byValue.has(parsed.value)) byValue.set(parsed.value, line);
          table.set(line.page, byValue);
        }
      }
      for (const [number, postages] of postageByPage) {
        const received = receivedByPage.get(number);
        if (postages.size !== 1 || received?.size !== 1) continue;
        const [value, line] = [...postages][0],
          paid = received.get(value);
        if (!paid) continue;
        totals.seen = true;
        totals.proposals.push({ value, lines: [line, paid] });
        currencies.proposals.push({ value: "BRL", lines: [line, paid] });
      }
    }

    if (merchants.proposals.length) {
      // Check identities before trading-name priority removes lower-ranked
      // legal names. Two contextual issuer IDs cannot be treated as aliases.
      samePageIssuerConflict ||= contextualIssuerConflict(
        lines,
        merchants.proposals,
      );
      const byPage = new Map<number, Proposal<string>[]>();
      for (const proposal of merchants.proposals) {
        const page = proposal.lines[0].page;
        const candidates = byPage.get(page) ?? [];
        candidates.push(proposal);
        byPage.set(page, candidates);
      }
      merchants.proposals = [...byPage.values()].flatMap((candidates) => {
        const rank = candidates.reduce(
          (highest, proposal) => Math.max(highest, proposal.rank ?? 0),
          0,
        );
        return candidates.filter((proposal) => proposal.rank === rank);
      });
      // A trading name on one page cannot suppress another page's issuer.
      // Do not combine field evidence across those unresolved identities.
      crossPageIssuerConflict ||=
        byPage.size > 1 &&
        new Set(
          merchants.proposals.map((proposal) =>
            (proposal.issuerIdentity ?? proposal.value).normalize("NFC").toLocaleLowerCase("pt-BR"),
          ),
        ).size > 1;
    }
    const currency = field("currency", currencies);
    if (bareCurrency && currency.status === "missing") {
      currency.status = "unsupported";
      currency.reason = "A bare dollar symbol does not identify a currency.";
    }
    const total = field("total", totals);
    if (
      totals.seen &&
      (totals.proposals.length > 0 ||
        (currency.status === "read" &&
          currency.candidates[0].value !== "BRL")) &&
      (currency.status !== "read" || currency.candidates[0].value !== "BRL")
    ) {
      total.status = "unsupported";
      total.candidates = [];
      total.reason =
        "Integer minor-unit normalization requires one explicitly supported BRL currency; other exponents are unsupported.";
      warn(
        "unsupported_currency",
        "The total cannot be normalized under the supported BRL minor-unit rules.",
      );
    }
    if (payableSummaries.unresolved) {
      total.status = "unsupported";
      total.candidates = [];
      total.reason = "An owned fiscal summary has an unreadable final payable. Its merchandise amount cannot replace that missing literal.";
      warn("low_quality", "The printed final payable cannot be safely read; no amount was calculated or substituted from merchandise.");
    }
    if (serviceInvoices.unresolved) {
      total.status = "unsupported";
      total.candidates = [];
      total.reason = "An NFS-e amount role has an unreadable or unowned value; no other service/net amount was substituted.";
      warn("low_quality", "A printed NFS-e service/net/final amount could not be safely read or owned; no amount was calculated or substituted.");
    } else if (serviceInvoices.differingAmounts && total.status === "ambiguous") {
      total.reason = "Printed NFS-e service and net amounts differ. Their roles remain distinct alternatives for human review.";
    }
    fields = {
      merchant: field("merchant", merchants),
      date: field("date", dates),
      total,
      currency,
    };
    if (incompleteHeaderProposals.size && fields.merchant.status === "unreadable")
      fields.merchant.reason =
        "An incomplete small-header word prefix was not selected as a complete issuer. Inspect the unchanged raw header evidence.";
    if (headerPrefixReview.budgetExhausted) {
      fields.merchant = {
        status: "unsupported",
        reason: "The bounded header-completeness comparison did not finish; inspect the unchanged issuer evidence before selecting a merchant.",
        candidates: [],
      };
      warn("low_quality", "Header-completeness comparison reached its bounded work limit; no unchecked merchant was selected.");
    }
    if (fiscalCaptionReview.unresolved) {
      fields.merchant = { status: "unsupported", candidates: [],
        reason: "A fiscal-marker-prefixed inferred title has no independently owned matching literal issuer. Inspect the unchanged source before choosing a merchant." };
    }
    if (uncertainEdgeIssuer) {
      // Refusing only the issuer still exposed date/amount/currency proposals
      // without a resolved document boundary. The entire original is the unit
      // of interpretation; never silently select its more complete receipt.
      multiple = true;
      fields = unsupportedFields(
        "Unresolved issuer-like evidence at an original edge requires manual boundary review before selecting any core field.",
      );
      warn(
        "low_quality",
        "Narrow original-edge issuer evidence is unresolved; no main document was selected automatically.",
      );
      warn(
        "multiple_documents",
        "An unresolved issuer-like fragment at an original edge may belong to another document. Core fields remain withheld for boundary review; no physical document count or separation was inferred.",
      );
    }
    if (goodsConflictReviewIncomplete) {
      fields.merchant = { status: "unsupported", candidates: [],
        reason: "Retained producer evidence could not be completely owned, interpreted or reviewed; merchant identity requires manual inspection." };
      warn("low_quality", "Retained producer-conflict evidence has unresolved ownership, interpretation or review coverage; no unchecked sole merchant was selected.");
    }
    if (!multiple) for (const [name, result] of Object.entries(fields))
      if (result.status !== "read") {
        const code =
          result.status === "ambiguous" ||
          (result.status === "unsupported" &&
            result.reason?.startsWith("More than five"))
            ? "ambiguous_field"
            : result.status === "missing"
              ? "missing_field"
              : result.status === "unsupported" && name === "currency"
                ? "unsupported_currency"
                : "unreadable_field";
        warn(code, `The ${name} field requires manual inspection.`);
      }
  }
  if (crossPageIssuerConflict || samePageIssuerConflict || normalizedIssuerConflict) {
    multiple = true;
    fields = unsupportedFields(
      normalizedIssuerConflict
        ? "Distinct complete issuer/store-code declarations normalize to the same merchant value; manual identity or document-boundary review is required before using core fields."
        : samePageIssuerConflict
        ? "Distinct contextual issuer names and printed CNPJs on one page require manual identity resolution or document separation."
        : "Distinct issuer names across pages require manual document separation or identity resolution.",
    );
    warn(
      "multiple_documents",
      normalizedIssuerConflict
        ? "Different source issuer/store-code declarations remain unresolved; no source identity or document boundary was selected after merchant normalization."
        : samePageIssuerConflict
        ? "Distinct named issuer/CNPJ associations cannot be combined into one merchant and total; core fields remain unapplied."
        : "Different printed issuer identities across pages may belong to separate documents; core fields remain unapplied.",
    );
  }
  if (panelReview.incomplete) {
    // Failure to finish this additional review cannot erase an already
    // established issuer/CNPJ or cross-page identity conflict.
    if (!multiple)
      fields = unsupportedFields("The bounded transaction-panel review could not finish; manual boundary review is required before selecting core fields.");
    warn("low_quality", "Transaction-panel review reached its bounded work limit; no unchecked core fields were selected.");
  }
  if (warnings.length > 100)
    throw new Error("OCR normalization warnings exceed the output limit.");
  const result: OcrNormalizedResult = {
    mode: input.mode,
    methodsUsed: input.methodsUsed,
    documentType: multiple ? "mixed" : (input.documentType ?? "unknown"),
    pageCount: input.pages.length,
    pages: input.pages,
    fields,
    warnings,
    verification: { fiscalAuthenticity: false, bankVerification: false },
  };
  const complete =
    input.pages.every((page) => page.status === "read") &&
    Object.values(fields).every((value) => value.status === "read") &&
    !hasIncompleteOcrCoverage(result);
  return { state: complete ? "completed" : "partial", result };
}
