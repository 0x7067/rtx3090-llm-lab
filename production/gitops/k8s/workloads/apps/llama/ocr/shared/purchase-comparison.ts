import { z } from "zod";
import type {
  PurchaseFields,
  PurchaseQuote,
  PurchaseRequest,
} from "./purchasing.js";
const day = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(`${v}T12:00:00Z`);
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  }, "Informe uma data válida.");
const text = z.string().trim().min(5).max(1500);
export const quoteComparisonFieldsSchema = z
  .object({
    scopeBasis: text,
    offers: z
      .array(
        z
          .object({
            id: z.string().uuid(),
            vendor: z.string().trim().min(2).max(160),
            quoteId: z.string().uuid(),
            amount: z.number().int().min(1).max(100_000_000),
            quotedOn: day,
            validUntil: day.nullable(),
            leadDays: z.number().int().min(0).max(366).nullable(),
            conditions: text,
          })
          .strict()
          .refine(
            (o) => !o.validUntil || o.validUntil >= o.quotedOn,
            "A validade não pode anteceder a cotação.",
          ),
      )
      .min(2)
      .max(5),
    selectedOfferId: z.string().uuid(),
    selectionReason: text,
  })
  .strict()
  .superRefine((v, ctx) => {
    for (const [key, values] of [
      ["id", v.offers.map((o) => o.id)],
      ["quoteId", v.offers.map((o) => o.quoteId)],
      ["vendor", v.offers.map((o) => normalizeVendor(o.vendor))],
    ] as const)
      if (new Set(values).size !== values.length)
        ctx.addIssue({
          code: "custom",
          path: ["offers"],
          message: `Não repita ${key === "vendor" ? "fornecedores" : key === "quoteId" ? "originais" : "identificadores"} na comparação.`,
        });
    if (!v.offers.some((o) => o.id === v.selectedOfferId))
      ctx.addIssue({
        code: "custom",
        path: ["selectedOfferId"],
        message: "Escolha uma proposta registrada.",
      });
  });
export const quoteComparisonCommandSchema = z
  .object({
    fields: quoteComparisonFieldsSchema.nullable(),
    version: z.number().int().positive(),
    reason: text,
    requestKey: z.string().uuid(),
  })
  .strict();
export type QuoteComparisonFields = z.infer<typeof quoteComparisonFieldsSchema>;
export type QuoteComparisonCommand = z.infer<
  typeof quoteComparisonCommandSchema
>;
export interface QuoteComparison extends QuoteComparisonFields {
  scope: PurchaseFields;
  preparedBy: string;
  preparedByName: string;
  preparedAt: string;
  reason: string;
  documents: PurchaseQuote[];
}
export const normalizeVendor = (s: string) =>
  s.trim().replace(/\s+/g, " ").toLocaleLowerCase("pt-BR");
export function comparisonScope(p: PurchaseFields): PurchaseFields {
  return {
    title: p.title,
    vendor: p.vendor,
    purpose: p.purpose,
    month: p.month,
    costCenter: p.costCenter,
    category: p.category,
    items: p.items,
  };
}
export function comparisonIssues(
  c: QuoteComparison | undefined,
  p: PurchaseFields,
  quotes: PurchaseQuote[],
  on: string,
): string[] {
  if (!c) return [];
  const issues: string[] = [];
  const selected = c.offers.find((o) => o.id === c.selectedOfferId);
  if (JSON.stringify(c.scope) !== JSON.stringify(comparisonScope(p)))
    issues.push(
      "O escopo mudou após a comparação. Confira as propostas e registre uma comparação atual antes do envio.",
    );
  if (
    !selected ||
    normalizeVendor(selected.vendor) !== normalizeVendor(p.vendor) ||
    selected.amount !==
      p.items.reduce((n, i) => n + i.quantity * i.unitPrice, 0)
  )
    issues.push(
      "O fornecedor e o total da proposta escolhida devem corresponder aos dados da solicitação. Preserve o original e corrija explicitamente os dados ou a escolha.",
    );
  if (
    c.offers.some(
      (o) =>
        !quotes.some(
          (q) =>
            q.id === o.quoteId &&
            c.documents.some((d) => d.id === q.id && d.sha256 === q.sha256),
        ),
    )
  )
    issues.push(
      "A comparação exige um original retido e distinto para cada proposta.",
    );
  if (c.offers.some((o) => o.quotedOn > on))
    issues.push(
      "A data informada de cotação não pode estar no futuro em São Paulo.",
    );
  if (selected?.validUntil && selected.validUntil < on)
    issues.push(
      "A proposta escolhida expirou. Obtenha evidência atual e prepare nova comparação antes de aprovar.",
    );
  return issues;
}
export function comparisonReviewDay(
  p: Pick<PurchaseRequest, "status" | "approval">,
) {
  return p.status === "approved" && p.approval?.at
    ? new Date(p.approval.at).toLocaleDateString("en-CA", {
        timeZone: "America/Sao_Paulo",
      })
    : new Date().toLocaleDateString("en-CA", { timeZone: "America/Sao_Paulo" });
}
