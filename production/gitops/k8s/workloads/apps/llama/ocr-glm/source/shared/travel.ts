import { z } from "zod";
import { payableDate } from "./payables.js";
import type { AuditEvent, Document, ExpenseAllocation } from "./types.js";
import type { MileageDeclaration } from "./mileage.js";

export const travelReason = z.string().trim().min(5).max(1500);
export const travelVersion = z.number().int().positive();
export const travelEstimateKind = z.enum([
  "transport",
  "lodging",
  "meals",
  "local-transport",
  "other",
]);
export const travelFieldsSchema = z
  .object({
    title: z.string().trim().min(3).max(160),
    destination: z.string().trim().min(2).max(160),
    purpose: z.string().trim().min(5).max(1500),
    startsOn: payableDate,
    endsOn: payableDate,
    costCenter: z.string().trim().min(1).max(100),
    estimates: z
      .array(
        z
          .object({
            kind: travelEstimateKind,
            description: z.string().trim().min(3).max(200),
            amount: z.number().int().positive().max(100_000_000),
          })
          .strict(),
      )
      .min(1)
      .max(30),
    advanceRequested: z.number().int().min(0).max(100_000_000),
  })
  .strict()
  .superRefine((v, ctx) => {
    const days = (Date.parse(v.endsOn) - Date.parse(v.startsOn)) / 86_400_000;
    if (days < 0 || days > 366)
      ctx.addIssue({
        code: "custom",
        path: ["endsOn"],
        message:
          "O término deve ser no início ou depois, com até 366 dias entre as datas (367 dias contando ambos).",
      });
    const total = v.estimates.reduce((n, item) => n + item.amount, 0);
    if (total > 100_000_000)
      ctx.addIssue({
        code: "custom",
        path: ["estimates"],
        message: "O orçamento máximo é R$ 1.000.000,00.",
      });
    if (v.advanceRequested > total)
      ctx.addIssue({
        code: "custom",
        path: ["advanceRequested"],
        message:
          "O adiantamento solicitado não pode superar o orçamento estimado.",
      });
  });
export type TravelFields = z.infer<typeof travelFieldsSchema>;
export const travelTotal = (fields: TravelFields) =>
  fields.estimates.reduce((n, item) => n + item.amount, 0);

const destinationText = (value: string) => value.trim().replace(/\s+/gu, " ");
// Exact text matching only. Do not infer a municipality, region or country.
export const travelDestinationKey = (value: string) =>
  destinationText(value)
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .toLocaleLowerCase("pt-BR");
export const travelPolicyRuleSchema = z
  .object({
    id: z.string().uuid(),
    kind: travelEstimateKind,
    destination: z
      .string()
      .trim()
      .min(2)
      .max(160)
      .transform(destinationText)
      .nullable(),
    unit: z.enum(["trip", "day", "night"]),
    limit: z.number().int().min(0).max(100_000_000),
  })
  .strict()
  .superRefine((rule, ctx) => {
    if (
      (rule.unit === "night" && rule.kind !== "lodging") ||
      (rule.unit === "day" &&
        !["meals", "local-transport", "other"].includes(rule.kind))
    )
      ctx.addIssue({
        code: "custom",
        path: ["unit"],
        message:
          "Use por viagem para qualquer categoria, por noite apenas para hospedagem ou por dia para alimentação, transporte local e outras estimativas.",
      });
  });
export type TravelPolicyRule = z.infer<typeof travelPolicyRuleSchema>;
export const travelPolicyFieldsSchema = z
  .object({
    maxTrip: z.number().int().min(0).max(100_000_000).nullable(),
    maxAdvance: z.number().int().min(0).max(100_000_000).nullable(),
    // Omission preserves the saved rules and the digest of older commands.
    rules: z.array(travelPolicyRuleSchema).max(32).optional(),
  })
  .strict()
  .superRefine((fields, ctx) => {
    const ids = new Set<string>(),
      scopes = new Set<string>();
    for (const [index, rule] of (fields.rules ?? []).entries()) {
      if (ids.has(rule.id))
        ctx.addIssue({
          code: "custom",
          path: ["rules", index, "id"],
          message: "Cada regra precisa de um identificador único.",
        });
      ids.add(rule.id);
      const scope = JSON.stringify([
        rule.kind,
        rule.destination === null
          ? null
          : travelDestinationKey(rule.destination),
      ]);
      if (scopes.has(scope))
        ctx.addIssue({
          code: "custom",
          path: ["rules", index, "destination"],
          message:
            "Use somente uma regra por categoria e destino, independentemente da unidade.",
        });
      scopes.add(scope);
    }
  });
export type TravelPolicyFields = z.infer<typeof travelPolicyFieldsSchema>;
export interface TravelPolicy extends TravelPolicyFields {
  rules: TravelPolicyRule[];
  version: number;
  updatedAt: string | null;
  updatedBy: string | null;
  history: AuditEvent[];
}
export interface TravelPolicyRuleCheck {
  rule: TravelPolicyRule;
  actualAmount: number;
  units: number;
  allowedAmount: number;
  excess: number;
}
export interface TravelPolicyCheck {
  policyVersion: number;
  maxTrip: number | null;
  maxAdvance: number | null;
  tripExcess: number;
  advanceExcess: number;
  exception: boolean;
  // Optional solely for unchanged approval snapshots from older policy versions.
  tripDays?: number;
  nights?: number;
  ruleChecks?: TravelPolicyRuleCheck[];
}
export function checkTravelPolicy(
  fields: TravelFields,
  policy: Pick<TravelPolicy, "version" | "maxTrip" | "maxAdvance"> & {
    rules?: TravelPolicyRule[];
  },
): TravelPolicyCheck {
  const nights =
    (Date.parse(fields.endsOn) - Date.parse(fields.startsOn)) / 86_400_000;
  const tripDays = nights + 1;
  if (!Number.isSafeInteger(nights) || nights < 0)
    throw new Error(
      "Datas inválidas para calcular os dias e noites da viagem.",
    );
  const destination = travelDestinationKey(fields.destination);
  const rules = policy.rules ?? [];
  const ruleChecks = travelEstimateKind.options.flatMap((kind) => {
    const candidates = rules.filter((rule) => rule.kind === kind);
    const rule =
      candidates.find(
        (candidate) =>
          candidate.destination !== null &&
          travelDestinationKey(candidate.destination) === destination,
      ) ?? candidates.find((candidate) => candidate.destination === null);
    if (!rule) return [];
    const actualAmount = fields.estimates
      .filter((item) => item.kind === kind)
      .reduce((total, item) => total + item.amount, 0);
    const units =
      rule.unit === "trip" ? 1 : rule.unit === "day" ? tripDays : nights;
    const allowedAmount = units * rule.limit;
    if (
      !Number.isSafeInteger(actualAmount) ||
      !Number.isSafeInteger(allowedAmount) ||
      actualAmount < 0 ||
      allowedAmount < 0
    )
      throw new Error(
        "O cálculo da política excede o limite seguro de centavos.",
      );
    return [
      {
        rule: { ...rule },
        actualAmount,
        units,
        allowedAmount,
        excess: Math.max(0, actualAmount - allowedAmount),
      },
    ];
  });
  const tripExcess =
    policy.maxTrip === null
      ? 0
      : Math.max(0, travelTotal(fields) - policy.maxTrip);
  const advanceExcess =
    policy.maxAdvance === null
      ? 0
      : Math.max(0, fields.advanceRequested - policy.maxAdvance);
  return {
    policyVersion: policy.version,
    maxTrip: policy.maxTrip,
    maxAdvance: policy.maxAdvance,
    tripExcess,
    advanceExcess,
    tripDays,
    nights,
    ruleChecks,
    exception:
      tripExcess > 0 ||
      advanceExcess > 0 ||
      ruleChecks.some((check) => check.excess > 0),
  };
}

export type TravelState =
  | "draft"
  | "submitted"
  | "needs-info"
  | "approved"
  | "rejected"
  | "cancelled"
  | "closed";
export interface TravelApproval {
  fields: TravelFields;
  amount: number;
  center: { id: string; name: string; version: number } | null;
  actor: string;
  reason: string;
  at: string;
  policyCheck?: TravelPolicyCheck;
  exceptionReason?: string | null;
}
export interface TravelExpenseSnapshot {
  id: string;
  version: number;
  owner: string;
  merchant: string;
  amount: number;
  date: string;
  costCenter: string;
  allocations?: ExpenseAllocation[];
  mileage?: MileageDeclaration;
  reimbursementState: string | null;
  documents: Pick<Document, "id" | "name" | "sha256">[];
}
export interface TravelExpenseLink {
  id: string;
  expenseId: string;
  state: "proposed" | "accepted" | "rejected" | "released";
  source: TravelExpenseSnapshot;
  linkedAmount?: number;
  allocatedCostCenter?: string;
  advanceApplied: number;
  fundingOriginals?: Pick<Document, "sha256" | "fiscal">[];
  reason: string;
  preparedAt: string;
  preparedBy: string;
  reviewedAt: string | null;
  reviewedBy: string | null;
  reviewReason: string | null;
  current: boolean;
  issues: string[];
}
export interface TravelCashRecord {
  id: string;
  kind: "advance" | "return";
  state: "proposed" | "accepted" | "rejected" | "voided";
  amount: number;
  reportedOn: string;
  reference: string;
  reason: string;
  preparedAt: string;
  reviewedAt: string | null;
  reviewReason: string | null;
  bankConfirmed: false;
}
export interface TravelSettlement {
  approvedBudget: number | null;
  linkedGross: number;
  advanceReported: number;
  advanceApplied: number;
  returnReported: number;
  provisionalOutstanding: number;
  bankConfirmed: false;
  actualSpend: null;
  notice: string;
}
export interface TravelRequest extends TravelFields {
  id: string;
  owner: string;
  employee: string;
  version: number;
  state: TravelState;
  estimatedTotal: number;
  approved: TravelApproval | null;
  policyCheck: TravelPolicyCheck;
  expenseLinks: TravelExpenseLink[];
  cashRecords: TravelCashRecord[];
  settlement: TravelSettlement;
  history: AuditEvent[];
  createdAt: string;
  updatedAt: string;
}
