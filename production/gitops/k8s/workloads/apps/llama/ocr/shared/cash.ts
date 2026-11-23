import { z } from "zod";
import { payableDate } from "./payables.js";
import type { StatementAccountFields } from "./reconciliation.js";

export const cashReason = z.string().trim().min(5).max(1500);
export const cashVersion = z.number().int().positive();
export const cashAccountFieldsSchema = z
  .object({
    lineId: z
      .string()
      .regex(
        /^(?:payable|purchase|reimbursement):[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
      ),
    anchorDate: payableDate,
    accountId: z.string().uuid(),
  })
  .strict();
export type CashAccountFields = z.infer<typeof cashAccountFieldsSchema>;

export interface CashAccountAttribution {
  id: string;
  version: number;
  state: "submitted" | "approved" | "rejected" | "cancelled";
  sourceCurrent: boolean;
  lineId: string;
  anchorDate: string;
  source: {
    kind: "payable" | "purchase" | "reimbursement_forecast";
    sourceId: string;
    sourceVersion: number | null;
    amount: number;
    date: string | null;
    month: string;
    evidence: "committed" | "forecast";
    fingerprint: string;
  };
  account: {
    id: string;
    label: string;
    mode: "manual-unverified";
  };
  preparedAt: string;
  updatedAt: string;
  preparedBy: string;
  preparationReason: string;
  approved: { by: string; at: string; reason: string } | null;
  decision: {
    by: string;
    at: string;
    action: "approve" | "reject" | "cancel";
    reason: string;
  } | null;
  history: {
    id: number;
    actor: string;
    action: string;
    note: string;
    createdAt: string;
  }[];
}
export const cashInputFieldsSchema = z
  .object({
    kind: z.enum(["receivable", "scenario_inflow", "scenario_outflow"]),
    title: z.string().trim().min(3).max(160),
    counterparty: z.string().trim().max(160),
    expectedOn: payableDate,
    amount: z.number().int().positive().max(100_000_000),
    assumption: z.string().trim().min(10).max(1500),
    reference: z.string().trim().max(160),
  })
  .strict()
  .refine(
    (value) => value.kind !== "receivable" || value.counterparty.length >= 2,
    { path: ["counterparty"], message: "Informe a contraparte prevista." },
  );
export type CashInputFields = z.infer<typeof cashInputFieldsSchema>;

export interface CashInput extends CashInputFields {
  id: string;
  version: number;
  state: "draft" | "submitted" | "approved" | "rejected" | "cancelled";
  createdAt: string;
  updatedAt: string;
  approved:
    | (CashInputFields & {
        by: string;
        at: string;
        reason: string;
      })
    | null;
  history: {
    id: number;
    actor: string;
    action: string;
    note: string;
    createdAt: string;
  }[];
}

export interface CashPlanLine {
  id: string;
  kind:
    | "statement"
    | "reimbursement_report"
    | "payment_simulation"
    | "payable"
    | "purchase"
    | "reimbursement_forecast"
    | "receivable"
    | "scenario";
  label: string;
  date: string | null;
  month: string;
  datePrecision: "day" | "month";
  direction: "inflow" | "outflow";
  amount: number;
  evidence:
    | "observed_unverified"
    | "reported_unverified"
    | "committed"
    | "forecast"
    | "simulated";
  sourceId: string;
  sourceVersion: number | null;
  basis: string;
  account: {
    id: string;
    label: string;
    mode: StatementAccountFields["mode"];
  } | null;
  plannedAccount: {
    attributionId: string;
    state: "submitted" | "approved";
    sourceCurrent: boolean;
    account: CashAccountAttribution["account"];
    preparationReason: string;
    decisionReason: string | null;
  } | null;
  reconciliationLinks: {
    matchId: string;
    entryId: string;
    entryDate: string;
    entryReference: string;
    targetKind: "reimbursement" | "payment_batch";
    targetId: string;
    targetLabel: string;
    kind: "payment" | "refund";
    amount: number;
    account: NonNullable<CashPlanLine["account"]>;
  }[];
  bankConfirmed: false;
}

export interface CashPlan {
  from: string;
  to: string;
  openingBalance: null;
  closingBalance: null;
  bankConfirmed: false;
  lines: CashPlanLine[];
  totals: Record<CashPlanLine["evidence"], { inflow: number; outflow: number }>;
  baselineNetChange: number;
  scenarioNetChange: number;
  notice: string;
}
