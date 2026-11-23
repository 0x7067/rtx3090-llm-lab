import { z } from "zod";
import {
  journalFieldsSchema,
  journalLineSchema,
  type Journal,
} from "./accounting.js";

// Empty lines explicitly declare a zero opening; ordinary journals still require two.
export const openingFieldsSchema = journalFieldsSchema
  .safeExtend({ lines: z.array(journalLineSchema).max(100) })
  .superRefine((fields, ctx) => {
    if (!fields.date.endsWith("-01"))
      ctx.addIssue({
        code: "custom",
        path: ["date"],
        message: "A abertura deve vigorar no primeiro dia da competência.",
      });
    if (
      new Set(fields.lines.map((line) => line.accountId)).size !==
      fields.lines.length
    )
      ctx.addIssue({
        code: "custom",
        path: ["lines"],
        message: "Informe o saldo líquido de cada conta uma única vez.",
      });
    if (fields.lines.some((line) => line.costCenter !== ""))
      ctx.addIssue({
        code: "custom",
        path: ["lines"],
        message: "A abertura não atribui saldos por centro de custo.",
      });
  });

export interface OpeningDeclaration {
  effectiveFrom: string;
  evidence: "human-declared-unverified";
}
export type BalanceJournal = Pick<
  Journal,
  | "id"
  | "date"
  | "description"
  | "reference"
  | "version"
  | "state"
  | "reversalOf"
  | "reversedByJournal"
>;
export interface AccountingBalances {
  month: string;
  anchor: (BalanceJournal & { opening: OpeningDeclaration }) | null;
  carried: BalanceJournal[];
  movements: BalanceJournal[];
  rows: {
    account: Journal["accounts"][number];
    openingNet: number | null;
    debit: number;
    credit: number;
    closingNet: number | null;
  }[];
  totals: {
    openingDebit: number | null;
    openingCredit: number | null;
    debit: number;
    credit: number;
    closingDebit: number | null;
    closingCredit: number | null;
  };
  basisToken: string;
  scope: "local-declared-ledger-balances";
  bankConfirmed: false;
  erpConfirmed: false;
  fiscalCertified: false;
}
export interface AccountingBalanceSnapshot extends AccountingBalances {
  closedBy: string;
  closedByName: string;
  closedAt: string;
}
