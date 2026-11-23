import { z } from "zod";
import { payableDate } from "./payables.js";
import type { Expense, ExpenseAllocation } from "./types.js";
import type { ApprovedExpenseFundingSource } from "./expense-funding.js";
export interface ReimbursementReview {
  expense: Expense;
  source: ApprovedExpenseFundingSource | null;
  basisToken: string;
  cardReserved: number;
  reimbursementReserved: number;
  reserved: number;
  available: number;
  maximum: number;
  overReserved: boolean;
  parts: (ExpenseAllocation & {
    cardReserved: number;
    reimbursementReserved: number;
    reserved: number;
    available: number;
    schedulable: number;
  })[];
  holds: string[];
  sourceIssues: string[];
  legacy: boolean;
  canSchedule: boolean;
  canReportPaid: boolean;
  canCancel: boolean;
  bankConfirmed: false;
}
const common = {
  version: z.number().int().positive(),
  basisToken: z.string().regex(/^[a-f0-9]{64}$/),
  requestKey: z.string().uuid(),
  date: payableDate,
  reference: z.string().trim().min(5).max(200),
  reason: z.string().trim().min(10).max(1500),
};
export const reimbursementCommandSchema = z.discriminatedUnion("action", [
  z
    .object({
      ...common,
      action: z.literal("schedule"),
      amount: z.number().int().positive().max(100_000_000),
      parts: z
        .array(
          z
            .object({
              allocationId: z.string().min(1).max(80),
              amount: z.number().int().positive().max(100_000_000),
            })
            .strict(),
        )
        .min(1)
        .max(30)
        .optional(),
      understood: z.literal(true),
    })
    .strict(),
  z
    .object({
      ...common,
      action: z.literal("report-paid"),
      understood: z.literal(true),
    })
    .strict(),
  z.object({ ...common, action: z.literal("cancel") }).strict(),
]);
export type ReimbursementCommand = z.infer<typeof reimbursementCommandSchema>;
