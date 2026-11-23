import { z } from "zod";
import type { Document, Status } from "./types.js";
const command = {
  version: z.number().int().positive(),
  reason: z.string().trim().min(5).max(1500),
  requestKey: z.string().uuid(),
};
export const expenseReservationCreateSchema = z
  .object({
    ...command,
    expenseId: z.string().min(1).max(100),
    expenseVersion: z.number().int().positive(),
    amount: z.number().int().positive().max(100_000_000),
  })
  .strict();
export const expenseReservationActionSchema = z
  .object({
    ...command,
    action: z.enum(["release", "convert"]),
    expenseVersion: z.number().int().positive().optional(),
  })
  .strict();
export type ExpenseReservationCreate = z.infer<
  typeof expenseReservationCreateSchema
>;
export type ExpenseReservationAction = z.infer<
  typeof expenseReservationActionSchema
>;
export interface ExpenseReservation {
  id: string;
  purchaseId: string;
  expenseId: string;
  amount: number;
  state: "active" | "released" | "converted";
  allocationId?: string;
  purchase: {
    number: number;
    title: string;
    month: string;
    costCenter: string;
  };
  source: {
    merchant: string;
    date: string;
    amount: number;
    centerAmount: number;
    owner: string;
    version: number;
    approvalRunId: string;
    fingerprint: string;
    documents: Pick<Document, "id" | "name" | "sha256" | "demo">[];
  };
  events: {
    state: ExpenseReservation["state"];
    actor: string;
    name: string;
    reason: string;
    at: string;
  }[];
}
export interface ExpenseReservationPosition extends ExpenseReservation {
  current: boolean;
  expenseStatus: Status | null;
  expenseVersion: number | null;
  warnings: string[];
}
