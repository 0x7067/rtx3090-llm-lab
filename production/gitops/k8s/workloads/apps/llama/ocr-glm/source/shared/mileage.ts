import { z } from "zod";
import { expenseAllocationsSchema } from "./expense-allocations.js";
import { payableDate } from "./payables.js";
import type { Expense } from "./types.js";

export const mileageRateFieldsSchema = z
  .object({
    effectiveOn: payableDate,
    enabled: z.boolean(),
    rateCentsPerKm: z.number().int().min(1).max(100_000),
  })
  .strict();
export type MileageRateFields = z.infer<typeof mileageRateFieldsSchema>;
export interface MileageRateRevision extends MileageRateFields {
  version: number;
  actor: string;
  reason: string;
  createdAt: string;
}
export interface MileageRateRegister {
  version: number;
  revisions: MileageRateRevision[];
}
export const mileageExpenseFieldsSchema = z
  .object({
    origin: z.string().trim().max(200),
    destination: z.string().trim().max(200),
    distanceMeters: z.number().int().min(0).max(10_000_000),
    date: payableDate,
    costCenter: z.string().trim().max(100),
    purpose: z.string().trim().max(1500),
    allocations: expenseAllocationsSchema.refine(rows => !rows.some(row => row.servicePeriod), "Quilometragem usa o mês do trajeto; não informe período de serviço.").optional(),
  })
  .strict();
export type MileageExpenseFields = z.infer<typeof mileageExpenseFieldsSchema>;
export interface MileageDeclaration {
  origin: string;
  destination: string;
  distanceMeters: number;
  method: "manual";
  rate: MileageRateRevision | null;
  calculatedAmount: number;
  rounding: "half-up-cent";
  calculatedAt: string;
}
export const mileageReason = z.string().trim().min(5).max(1500);
export const mileageCommandSchema = z
  .object({
    fields: mileageExpenseFieldsSchema,
    rateVersion: z.number().int().positive().nullable(),
    reason: mileageReason,
    requestKey: z.string().uuid(),
  })
  .strict();
export const mileageEditSchema = mileageCommandSchema.extend({
  version: z.number().int().positive(),
});
export type MileageCreateCommand = z.infer<typeof mileageCommandSchema>;
export type MileageEditCommand = z.infer<typeof mileageEditSchema>;

export function selectMileageRate(
  revisions: readonly MileageRateRevision[],
  date: string,
): MileageRateRevision | null {
  return (
    revisions
      .filter((rate) => rate.effectiveOn <= date)
      .sort(
        (a, b) =>
          b.effectiveOn.localeCompare(a.effectiveOn) || b.version - a.version,
      )[0] ?? null
  );
}

export function calculateMileageAmount(
  distanceMeters: number,
  rateCentsPerKm: number,
): number {
  if (
    !Number.isSafeInteger(distanceMeters) ||
    distanceMeters < 0 ||
    distanceMeters > 10_000_000 ||
    !Number.isSafeInteger(rateCentsPerKm) ||
    rateCentsPerKm < 1 ||
    rateCentsPerKm > 100_000
  )
    throw new Error("Confira a distância e a tarifa por quilômetro.");
  const amount = Number(
    (BigInt(distanceMeters) * BigInt(rateCentsPerKm) + 500n) / 1000n,
  );
  if (amount > 100_000_000)
    throw new Error("O cálculo excede o limite de valor da despesa.");
  return amount;
}

// Exact decimal parsing: a kilometer has 1,000 meters. Grouping separators are not accepted.
export function parseMileageKilometers(value: string): number | null {
  const match = /^(\d+)(?:[.,](\d{1,3}))?$/.exec(value.trim());
  if (!match) return null;
  const meters =
    BigInt(match[1]) * 1000n + BigInt((match[2] ?? "").padEnd(3, "0"));
  return meters <= 10_000_000n ? Number(meters) : null;
}

export function isCompleteMileage(
  mileage: MileageDeclaration | undefined | null,
): boolean {
  if (
    !mileage ||
    mileage.method !== "manual" ||
    mileage.rounding !== "half-up-cent" ||
    !mileage.origin.trim() ||
    !mileage.destination.trim() ||
    mileage.distanceMeters <= 0 ||
    !mileage.rate?.enabled ||
    mileage.calculatedAmount <= 0
  )
    return false;
  try {
    return (
      calculateMileageAmount(
        mileage.distanceMeters,
        mileage.rate.rateCentsPerKm,
      ) === mileage.calculatedAmount
    );
  } catch {
    return false;
  }
}

export function hasExpenseEvidence(
  expense: Pick<
    Expense,
    "mileage" | "documents" | "amount" | "category" | "allocations"
  >,
): boolean {
  return expense.mileage
    ? isCompleteMileage(expense.mileage) &&
        expense.amount === expense.mileage.calculatedAmount &&
        expense.category === "Transporte" &&
        !(expense.allocations ?? []).some(
          (row) => row.category !== "Transporte",
        )
    : expense.documents.length > 0;
}
