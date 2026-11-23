import { z } from "zod";

export const itemUnits = [
  "un",
  "kg",
  "g",
  "L",
  "m",
  "m²",
  "m³",
  "h",
  "dia",
  "cx",
  "pct",
] as const;
export type ItemUnit = (typeof itemUnits)[number];
const precision = /^-?\d+(?:\.\d{1,3})?$/;
export function quantityThousandths(quantity: number): number {
  if (!Number.isFinite(quantity) || !precision.test(String(quantity)))
    throw new Error("Quantidade inválida: use até três casas decimais.");
  const scaled = Math.round(quantity * 1000);
  if (!Number.isSafeInteger(scaled))
    throw new Error("Quantidade fora do limite seguro.");
  return scaled;
}
export function validQuantity(quantity: number): boolean {
  try {
    return (
      quantity >= 0.001 &&
      quantity <= 10_000 &&
      quantityThousandths(quantity) > 0
    );
  } catch {
    return false;
  }
}
export const quantitySchema = z
  .number()
  .refine(
    validQuantity,
    "Informe uma quantidade de 0,001 a 10.000, com até três casas decimais.",
  );
export const itemSchema = z
  .object({
    description: z.string().trim().min(2).max(200),
    quantity: quantitySchema,
    unitPrice: z.number().int().min(1).max(100_000_000),
    unit: z.enum(itemUnits).optional(),
  })
  .strict()
  .superRefine((item, ctx) => {
    if (!Number.isInteger(item.quantity) && !item.unit)
      ctx.addIssue({
        code: "custom",
        path: ["unit"],
        message: "Declare a unidade do item fracionado.",
      });
    if (
      validQuantity(item.quantity) &&
      Number.isSafeInteger(item.unitPrice) &&
      item.unitPrice >= 1 &&
      item.unitPrice <= 100_000_000 &&
      lineTotal(item) < 1
    )
      ctx.addIssue({
        code: "custom",
        path: ["quantity"],
        message:
          "O total de cada item deve resultar em pelo menos R$ 0,01 após o arredondamento.",
      });
  });
export type PricedItem = z.infer<typeof itemSchema>;
export const itemUnit = (item: { unit?: ItemUnit } | undefined): ItemUnit =>
  item?.unit ?? "un";
export function lineTotal(
  item: Pick<PricedItem, "quantity" | "unitPrice">,
): number {
  const product = quantityThousandths(item.quantity) * item.unitPrice;
  if (!Number.isSafeInteger(product + 500) || product < 0)
    throw new Error("Valor do item fora do limite seguro.");
  return Math.floor((product + 500) / 1000);
}
export const itemsTotal = (
  items: readonly Pick<PricedItem, "quantity" | "unitPrice">[],
) => items.reduce((sum, item) => sum + lineTotal(item), 0);
export function sumQuantities(values: Iterable<number>): number {
  let total = 0;
  for (const value of values) {
    total += quantityThousandths(value);
    if (!Number.isSafeInteger(total))
      throw new Error("Soma de quantidades fora do limite seguro.");
  }
  return total / 1000;
}
export function subtractQuantities(
  value: number,
  ...deductions: number[]
): number {
  let total = quantityThousandths(value);
  for (const deduction of deductions) {
    total -= quantityThousandths(deduction);
    if (!Number.isSafeInteger(total))
      throw new Error("Saldo de quantidades fora do limite seguro.");
  }
  return total / 1000;
}
export function parseQuantity(text: string): number {
  const value = text.trim();
  if (!/^\d+(?:[.,]\d{1,3})?$/.test(value)) return NaN;
  const number = Number(value.replace(",", "."));
  return validQuantity(number) ? number : NaN;
}
/** Optional receiving/return rows: an empty or explicitly zero row is omitted. */
export function enteredQuantity(text: string): number | undefined {
  const value = text.trim();
  return !value || /^0+(?:[.,]0{1,3})?$/.test(value)
    ? undefined
    : parseQuantity(value);
}
export const formatQuantity = (quantity: number) =>
  new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 3 }).format(quantity);
export const quantityLabel = (quantity: number, item?: { unit?: ItemUnit }) =>
  `${formatQuantity(quantity)} ${item?.unit ?? "unidade(s)"}`;
