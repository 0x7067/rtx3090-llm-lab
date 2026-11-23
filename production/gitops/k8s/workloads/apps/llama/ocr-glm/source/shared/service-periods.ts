import { z } from "zod";
const date = z
  .string()
  .regex(/^20\d{2}-\d{2}-\d{2}$/)
  .refine((v) => {
    const d = new Date(v + "T12:00:00Z");
    return Number.isFinite(+d) && d.toISOString().slice(0, 10) === v;
  }, "Informe uma data válida para o período.");
export function serviceMonths(from: string, to: string): string[] {
  if (!date.safeParse(from).success || !date.safeParse(to).success || to < from)
    return [];
  const first = Number(from.slice(0, 4)) * 12 + Number(from.slice(5, 7)) - 1;
  const last = Number(to.slice(0, 4)) * 12 + Number(to.slice(5, 7)) - 1;
  if (last - first >= 36) return [];
  return Array.from({ length: last - first + 1 }, (_, i) => {
    const n = first + i;
    return `${Math.floor(n / 12)}-${String((n % 12) + 1).padStart(2, "0")}`;
  });
}
export const servicePeriodSchema = z
  .object({
    from: date,
    to: date,
    basis: z.string().trim().min(5).max(1500),
    months: z
      .array(
        z
          .object({
            month: z.string().regex(/^20\d{2}-(0[1-9]|1[0-2])$/),
            amount: z.number().int().positive().max(100_000_000),
          })
          .strict(),
      )
      .min(1)
      .max(36),
  })
  .strict()
  .superRefine((v, ctx) => {
    const months = serviceMonths(v.from, v.to);
    if (
      !months.length ||
      months.length !== v.months.length ||
      months.some((m, i) => m !== v.months[i]?.month)
    )
      ctx.addIssue({
        code: "custom",
        message:
          "Informe todos os meses do período em ordem, uma vez cada; limite de 36 meses.",
      });
  });
export type ServicePeriod = z.infer<typeof servicePeriodSchema>;
