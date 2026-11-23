import {
  expenseBudgetMonths,
  expenseBudgetCenterAmount,
  expenseBudgetBasis,
} from "./expense-service-periods.js";
import { z } from "zod";
import type { Budget, Expense, Document } from "./types.js";
import type { PurchaseRequest } from "./purchasing.js";
import type { TravelRequest } from "./travel.js";
import type { Subscription } from "./subscriptions.js";
import type { SubscriptionObservation } from "./subscription-evidence.js";
import { anchoredMonth, projectSubscription } from "./subscriptions.js";
import { directoryKey } from "./company.js";
import {
  effectiveExpenseAllocations,
  expenseCenterAmount,
} from "./expense-allocations.js";
import { budgetConsumption, type BudgetPayable } from "./budget-consumption.js";
export const budgetForecastQuerySchema = z
  .object({
    from: z.string().regex(/^20\d{2}-(0[1-9]|1[0-2])$/),
    months: z.coerce.number().int().min(1).max(12).default(3),
    subscriptions: z
      .enum(["true", "false"])
      .default("true")
      .transform((v) => v === "true"),
    travel: z
      .enum(["true", "false"])
      .default("true")
      .transform((v) => v === "true"),
    purchases: z
      .enum(["true", "false"])
      .default("true")
      .transform((v) => v === "true"),
  })
  .strict()
  .refine(
    (v) => anchoredMonth(v.from + "-01", v.months - 1).slice(0, 7) <= "2099-12",
    "O horizonte deve terminar até dezembro de 2099.",
  );
export type BudgetForecastQuery = z.output<typeof budgetForecastQuerySchema>;
export type BudgetForecastOrigin = {
  kind: "expense" | "payable" | "purchase" | "subscription" | "travel";
  id: string;
};
export interface BudgetForecastLine {
  id: string;
  origin: BudgetForecastOrigin;
  version: number;
  label: string;
  status: string;
  layer: "baseline" | "subscription" | "travel" | "purchase-proposal";
  month: string;
  costCenter: string;
  date: string | null;
  amount: number;
  represented: number;
  contribution: number;
  basis: string;
  warnings: string[];
  documents: Pick<Document, "id" | "name" | "sha256">[];
  coverage: {
    origin: BudgetForecastOrigin;
    version: number;
    label: string;
    claimed: number;
    used: number;
    basis: string;
  }[];
}
interface Inputs {
  budgets: Budget[];
  expenses: Expense[];
  purchases: PurchaseRequest[];
  payables: BudgetPayable[];
  subscriptions: Subscription[];
  observations: SubscriptionObservation[];
  travel: TravelRequest[];
}
const total = (values: number[]) => {
  const n = values.reduce((n, v) => n + v, 0);
  if (!Number.isSafeInteger(n))
    throw new Error("O total excede o limite seguro de centavos.");
  return n;
};
const endOfMonth = (month: string) =>
  new Date(
    Date.parse(anchoredMonth(month + "-01", 1) + "T12:00:00Z") - 86_400_000,
  )
    .toISOString()
    .slice(0, 10);
export function buildBudgetForecast(
  query: BudgetForecastQuery,
  input: Inputs,
  asOf: string,
) {
  const months = Array.from({ length: query.months }, (_, i) =>
    anchoredMonth(query.from + "-01", i).slice(0, 7),
  );
  const from = query.from + "-01",
    to = endOfMonth(months.at(-1)!);
  const lines: BudgetForecastLine[] = [];
  const add = (
    line: Omit<
      BudgetForecastLine,
      "represented" | "contribution" | "coverage" | "documents"
    > &
      Partial<Pick<BudgetForecastLine, "coverage" | "documents">>,
  ) => {
    if (months.includes(line.month))
      lines.push({
        ...line,
        represented: 0,
        contribution: line.amount,
        documents: [],
        coverage: [],
        ...line,
      });
  };
  const centerNames = new Map<string, string>();
  const remember = (name: string) =>
    centerNames.set(
      directoryKey(name),
      centerNames.get(directoryKey(name)) ?? name,
    );
  input.budgets
    .filter((b) => months.includes(b.month))
    .forEach((b) => remember(b.costCenter));
  for (const e of input.expenses.filter((e) =>
    ["approved", "submitted", "needs_info"].includes(e.status),
  )) {
    for (const allocation of effectiveExpenseAllocations(e))
      remember(allocation.costCenter);
    const names = [
      ...new Set(
        effectiveExpenseAllocations(e).map((a) => directoryKey(a.costCenter)),
      ),
    ];
    for (const month of expenseBudgetMonths(e))
      for (const key of names) {
        const costCenter = centerNames.get(key)!;
        const amount = expenseBudgetCenterAmount(e, month, costCenter);
        if (!amount) continue;
        add({
          id: `expense:${e.id}:${key}${e.allocations?.some((a) => a.servicePeriod) ? `:${month}` : ""}`,
          origin: { kind: "expense", id: e.id },
          version: e.version,
          label: e.merchant,
          status: e.status,
          layer: "baseline",
          month,
          costCenter,
          date: e.date,
          amount,
          basis: expenseBudgetBasis(e, month, costCenter),
          warnings: e.demo
            ? [
                "Registro fictício com campos declarados; sem extração automática.",
              ]
            : [],
          documents: e.documents.map(({ id, name, sha256 }) => ({
            id,
            name,
            sha256,
          })),
        });
      }
  }
  for (const b of input.payables) {
    remember(b.costCenter);
    for (const part of b.servicePeriod?.months ?? [
      { month: b.month, amount: b.amount },
    ])
      add({
        id: `payable:${b.id}${b.servicePeriod ? `:${part.month}` : ""}`,
        origin: { kind: "payable", id: b.id },
        version: b.version,
        label: `${b.vendor} · ${b.invoiceNumber}`,
        status: b.status,
        layer: "baseline",
        month: part.month,
        costCenter: b.costCenter,
        date: b.issueDate,
        amount: part.amount,
        basis: b.servicePeriod
          ? `Parcela mensal declarada de ${part.amount} centavos, total da conta ${b.amount} centavos. Período ${b.servicePeriod.from} a ${b.servicePeriod.to}. ${b.servicePeriod.basis}. Emissão/vencimento preservados; não é lançamento contábil.`
          : b.monthBasis === "purchase"
            ? "Mês de planejamento preservado/atual da compra; o vencimento não atribui o orçamento."
            : "Mês de emissão do documento; o vencimento não atribui o orçamento.",
        warnings: b.warnings,
        documents: b.documents,
      });
  }

  for (const p of input.purchases) {
    remember(p.costCenter);
    if (query.purchases && ["submitted", "needs_info"].includes(p.status))
      add({
        id: `proposal:${p.id}`,
        origin: { kind: "purchase", id: p.id },
        version: p.version,
        label: `SC-${String(p.number).padStart(4, "0")} · ${p.title}`,
        status: p.status,
        layer: "purchase-proposal",
        month: p.month,
        costCenter: p.costCenter,
        date: null,
        amount: p.amount,
        basis:
          "Escopo solicitado ainda sem autorização final. Não é obrigação nem despesa.",
        warnings:
          p.status === "needs_info"
            ? ["Solicitação em ajuste; depende de correção e nova decisão."]
            : [],
      });
  }
  if (query.subscriptions)
    for (const s of input.subscriptions) {
      for (const projected of projectSubscription(s, from, to)) {
        const r = s.revisions.find((r) => r.number === projected.revision)!;
        const costCenter = r.context.center.name;
        remember(costCenter);
        const coverage: BudgetForecastLine["coverage"] = [];
        const warnings = [...s.contextIssues];
        for (const o of input.observations.filter(
          (o) =>
            o.subscriptionId === s.id &&
            o.billingDate === projected.date &&
            ["accepted", "proposed"].includes(o.state),
        )) {
          if (o.sourceKind === "card-refund") {
            warnings.push(
              "Estorno observado não reduz os valores brutos desta previsão de consumo.",
            );
            continue;
          }
          if (o.state !== "accepted" || !o.current) {
            warnings.push(
              "Há cobrança proposta ou divergente; somente vínculo aceito e atual pode substituir estimativa.",
            );
            continue;
          }
          if (o.sourceKind !== "expense" && o.sourceKind !== "payable") {
            warnings.push(
              "Cobrança de cartão simulada não é uma despesa/conta nesta base; não abatemos a estimativa por inferência.",
            );
            continue;
          }
          coverage.push({
            origin: { kind: o.sourceKind, id: o.sourceId },
            version: 0,
            label: o.source.label,
            claimed: o.allocationAmount ?? o.source.amount,
            used: 0,
            basis: `Ciclo ${o.billingDate}; observação aceita ${o.id}. ${o.allocationBasis ?? "Valor integral da fonte."}`,
          });
        }
        add({
          id: `subscription:${s.id}:${projected.date}`,
          origin: { kind: "subscription", id: s.id },
          version: s.version,
          label: projected.service,
          status: s.state,
          layer: "subscription",
          month: projected.date.slice(0, 7),
          costCenter,
          date: projected.date,
          amount: projected.amount,
          basis: `Termos aprovados na revisão ${r.number}, cobrança na data de recorrência preservada. Sem reajuste/tributo não informado. Referência: ${r.fields.contractReference}.`,
          warnings,
          coverage,
        });
      }
    }
  if (query.travel)
    for (const trip of input.travel.filter((t) =>
      ["submitted", "needs-info", "approved"].includes(t.state),
    )) {
      const fields =
        trip.state === "approved" && trip.approved
          ? trip.approved.fields
          : trip;
      const costCenter = fields.costCenter;
      remember(costCenter);
      const coverage: BudgetForecastLine["coverage"] = [];
      const warnings: string[] = [];
      if (trip.state !== "approved")
        warnings.push(
          "Viagem ainda sem autorização final; estimativa sujeita à revisão.",
        );
      for (const link of trip.expenseLinks.filter((l) =>
        ["accepted", "proposed"].includes(l.state),
      )) {
        if (link.state !== "accepted" || !link.current) {
          warnings.push(
            "Há vínculo de viagem proposto ou divergente; não abatemos a estimativa.",
          );
          continue;
        }
        coverage.push({
          origin: { kind: "expense", id: link.expenseId },
          version: link.source.version,
          label: link.source.merchant,
          claimed:
            link.linkedAmount ?? expenseCenterAmount(link.source, costCenter),
          used: 0,
          basis: `Parcela conferida no vínculo ${link.id}; adiantamento não é gasto adicional.`,
        });
      }
      add({
        id: `travel:${trip.id}`,
        origin: { kind: "travel", id: trip.id },
        version: trip.version,
        label: fields.title,
        status: trip.state,
        layer: "travel",
        month: fields.startsOn.slice(0, 7),
        costCenter,
        date: fields.startsOn,
        amount:
          trip.state === "approved" && trip.approved
            ? trip.approved.amount
            : trip.estimatedTotal,
        basis: `Estimativa integral no mês de início (${fields.startsOn} a ${fields.endsOn}), sem rateio diário implícito. Adiantamentos/devoluções não entram como outra despesa.`,
        warnings,
        coverage,
      });
    }
  // Stable, bounded representation across estimate families. Never spend a source portion twice.
  const used = new Map<string, number>();
  const estimates = lines
    .filter((l) => l.layer !== "baseline")
    .sort((a, b) => a.id.localeCompare(b.id));
  for (const line of estimates)
    for (const claim of line.coverage) {
      const source = lines.find(
        (l) =>
          l.layer === "baseline" &&
          l.origin.kind === claim.origin.kind &&
          l.origin.id === claim.origin.id &&
          l.status === "approved" &&
          l.month === line.month &&
          directoryKey(l.costCenter) === directoryKey(line.costCenter),
      );
      if (!source) {
        line.warnings.push(
          "Fonte vinculada não está aprovada nesta base, centro e mês. A data original de emissão e eventual atribuição mensal declarada são preservadas; não deslocamos o gasto por inferência para abater esta estimativa.",
        );
        continue;
      }
      claim.version = source.version;
      const key = `${source.origin.kind}:${source.origin.id}:${line.month}:${directoryKey(line.costCenter)}`;
      const capacity = Math.max(0, source.amount - (used.get(key) ?? 0));
      claim.used = Math.max(
        0,
        Math.min(claim.claimed, capacity, line.amount - line.represented),
      );
      line.represented += claim.used;
      used.set(key, (used.get(key) ?? 0) + claim.used);
      if (claim.used < claim.claimed && capacity < claim.claimed)
        line.warnings.push(
          "A parcela da fonte já está representada ou excede o saldo declarado. Abatimento limitado; confira vínculos sobrepostos.",
        );
    }
  for (const line of estimates)
    line.contribution = Math.max(0, line.amount - line.represented);
  const rows = months.flatMap((month) => {
    const centers = new Set([
      ...input.budgets
        .filter((b) => b.month === month)
        .map((b) => directoryKey(b.costCenter)),
      ...lines
        .filter((l) => l.month === month)
        .map((l) => directoryKey(l.costCenter)),
      ...input.purchases
        .filter((p) => p.month === month && p.committed > 0)
        .map((p) => directoryKey(p.costCenter)),
    ]);
    return [...centers].sort().map((key) => {
      const costCenter = centerNames.get(key)!;
      const plans = input.budgets.filter(
        (b) => b.month === month && directoryKey(b.costCenter) === key,
      );
      const usage = budgetConsumption(
        { month, costCenter, limit: 0, version: plans[0]?.version ?? 0 },
        input.expenses,
        input.purchases,
        input.payables,
      );
      const sourceLines = lines.filter(
        (l) => l.month === month && directoryKey(l.costCenter) === key,
      );
      for (const p of input.purchases.filter(
        (p) =>
          p.month === month &&
          directoryKey(p.costCenter) === key &&
          p.status === "approved",
      )) {
        const relief =
          total(
            usage.bills
              .filter((b) => b.purchase?.id === p.id)
              .map((b) => b.coveredCommitment),
          ) +
          total(
            usage.reservations
              .filter((r) => r.purchaseId === p.id)
              .map((r) => r.coveredCommitment),
          );
        sourceLines.push({
          id: `purchase:${p.id}`,
          origin: { kind: "purchase", id: p.id },
          version: p.version,
          label: `SC-${String(p.number).padStart(4, "0")} · ${p.title}`,
          status: p.status,
          layer: "baseline",
          month,
          costCenter,
          date: null,
          amount: p.committed,
          represented: Math.min(p.committed, relief),
          contribution: Math.max(0, p.committed - relief),
          basis:
            "Compromisso residual após vínculos aprovados, contas verificadas e reservas atuais. Não é outra despesa nem pagamento.",
          warnings: usage.reservations
            .filter((r) => r.purchaseId === p.id)
            .flatMap((r) => r.warnings),
          documents: [],
          coverage: [],
        });
      }
      const baseline = total([
        usage.approved,
        usage.pending,
        usage.approvedPayables,
        usage.pendingPayables,
        usage.committed,
      ]);
      const byLayer = (layer: BudgetForecastLine["layer"]) =>
        total(
          sourceLines
            .filter((l) => l.layer === layer)
            .map((l) => l.contribution),
        );
      const subscriptionEstimate = byLayer("subscription"),
        travelEstimate = byLayer("travel"),
        purchaseProposals = byLayer("purchase-proposal");
      const estimated = total([
          subscriptionEstimate,
          travelEstimate,
          purchaseProposals,
        ]),
        projected = total([baseline, estimated]);
      const limit = plans.length === 1 ? plans[0].limit : null;
      const warnings = [
        ...(plans.length > 1
          ? [
              "Mais de um limite usa nomes equivalentes neste mês. Não somamos limites ambíguos.",
            ]
          : []),
        ...(!plans.length
          ? [
              "Sem limite definido: saldo e desvio projetados são desconhecidos.",
            ]
          : []),
      ];
      if (
        sourceLines.filter((l) => l.layer !== "baseline" && l.contribution > 0)
          .length > 1 ||
        (estimated > 0 && usage.committed > 0)
      )
        warnings.push(
          "Planos de origens diferentes podem se sobrepor. Sem vínculo explícito, mantemos as estimativas e o compromisso; compare as camadas antes de decidir.",
        );
      return {
        month,
        costCenter,
        limit,
        budgetVersion: plans.length === 1 ? plans[0].version : null,
        baseline,
        approvedExpenses: usage.approved,
        pendingExpenses: usage.pending,
        approvedPayables: usage.approvedPayables,
        pendingPayables: usage.pendingPayables,
        residualCommitments: usage.committed,
        subscriptionEstimate,
        travelEstimate,
        purchaseProposals,
        additionalEstimates: estimated,
        projected,
        baselineRemaining: limit === null ? null : limit - baseline,
        projectedRemaining: limit === null ? null : limit - projected,
        warnings,
        lines: sourceLines.sort((a, b) => a.id.localeCompare(b.id)),
      };
    });
  });
  return {
    asOf,
    query,
    from,
    to,
    rows,
    months,
    method: "known-sources-budget-forecast-v1" as const,
    bankConfirmed: false as const,
    actualSpending: null,
    realizedSavings: null,
    notice:
      "Cenário das fontes locais conhecidas, sem extrapolação estatística/IA. Estimativas opcionais não alteram orçamento, aprovação, despesa ou pagamento. Ausência de registro não prova gasto zero; disponibilidade bancária e economia realizada permanecem desconhecidas.",
  };
}
export type BudgetForecast = ReturnType<typeof buildBudgetForecast>;
