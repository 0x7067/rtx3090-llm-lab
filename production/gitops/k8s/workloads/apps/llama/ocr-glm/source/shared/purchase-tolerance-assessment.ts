import type { BillFields } from "./payables.js";
import type { PricedItem } from "./item-quantities.js";
import {
  itemUnit,
  lineTotal,
  quantityThousandths,
} from "./item-quantities.js";
import type {
  PurchasePriceAssessment,
  PurchaseTolerancePolicyFields,
} from "./purchase-tolerances.js";
import { purchaseMatchLinesSchema } from "./payable-matching.js";

/** Commercial comparison only. It never waives source, receiving or gross-capacity gates. */
export function assessPurchasePriceTolerance(
  bill: BillFields,
  purchaseItems: PricedItem[] | null,
  policy: PurchaseTolerancePolicyFields,
): PurchasePriceAssessment {
  const result: PurchasePriceAssessment = {
    hasPriceVariance: false,
    eligible: false,
    issues: [],
    expectedGross: null,
    declaredGross: bill.amount,
    difference: null,
    absoluteItemDifference: null,
    items: [],
    rates: [],
  };
  const issue = (s: string) => {
    if (!result.issues.includes(s)) result.issues.push(s);
  };
  const parsed = purchaseMatchLinesSchema.safeParse(bill.purchaseMatch?.lines);
  if (!parsed.success || !purchaseItems) {
    issue(
      "Selecione um vínculo completo de itens e recebimentos da compra autorizada.",
    );
    return result;
  }
  const expected = new Map<number, bigint>(),
    quantities = new Map<number, number>();
  const rateKeys = new Set<string>();
  let completeMapping = true;
  for (const line of parsed.data) {
    const declared = bill.items[line.billItemIndex],
      authorized = purchaseItems[line.purchaseItemIndex];
    if (!declared || !authorized) {
      completeMapping = false;
      issue("Confira os índices dos itens da fatura e da compra.");
      continue;
    }
    if (itemUnit(declared) !== itemUnit(authorized)) {
      completeMapping = false;
      issue("Unidades diferentes não são elegíveis para tolerância de preço.");
      continue;
    }
    const q = quantityThousandths(line.quantity);
    quantities.set(
      line.billItemIndex,
      (quantities.get(line.billItemIndex) ?? 0) + q,
    );
    expected.set(
      line.billItemIndex,
      (expected.get(line.billItemIndex) ?? 0n) +
        BigInt(q) * BigInt(authorized.unitPrice),
    );
    if (declared.unitPrice !== authorized.unitPrice)
      result.hasPriceVariance = true;
    const key = `${line.billItemIndex}:${line.purchaseItemIndex}`;
    if (!rateKeys.has(key)) {
      rateKeys.add(key);
      const delta = BigInt(Math.abs(declared.unitPrice - authorized.unitPrice));
      const bps = Number(
        (delta * 10_000n + BigInt(authorized.unitPrice) - 1n) /
          BigInt(authorized.unitPrice),
      );
      result.rates.push({
        billItemIndex: line.billItemIndex,
        purchaseItemIndex: line.purchaseItemIndex,
        expectedUnitPrice: authorized.unitPrice,
        declaredUnitPrice: declared.unitPrice,
        differenceBpsCeiling: bps,
      });
      if (
        delta * 10_000n >
        BigInt(authorized.unitPrice) * BigInt(policy.maximumUnitPriceBps)
      )
        issue(
          "A diferença de preço unitário excede o percentual da política, inclusive em linhas com arredondamento igual.",
        );
    }
  }
  if (
    bill.items.some(
      (item, i) => quantities.get(i) !== quantityThousandths(item.quantity),
    ) ||
    [...quantities.keys()].some((i) => !bill.items[i])
  ) {
    completeMapping = false;
    issue(
      "O vínculo deve cobrir exatamente cada quantidade da fatura; tolerância não admite sobras ou unidades adicionais.",
    );
  }
  if (!completeMapping) return result;
  let total = 0,
    absolute = 0;
  bill.items.forEach((item, i) => {
    const value = Number((expected.get(i)! + 500n) / 1000n),
      declaredGross = lineTotal(item),
      difference = declaredGross - value;
    result.items.push({
      billItemIndex: i,
      expectedGross: value,
      declaredGross,
      difference,
    });
    total += value;
    absolute += Math.abs(difference);
  });
  result.expectedGross = total + bill.freight - bill.discount;
  result.difference = bill.amount - result.expectedGross;
  result.absoluteItemDifference = absolute;
  if (result.expectedGross <= 0)
    issue(
      "O total esperado da fatura deve permanecer positivo após frete e desconto declarados.",
    );
  if (!result.hasPriceVariance)
    issue("Não há divergência de preço unitário; use o vínculo exato normal.");
  if (!policy.enabled)
    issue(
      "A tolerância de preço está desativada; nenhuma divergência é aprovada automaticamente.",
    );
  if (absolute > policy.maximumAbsoluteGross)
    issue(
      "A soma absoluta das diferenças por item excede o limite da política; diferenças opostas não se compensam.",
    );
  result.eligible = result.issues.length === 0;
  return result;
}
