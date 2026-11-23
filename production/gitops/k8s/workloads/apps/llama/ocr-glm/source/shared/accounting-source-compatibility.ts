import type { AccountingSource } from "./accounting.js";
// Gross recognition follows the approved document and card evidence. A change
// only to reimbursement reservations cannot turn it into another gross expense.
// Historical snapshots are retained verbatim; all other facts remain compared.
export function sameRecognitionSource(
  a: AccountingSource,
  b: AccountingSource,
): boolean {
  if (a.token === b.token) return true;
  if (
    a.kind !== "expense" ||
    b.kind !== "expense" ||
    !a.cardFunding ||
    !b.cardFunding ||
    a.cardFunding.reimbursementReserved === b.cardFunding.reimbursementReserved
  )
    return false;
  const stable = (v: unknown): unknown =>
    Array.isArray(v)
      ? v.map(stable)
      : v && typeof v === "object"
        ? Object.fromEntries(
            Object.entries(v)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([k, x]) => [k, stable(x)]),
          )
        : v;
  const normalized = (s: AccountingSource) => {
    const { token, ...rest } = s;
    if (s.kind === "expense" && s.cardFunding) {
      const { reimbursementReserved, ...cardFunding } = s.cardFunding;
      return {
        ...rest,
        cardFunding,
        fundingReference: "card-portions:approved-evidence",
      };
    }
    return rest;
  };
  return (
    JSON.stringify(stable(normalized(a))) ===
    JSON.stringify(stable(normalized(b)))
  );
}
