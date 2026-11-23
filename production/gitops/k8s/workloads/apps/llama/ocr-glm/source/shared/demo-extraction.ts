// Hand-authored positions and text from the bundled fictional receipt image.
// This is a UI fixture, never OCR output or evidence from an uploaded file.
export const demoReceiptDocumentId = "demo-receipt";
export const demoReceiptHash = "synthetic-example";
export const demoReceiptFields = [
  {
    id: "demo-merchant",
    field: "merchant",
    value: "Restaurante Exemplo",
    box: { x: 0.14, y: 0.16, width: 0.69, height: 0.06 },
  },
  {
    id: "demo-date",
    field: "date",
    value: "2026-10-21",
    box: { x: 0.14, y: 0.33, width: 0.32, height: 0.045 },
  },
  {
    id: "demo-amount",
    field: "amount",
    value: "8650",
    box: { x: 0.55, y: 0.65, width: 0.31, height: 0.055 },
  },
] as const;
