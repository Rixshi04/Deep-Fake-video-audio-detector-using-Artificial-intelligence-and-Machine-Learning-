export function toast({
  title,
  description,
}: {
  title?: string;
  description?: string;
  variant?: string;
}) {
  console.error([title, description].filter(Boolean).join(": "));
}
