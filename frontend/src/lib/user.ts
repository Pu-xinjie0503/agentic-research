const STORAGE_KEY = "deepsearch.user_id";

export function createUserId(): string {
  if (crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `user-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function getStoredUserId(): string {
  const existing = window.localStorage.getItem(STORAGE_KEY);
  if (existing) {
    return existing;
  }

  const userId = createUserId();
  window.localStorage.setItem(STORAGE_KEY, userId);
  return userId;
}
