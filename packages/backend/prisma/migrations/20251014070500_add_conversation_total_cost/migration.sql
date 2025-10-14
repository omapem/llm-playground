-- Add totalCost column to Conversation if not exists (SQLite lacks IF NOT EXISTS for alter)
PRAGMA foreign_keys=OFF;

CREATE TABLE "new_Conversation" (
  "id" TEXT NOT NULL PRIMARY KEY,
  "title" TEXT,
  "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" DATETIME NOT NULL,
  "userId" TEXT,
  "totalCost" REAL NOT NULL DEFAULT 0
);
INSERT INTO "new_Conversation" ("id", "title", "createdAt", "updatedAt", "userId", "totalCost")
  SELECT "id", "title", "createdAt", "updatedAt", "userId", COALESCE("totalCost",0) FROM "Conversation";
DROP TABLE "Conversation";
ALTER TABLE "new_Conversation" RENAME TO "Conversation";

PRAGMA foreign_key_check;
PRAGMA foreign_keys=ON;