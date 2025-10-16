-- No-op: tokens and cost columns already exist in Message model. This migration placeholder ensures versioning for downstream changes.

-- ALTER TABLE "Message" ADD COLUMN "tokens" INTEGER;
-- ALTER TABLE "Message" ADD COLUMN "cost" REAL;