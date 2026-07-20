import { cpSync, rmSync } from "node:fs";

rmSync("public/pagefind", {
  recursive: true,
  force: true,
});

cpSync("dist/pagefind", "public/pagefind", {
  recursive: true,
});