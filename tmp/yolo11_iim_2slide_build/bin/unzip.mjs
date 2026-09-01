import fs from "node:fs/promises";
import JSZip from "jszip";

const args = process.argv.slice(2);
const mode = args[0];
const archive = args[1];
const entry = args[2];

if (!archive || !["-Z1", "-p"].includes(mode)) {
  process.stderr.write("usage: unzip -Z1 archive | unzip -p archive entry\n");
  process.exit(2);
}

const zip = await JSZip.loadAsync(await fs.readFile(archive));
if (mode === "-Z1") {
  process.stdout.write(Object.keys(zip.files).join("\n") + "\n");
} else {
  const file = zip.file(entry);
  if (!file) {
    process.stderr.write(`missing zip entry: ${entry}\n`);
    process.exit(1);
  }
  process.stdout.write(await file.async("nodebuffer"));
}
