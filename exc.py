import pathway as pw
import asyncio
import time
table = pw.io.gdrive.read(
    object_id="13eDgt0YghQU2qlogGrTrXJzfD0h0F2Iw",
    service_user_credentials_file="kdsh-pathway-72c63a387058.json",
    mode="streaming",
    with_metadata=True,
)
print(table)
pw.io.jsonlines.write(table, "test.jsonl")
pw.run()