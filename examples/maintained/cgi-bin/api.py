#!/usr/bin/env python3
import os
import sys
import json
import urllib.parse
import mysql.connector

print("Content-Type: application/json")
print()

query = dict(urllib.parse.parse_qsl(os.getenv("QUERY_STRING", default="")))

action = query.get("action", None)

if action is None:
    print(json.dumps({"error": "Missing 'action' query parameter"}))
    sys.exit(1)

cnx = mysql.connector.connect(
    user="user", passwd="pass", database="db", port=3306,
)
cursor = cnx.cursor()

if action == "dump":
    cursor.execute("SELECT `key`, `maintained` FROM `status`")
    print(json.dumps(dict(cursor)))
elif action == "set":
    cursor.execute(
        "REPLACE INTO status (`key`, `maintained`) VALUES(%s, %s)",
        (query["URL"], query["maintained"],),
    )
    cnx.commit()
    print(json.dumps({"success": True}))
else:
    print(json.dumps({"error": "Unknown action"}))

sys.stdout.flush()
cursor.close()
cnx.close()
