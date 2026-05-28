For å kjøre frontend inne på klusteret.

1. På c6-4 kjør:
    npm run dev -- --host 0.0.0.0

2. lokalt i terminalen på PC'en din, kjør:
    ssh -L 8000:c0-0:3000 abc123@ificluster.ifi.uit.no (gammel)
    ssh -L 8080:localhost:3000 -J ttr042@ificluster.ifi.uit.no ttr042@c6-4 (ny test)

3. på din lokale PC gå til denne lenken:
    http://localhost:8000
